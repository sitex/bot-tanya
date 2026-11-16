"""Bounded, local-only attachment wire values; never dereference a URL."""

import base64
import binascii
import warnings
from io import BytesIO
from typing import Annotated, ClassVar, Final, Literal, Self

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

IMAGE_BYTES: Final = 1_048_576
TOTAL_IMAGE_BYTES: Final = 4 * IMAGE_BYTES
MEDIA_BODY_BYTES: Final = 6 * IMAGE_BYTES
MAX_PARTS: Final = 8
MAX_PIXELS: Final = 1280
type FileName = Annotated[str, Field(min_length=1, max_length=160, strict=True)]


def invalid_media() -> PydanticCustomError:
    """Return a safe error with no submitted data or file paths."""
    return PydanticCustomError("invalid_media", "Invalid or oversized attachment")


def validate_image(data: bytes, image_format: Literal["JPEG", "PNG"]) -> None:
    """Check size before pixel allocation, verify integrity, then decode real pixels."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with (
                BytesIO(data) as source,
                Image.open(source, formats=(image_format,)) as image,
            ):
                width, height = image.size
                if not (0 < width <= MAX_PIXELS and 0 < height <= MAX_PIXELS):
                    raise invalid_media()
                image.verify()
            with (
                BytesIO(data) as source,
                Image.open(source, formats=(image_format,)) as image,
            ):
                _ = image.load()
    except (
        OSError,
        IndexError,
        ValueError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        raise invalid_media() from None


class MediaBase(BaseModel):
    """Common strict immutable attachment fields."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    message_index: Annotated[int, Field(ge=0, strict=True)]
    file_name: FileName

    @field_validator("file_name")
    @classmethod
    def safe_name(cls, value: str) -> str:
        """Names are labels, never paths or control sequences."""
        if not value.strip() or any(
            character in "/\\" or not character.isprintable() for character in value
        ):
            raise invalid_media()
        return value


class ImageInput(MediaBase):
    """One normalized image, carried inline rather than fetched remotely."""

    kind: Literal["image"]
    data_url: Annotated[
        str, Field(max_length=4 * ((IMAGE_BYTES + 2) // 3) + 32, strict=True)
    ]

    @model_validator(mode="after")
    def image_data(self) -> Self:
        """Reject bad MIME/base64 and require bounded, fully decodable pixels."""
        header, separator, encoded = self.data_url.partition(",")
        if not separator or header not in {
            "data:image/jpeg;base64",
            "data:image/png;base64",
        }:
            raise invalid_media()
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            raise invalid_media() from None
        if (
            not decoded
            or len(decoded) > IMAGE_BYTES
            or base64.b64encode(decoded).decode("ascii") != encoded
        ):
            raise invalid_media()
        image_format = "JPEG" if header == "data:image/jpeg;base64" else "PNG"
        validate_image(decoded, image_format)
        return self

    def byte_count(self) -> int:
        """Count already-validated base64 bytes without allocating again."""
        encoded = self.data_url.partition(",")[2]
        return len(encoded) // 4 * 3 - (len(encoded) - len(encoded.rstrip("=")))


class TextInput(MediaBase):
    """Actual extracted document text or a literal link, never a placeholder."""

    kind: Literal["text"]
    text: Annotated[str, Field(min_length=1, max_length=16_000, strict=True)]

    @field_validator("text")
    @classmethod
    def meaningful_text(cls, value: str) -> str:
        """Whitespace-only documents have no model-readable content."""
        if not value.strip() or "\x00" in value:
            raise invalid_media()
        return value


type MediaInput = Annotated[ImageInput | TextInput, Field(discriminator="kind")]
