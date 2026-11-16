"""Build real Responses content while retaining the legacy plain-text wire."""

from typing import Final, Literal, TypedDict, assert_never

from tanya_bot.errors import ServiceError
from tanya_bot.media_models import ImageInput, TextInput
from tanya_bot.schemas import ChatRequest

PLAIN_BODY_BYTES: Final = 128_000


def enforce_plain_body_limit(body: bytearray, request: ChatRequest | None) -> None:
    """Only parsed, validated inline images qualify for the larger wire budget."""
    has_images = request is not None and any(
        isinstance(part, ImageInput) for part in request.attachments or ()
    )
    if len(body) > PLAIN_BODY_BYTES and not has_images:
        raise ServiceError(413, "request_too_large", "Request body too large.")


class InputText(TypedDict):
    """Responses text content part."""

    type: Literal["input_text"]
    text: str


class InputImage(TypedDict):
    """Responses inline image content part."""

    type: Literal["input_image"]
    image_url: str
    detail: Literal["auto"]


class InputMessage(TypedDict):
    """A legacy message or a user message with structured content."""

    role: Literal["user", "assistant"]
    content: str | list[InputText | InputImage]


def response_input(request: ChatRequest) -> list[InputMessage]:
    """Attach only to validated user indices, preserving original turn order."""
    result: list[InputMessage] = []
    for index, message in enumerate(request.messages):
        attachments = tuple(
            part for part in request.attachments or () if part.message_index == index
        )
        if not attachments:
            result.append({"role": message.role, "content": message.content})
            continue
        content: list[InputText | InputImage] = []
        if message.content:
            content.append({"type": "input_text", "text": message.content})
        for part in attachments:
            content.append({"type": "input_text", "text": part.file_name})
            match part:
                case ImageInput():
                    content.append(
                        {
                            "type": "input_image",
                            "image_url": part.data_url,
                            "detail": "auto",
                        }
                    )
                case TextInput():
                    content.append({"type": "input_text", "text": part.text})
                case _:
                    assert_never(part)
        result.append({"role": message.role, "content": content})
    return result
