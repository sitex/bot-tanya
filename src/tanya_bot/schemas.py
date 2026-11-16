"""Immutable public HTTP contracts."""

from typing import Annotated, ClassVar, Final, Literal, Self, assert_never
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError
from tanya_persona import CharacterProfile as CompanionProfile

from tanya_bot.media_models import (
    MAX_PARTS,
    TOTAL_IMAGE_BYTES,
    ImageInput,
    MediaInput,
    TextInput,
    invalid_media,
)

__all__ = (
    "CONTEXT_CHARACTERS",
    "REMEMBERED_CHARACTERS",
    "ChatRequest",
    "ChatResponse",
    "CompanionProfile",
    "ErrorDetail",
    "ErrorResponse",
    "FrozenModel",
    "HealthResponse",
    "MemoryUpdate",
    "Message",
)

CONTEXT_CHARACTERS: Final = 16_000
REMEMBERED_CHARACTERS: Final = 500
type Text = Annotated[str, Field(min_length=1, max_length=CONTEXT_CHARACTERS)]


class FrozenModel(BaseModel):
    """Reject extra fields and keep boundary values immutable."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")


class Message(FrozenModel):
    """One text turn, excluding privileged instruction roles."""

    role: Literal["user", "assistant"]
    content: Annotated[str, Field(max_length=CONTEXT_CHARACTERS)]


class ChatRequest(FrozenModel):
    """Client-owned context, with a bound shared by messages and memories."""

    request_id: UUID
    messages: Annotated[tuple[Message, ...], Field(min_length=1, max_length=32)]
    companion_name: Annotated[str, Field(min_length=1, max_length=80)] = "Tolan"
    language: Annotated[str, Field(min_length=1, max_length=40)] = "ru"
    memories: Annotated[tuple[Text, ...], Field(max_length=32)] = ()
    profile: CompanionProfile | None = None
    capture_memory: Annotated[bool, Field(strict=True)] | None = None
    attachments: (
        Annotated[tuple[MediaInput, ...], Field(max_length=MAX_PARTS)] | None
    ) = None

    @model_validator(mode="after")
    def bound_context(self) -> Self:
        """Include every client-supplied prompt field in the character budget."""
        size = sum(len(message.content) for message in self.messages)
        size += sum(map(len, self.memories))
        size += len(self.companion_name) + len(self.language)
        if self.profile is not None:
            size += self.profile.context_size()
        attached_indices = {part.message_index for part in self.attachments or ()}
        if any(
            not message.content
            and (message.role != "user" or index not in attached_indices)
            for index, message in enumerate(self.messages)
        ):
            raise invalid_media()
        image_bytes = 0
        for part in self.attachments or ():
            if (
                part.message_index >= len(self.messages)
                or self.messages[part.message_index].role != "user"
            ):
                raise invalid_media()
            size += len(part.file_name)
            match part:
                case ImageInput():
                    image_bytes += part.byte_count()
                case TextInput():
                    size += len(part.text)
                case _:
                    assert_never(part)
        if image_bytes > TOTAL_IMAGE_BYTES:
            raise invalid_media()
        if size > CONTEXT_CHARACTERS:
            message = "Context exceeds 16000 characters"
            error_type = "context_limit"
            raise PydanticCustomError(error_type, message)
        return self


class MemoryUpdate(FrozenModel):
    """One bounded remembered fact or an exact existing fact to forget."""

    action: Literal["remember", "forget"]
    text: Text

    @model_validator(mode="after")
    def _bound_remembered_fact(self) -> Self:
        match self.action:
            case "remember":
                if len(self.text) > REMEMBERED_CHARACTERS:
                    error_type = "memory_limit"
                    message = "New facts exceed 500 characters"
                    raise PydanticCustomError(error_type, message)
            case "forget":
                pass
            case _:
                assert_never(self.action)
        return self


class ChatResponse(FrozenModel):
    """Successful reply; also the durable replay receipt."""

    request_id: UUID
    reply: Annotated[str, Field(min_length=1)]
    model: Annotated[str, Field(min_length=1)]
    input_tokens: Annotated[int, Field(ge=0, strict=True)]
    output_tokens: Annotated[int, Field(ge=0, strict=True)]
    memory_updates: Annotated[tuple[MemoryUpdate, ...], Field(max_length=8)] | None = (
        None
    )


class HealthResponse(FrozenModel):
    """Local readiness, without any provider request or balance claim."""

    ready: bool
    model: str
    api_key_configured: bool


class ErrorDetail(FrozenModel):
    """Stable machine code and safe human-readable explanation."""

    code: str
    message: str


class ErrorResponse(FrozenModel):
    """Public error envelope."""

    detail: ErrorDetail
