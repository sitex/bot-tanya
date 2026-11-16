"""Authenticated transport events, independent of conversation persistence."""

from typing import Annotated

from pydantic import Field

from tanya_bot.schemas import FrozenModel

type PositiveId = Annotated[int, Field(gt=0, strict=True)]
type ConversationId = Annotated[int, Field(ge=0, strict=True)]
type EventId = Annotated[str, Field(min_length=1, max_length=160, strict=True)]


class IncomingText(FrozenModel):
    """A stable transport identity is retained for transcript replay protection."""

    update_id: EventId
    conversation_id: ConversationId | None = None
    sender_id: PositiveId
    chat_id: PositiveId
    private: Annotated[bool, Field(strict=True)]
    text: Annotated[str, Field(min_length=1, max_length=16_000, strict=True)]


class IncomingVoice(FrozenModel):
    """Gate media before download or billable transcription."""

    update_id: EventId
    sender_id: PositiveId
    chat_id: PositiveId
    private: Annotated[bool, Field(strict=True)]


class IncomingCallback(FrozenModel):
    """The callback binds to both message identity and authenticated owner."""

    update_id: EventId
    sender_id: PositiveId
    chat_id: PositiveId
    private: Annotated[bool, Field(strict=True)]
    callback_id: EventId
    message_id: PositiveId
    data: Annotated[str, Field(min_length=1, max_length=64, strict=True)]
