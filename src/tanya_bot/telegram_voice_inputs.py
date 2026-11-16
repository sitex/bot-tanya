"""Reserve voice input identities without ever retrying transcription."""

from uuid import UUID

from tanya_bot.telegram_core_session import CoreSession
from tanya_bot.telegram_models import (
    EVENT_CAPACITY,
    IncomingText,
    IncomingVoice,
    TelegramState,
    VoiceInput,
)
from tanya_bot.telegram_store import CoreError


def reserve_input(
    session: CoreSession, event: IncomingVoice, request_id: UUID
) -> tuple[VoiceInput, bool] | None:
    """Only a fresh identity may generate; duplicates recover in the original chat."""
    if f"text:{event.update_id}" in session.state.seen_updates:
        return None
    previous = next(
        (
            item
            for item in session.state.voice_inputs
            if item.update_id == event.update_id
        ),
        None,
    )
    if previous is not None:
        if previous.request_id != request_id:
            raise CoreError(code="voice_input_conflict")
        if (
            previous.conversation_id != session.state.conversation_id
            or previous.cancelled
        ):
            return None
        return previous, False
    if len(session.state.voice_inputs) >= EVENT_CAPACITY:
        raise CoreError(code="voice_input_capacity")
    reserved = VoiceInput(
        update_id=event.update_id,
        request_id=request_id,
        conversation_id=session.state.conversation_id,
    )
    session.save(
        session.state.model_copy(
            update={
                "voice_inputs": (*session.state.voice_inputs, reserved),
            }
        )
    )
    return reserved, True


def voice_text_allowed(state: TelegramState, event: IncomingText) -> bool:
    """A late transcript must still belong to a live, uncancelled input reservation."""
    if event.conversation_id is None:
        return True
    return event.conversation_id == state.conversation_id and any(
        item.update_id == event.update_id
        and not item.cancelled
        and item.conversation_id == event.conversation_id
        for item in state.voice_inputs
    )
