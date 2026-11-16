"""Separate new conversations while retaining every old request identity."""

from tanya_memory import DerivedSummary

from tanya_bot.telegram_models import StoredReply, TelegramState


def voice_pending(state: TelegramState) -> bool:
    """Only a current conversation hold blocks a fresh voice reservation."""
    return any(
        item.conversation_id == state.conversation_id
        and item.voice_request is not None
        and item.voice_result is None
        for item in state.replies
    )


def confirm_new(state: TelegramState, reply: StoredReply) -> TelegramState | None:
    """Consume the current confirmation atomically; keep archive and memories."""
    if (
        reply.token != state.new_confirmation_token
        or reply.conversation_id != state.conversation_id
    ):
        return None
    return state.model_copy(
        update={
            "context": state.context.model_copy(
                update={
                    "epoch": state.context.epoch + 1,
                    "summary": DerivedSummary(),
                }
            ),
            "conversation_id": state.conversation_id + 1,
            "new_confirmation_token": None,
            "settings_menu_token": None,
            "editing": None,
            "composing": None,
            "history": (),
            "history_roles": (),
            "history_barrier": (),
            "requests": tuple(
                item.model_copy(update={"cancelled": True})
                if item.response is None
                else item
                for item in state.requests
            ),
            "replies": tuple(
                item.model_copy(
                    update={
                        "actions": (),
                        "voice_cancelled": item.voice_cancelled
                        or (
                            item.voice_request is not None and item.voice_result is None
                        ),
                    }
                )
                for item in state.replies
            ),
        }
    )
