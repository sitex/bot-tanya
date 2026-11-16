"""Core authority precedes the sidecar lock at every external dispatch claim."""

from collections.abc import Generator
from contextlib import contextmanager

from tanya_memory import same_revision

from tanya_bot.dialogue_models import DialogueRoot
from tanya_bot.errors import ServiceError
from tanya_bot.telegram_config import BOT_DIRECTORY, identity
from tanya_bot.telegram_store import StateStore, TelegramStore


def configured_core() -> StateStore:
    """Resolve only the host-provisioned bot identity, after API authentication."""
    config = identity()
    return TelegramStore(BOT_DIRECTORY / "core", config.owner_id)


@contextmanager
def authority(core: StateStore, root: DialogueRoot) -> Generator[None]:
    """Keep cancellation atomic with the claim, releasing both locks before I/O."""
    with core.exclusive():
        state = core.load()
        pending = next(
            (
                item
                for item in state.requests
                if item.request.request_id == root.turn.chat.request_id
            ),
            None,
        )
        if (
            root.cancelled
            or state.owner_id != root.owner_id
            or state.conversation_id != root.conversation_id
            or pending is None
            or pending.cancelled
            or pending.response is not None
            or pending.dialogue_version is None
            or pending.update_id != root.update_id
            or pending.request != root.turn.chat
            or pending.context != root.turn.context
            or not same_revision(state.context, root.turn.context)
        ):
            raise ServiceError(409, "dialogue_cancelled", "Turn authority expired.")
        yield
