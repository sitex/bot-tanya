"""Transaction-scoped state and durable outgoing message bindings."""

from tanya_memory import invalidate

from tanya_bot.i18n import localize
from tanya_bot.telegram_conversations import voice_pending
from tanya_bot.telegram_keyboards import keyboard, reply_actions
from tanya_bot.telegram_models import (
    ENTRY_CAPACITY,
    EVENT_CAPACITY,
    TELEGRAM_CHUNK,
    SettingField,
    StoredReply,
    TelegramState,
    Transport,
)
from tanya_bot.telegram_store import CoreError, StateStore


class CoreSession:
    """Mutable transaction accumulator; every publication requires a saved snapshot."""

    def __init__(self, store: StateStore, transport: Transport, *, voice: bool) -> None:
        """Load the owner-bound snapshot under the caller's store lock."""
        self.store: StateStore = store
        self.transport: Transport = transport
        self.voice: bool = voice
        self.state: TelegramState = store.load()

    def save(self, state: TelegramState, *, completion: bool = False) -> None:
        """Never update in-memory authority until durable save succeeds."""
        if not completion and (
            state.memories != self.state.memories
            or state.settings != self.state.settings
        ):
            after = {entry.id: entry.text for entry in state.memories}
            removed = any(
                after.get(entry.id) != entry.text for entry in self.state.memories
            )
            context = invalidate(self.state.context)
            state = state.model_copy(update={"context": context})
            if removed:
                state = state.model_copy(
                    update={
                        "history_barrier": tuple(entry.id for entry in state.history),
                    }
                )
        self.store.save(state)
        self.state = state

    def claim(self, update_id: str) -> bool:
        """Never evict deduplication identities; full ledgers fail closed."""
        if update_id in self.state.seen_updates:
            return False
        if len(self.state.seen_updates) >= EVENT_CAPACITY:
            raise CoreError(code="update_capacity")
        self.save(
            self.state.model_copy(
                update={
                    "seen_updates": (*self.state.seen_updates, update_id),
                }
            )
        )
        return True

    def replace(self, reply: StoredReply) -> None:
        """Save a changed action/delivery state against its stable token."""
        self.save(
            self.state.model_copy(
                update={
                    "replies": tuple(
                        reply if old.token == reply.token else old
                        for old in self.state.replies
                    ),
                }
            )
        )

    def append(self, reply: StoredReply) -> None:
        """Reserve the result before attempting Telegram delivery."""
        if len(self.state.replies) >= EVENT_CAPACITY:
            raise CoreError(code="reply_capacity")
        updated = self.state.model_copy(
            update={"replies": (*self.state.replies, reply)}
        )
        if "new_confirm" in reply.actions:
            updated = updated.model_copy(update={"new_confirmation_token": reply.token})
        if reply.setting_field is not None:
            updated = updated.model_copy(update={"settings_menu_token": reply.token})
        self.save(updated)

    async def deliver(self, reply: StoredReply) -> None:
        """Attempt a send once; attach buttons only after saving the returned id."""
        if (
            reply.delivery_attempted
            or reply.receipt_only
            or reply.conversation_id != self.state.conversation_id
        ):
            return
        claimed = reply.model_copy(update={"delivery_attempted": True})
        self.replace(claimed)
        message_id = 0
        for offset in range(0, len(reply.text), TELEGRAM_CHUNK):
            message_id = await self.transport.send_text(
                self.state.owner_id,
                reply.text[offset : offset + TELEGRAM_CHUNK],
                (),
            )
        bound = claimed.model_copy(update={"message_id": message_id})
        self.replace(bound)
        await self.transport.edit_keyboard(
            self.state.owner_id,
            message_id,
            keyboard(bound, self.state.settings.language),
        )

    async def say(
        self,
        text: str,
        actions: tuple[str, ...] = (),
        *,
        setting_field: SettingField | None = None,
    ) -> None:
        """Split long local lists into Telegram-safe plain text messages."""
        text = localize(text, self.state.settings.language)
        await self.say_plain(text, actions, setting_field=setting_field)

    async def say_plain(
        self,
        text: str,
        actions: tuple[str, ...] = (),
        *,
        setting_field: SettingField | None = None,
    ) -> None:
        """Publish untrusted plain text without translating or interpreting content."""
        for offset in range(0, len(text), TELEGRAM_CHUNK):
            reply = StoredReply(
                conversation_id=self.state.conversation_id,
                setting_field=setting_field,
                text=text[offset : offset + TELEGRAM_CHUNK],
                kind="menu",
                actions=actions,
            )
            self.append(reply)
            await self.deliver(reply)

    def available(self, reply: StoredReply) -> StoredReply:
        """Recompute reply actions from actual durable state and capacity."""
        if reply.conversation_id != self.state.conversation_id:
            return reply.model_copy(update={"actions": ()})
        actions = reply_actions(
            reply, voice_available=self.voice and not voice_pending(self.state)
        )
        if len(self.state.memories) >= ENTRY_CAPACITY:
            actions = tuple(action for action in actions if action != "remember")
        if len(self.state.journal) >= ENTRY_CAPACITY:
            actions = tuple(action for action in actions if action != "journal")
        return reply.model_copy(update={"actions": actions})
