"""Actions may mutate only the stored reply selected by its callback binding."""

from typing import Final, Literal, assert_never

from tanya_bot.schemas import REMEMBERED_CHARACTERS
from tanya_bot.telegram_conversations import voice_pending
from tanya_bot.telegram_core_session import CoreSession
from tanya_bot.telegram_core_voice import voice_reply
from tanya_bot.telegram_keyboards import keyboard
from tanya_bot.telegram_menu_actions import new_choice, run_command, settings_action
from tanya_bot.telegram_models import ENTRY_CAPACITY, Entry, StoredReply, VoiceGateway

type Action = Literal[
    "voice",
    "remember",
    "journal",
    "more",
    "back",
    "settings",
    "memory",
    "journal_list",
    "gmail",
    "new_confirm",
    "new_stay",
]
ACTIONS: Final[tuple[Action, ...]] = (
    "voice",
    "remember",
    "journal",
    "more",
    "back",
    "settings",
    "memory",
    "journal_list",
    "gmail",
    "new_confirm",
    "new_stay",
)


def _save_entry(
    session: CoreSession, reply: StoredReply, action: Literal["remember", "journal"]
) -> StoredReply:
    state = session.state
    match action:
        case "remember":
            memories = state.memories
            text = reply.text.strip()
            if len(memories) >= ENTRY_CAPACITY or len(text) > REMEMBERED_CHARACTERS:
                return session.available(reply)
            if all(item.text.casefold() != text.casefold() for item in memories):
                memories = (*memories, Entry(text=text))
            state = state.model_copy(update={"memories": memories})
            reply = reply.model_copy(update={"remembered": True})
        case "journal":
            if len(state.journal) >= ENTRY_CAPACITY:
                return session.available(reply)
            state = state.model_copy(
                update={"journal": (*state.journal, Entry(text=reply.text))}
            )
            reply = reply.model_copy(update={"journaled": True})
        case _:
            assert_never(action)
    reply = session.available(reply)
    session.save(
        state.model_copy(
            update={
                "replies": tuple(
                    reply if old.token == reply.token else old for old in state.replies
                )
            }
        )
    )
    return reply


async def perform_action(
    session: CoreSession, reply: StoredReply, action: str, voice: VoiceGateway | None
) -> None:
    """Persist local effects and availability before editing the visible keyboard."""
    if await settings_action(session, reply, action):
        return
    parsed: Action | None = next((item for item in ACTIONS if item == action), None)
    if parsed is None:
        return
    match parsed:
        case "voice":
            await _voice_action(session, reply, voice)
            return
        case "new_confirm" | "new_stay":
            await new_choice(session, reply, parsed)
            return
        case "remember" | "journal":
            reply = _save_entry(session, reply, parsed)
        case "more":
            reply = reply.model_copy(
                update={
                    "actions": ("settings", "memory", "journal_list", "gmail", "back")
                }
            )
        case "back":
            reply = session.available(reply)
        case "settings" | "memory" | "journal_list" | "gmail":
            command = {
                "settings": "/settings",
                "memory": "/memory",
                "journal_list": "/journal",
                "gmail": "/gmail",
            }[parsed]
            await run_command(session, command)
            return
        case _:
            assert_never(parsed)
    session.replace(reply)
    if reply.message_id is not None:
        await session.transport.edit_keyboard(
            session.state.owner_id,
            reply.message_id,
            keyboard(reply, session.state.settings.language),
        )


async def _voice_action(
    session: CoreSession, reply: StoredReply, voice: VoiceGateway | None
) -> None:
    if (
        voice is not None
        and reply.voice_request is None
        and (not voice_pending(session.state))
    ):
        await voice_reply(session, reply, voice)
