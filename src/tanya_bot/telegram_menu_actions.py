"""Persist local menu choices before changing their Telegram keyboards."""

from typing import Literal, assert_never

from tanya_bot.telegram_conversations import confirm_new
from tanya_bot.telegram_core_commands import local_command
from tanya_bot.telegram_core_session import CoreSession
from tanya_bot.telegram_models import StoredReply, TelegramState
from tanya_bot.telegram_setting_choices import manual_field, preset_value


def consumed_menu(state: TelegramState, reply: StoredReply) -> TelegramState:
    """Remove offered actions without removing the saved message identity."""
    return state.model_copy(
        update={
            "replies": tuple(
                item.model_copy(update={"actions": ()})
                if item.token == reply.token
                else item
                for item in state.replies
            )
        }
    )


async def run_command(
    session: CoreSession, command: str, *, consumed: StoredReply | None = None
) -> None:
    """Apply the existing validated command with one saved menu consumption."""
    result = local_command(session.state, command)
    if result is None:
        return
    state = result.state if consumed is None else consumed_menu(result.state, consumed)
    session.save(state)
    if consumed is not None and consumed.message_id is not None:
        await session.transport.edit_keyboard(state.owner_id, consumed.message_id, ())
    await session.say(result.text, result.actions, setting_field=result.setting_field)


async def settings_action(
    session: CoreSession, reply: StoredReply, action: str
) -> bool:
    """Use only offered preset identifiers; manual entry preserves schema validation."""
    if action.startswith("set_"):
        await run_command(session, f"/settings {action.removeprefix('set_')}")
        return True
    value = preset_value(action)
    if value is not None:
        field, text = value
        await run_command(session, f"/settings {field} {text}", consumed=reply)
        return True
    field = manual_field(action)
    if field is None:
        return False
    state = consumed_menu(session.state, reply).model_copy(
        update={"editing": field, "composing": None, "settings_menu_token": None}
    )
    session.save(state)
    if reply.message_id is not None:
        await session.transport.edit_keyboard(state.owner_id, reply.message_id, ())
    await session.say("Send the value as text. /cancel — cancel.")
    return True


async def new_choice(
    session: CoreSession, reply: StoredReply, action: Literal["new_confirm", "new_stay"]
) -> None:
    """Only the current saved confirmation can advance the conversation boundary."""
    match action:
        case "new_confirm":
            updated = confirm_new(session.state, reply)
            if updated is None:
                return
            text = "New conversation started. Memory and settings saved."
        case "new_stay":
            if session.state.new_confirmation_token != reply.token:
                return
            updated = consumed_menu(session.state, reply).model_copy(
                update={"new_confirmation_token": None}
            )
            text = "Current conversation saved."
        case _:
            assert_never(action)
    session.save(updated)
    if reply.message_id is not None:
        await session.transport.edit_keyboard(
            session.state.owner_id, reply.message_id, ()
        )
    await session.say(text)
