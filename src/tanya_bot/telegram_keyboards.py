"""Context buttons expose only actions available for this durable message."""

from typing import Final

from tanya_bot.i18n import localize
from tanya_bot.schemas import REMEMBERED_CHARACTERS
from tanya_bot.telegram_models import (
    VOICE_TEXT_CHARACTERS,
    Button,
    Keyboard,
    StoredReply,
)
from tanya_bot.telegram_setting_choices import preset_label
from tanya_bot.telegram_settings import FIELDS, LABELS

ACTION_LABELS: Final = {
    "voice": "Read aloud",
    "remember": "Remember",
    "journal": "To journal",
    "more": "More",
    "settings": "Settings",
    "memory": "Memory",
    "journal_list": "Journal",
    "gmail": "Gmail",
    "gmail_status": "Gmail status",
    "gmail_connect": "Connect Gmail",
    "gmail_list": "List email",
    "gmail_disconnect": "Disconnect Gmail",
    "back": "Back",
    "new_confirm": "Start new",
    "new_stay": "Stay",
}


def reply_actions(reply: StoredReply, *, voice_available: bool) -> tuple[str, ...]:
    """A pending/completed voice action is never offered for paid regeneration."""
    actions: list[str] = []
    if (
        voice_available
        and reply.voice_request is None
        and (len(reply.text) <= VOICE_TEXT_CHARACTERS)
    ):
        actions.append("voice")
    if not reply.remembered and len(reply.text.strip()) <= REMEMBERED_CHARACTERS:
        actions.append("remember")
    if not reply.journaled:
        actions.append("journal")
    actions.append("more")
    return tuple(actions)


def keyboard(reply: StoredReply, language: str = "en") -> Keyboard:
    """Only saved message-bound action identifiers are encoded."""
    buttons: list[Button] = []
    for action in reply.actions:
        label = ACTION_LABELS.get(action)
        if label is None:
            label = preset_label(action)
        if action.startswith("set_"):
            field = action.removeprefix("set_")
            label = next(
                (
                    label
                    for key, label in zip(FIELDS, LABELS, strict=True)
                    if key == field
                ),
                None,
            )
        if label is not None:
            buttons.append(
                Button(text=localize(label, language), data=f"{reply.token}:{action}")
            )
    return tuple(
        tuple(buttons[index : index + 2]) for index in range(0, len(buttons), 2)
    )
