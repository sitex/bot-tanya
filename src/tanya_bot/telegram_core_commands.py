"""Local settings, memory and journal commands; never call a provider."""

from dataclasses import dataclass
from typing import Final, Literal, assert_never
from uuid import UUID

from tanya_bot.schemas import CONTEXT_CHARACTERS, REMEMBERED_CHARACTERS
from tanya_bot.telegram_models import ENTRY_CAPACITY, Entry, SettingField, TelegramState
from tanya_bot.telegram_setting_choices import field_prompt, preset_actions
from tanya_bot.telegram_settings import (
    FIELDS,
    field_named,
    settings_text,
    update_setting,
)
from tanya_bot.telegram_store import CoreError

HELP: Final = (
    "/settings — personality, name, language, and voice\n"
    "/memory — memory; /memory add text; /memory delete UUID\n"
    "/journal — journal; /journal add text; /journal delete UUID\n"
    "/gmail — Gmail status\n/recover — check the saved result\n"
    "/new — new conversation after confirmation\n"
    "/cancel — cancel field input; the submitted request remains under review."
)
GMAIL: Final = "Gmail is not set up. Access will be limited to your personal mailbox."
NEW_PROMPT: Final = (
    "Start an independent conversation? Its history will be separate; "
    "memory, settings, and journal will be preserved. Old unknown requests "
    "will remain in the archive and will not be resubmitted."
)
type SlashCommand = Literal[
    "/start", "/help", "/cancel", "/gmail", "/memory", "/journal", "/settings", "/new"
]
COMMANDS: Final[tuple[SlashCommand, ...]] = (
    "/start",
    "/help",
    "/cancel",
    "/gmail",
    "/memory",
    "/journal",
    "/settings",
    "/new",
)
ENTRY_VERBS: Final[tuple[Literal["add", "delete", ""], ...]] = ("add", "delete", "")


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Local mutation and response are committed together before delivery."""

    state: TelegramState
    text: str
    actions: tuple[str, ...] = ()
    setting_field: SettingField | None = None


def _entries(state: TelegramState, command: str, tail: str) -> CommandResult:
    field = "memories" if command == "/memory" else "journal"
    entries = state.memories if command == "/memory" else state.journal
    raw_verb, _, value = tail.partition(" ")
    verb: Literal["add", "delete", ""] | None = next(
        (item for item in ENTRY_VERBS if item == raw_verb), None
    )
    if verb is None:
        raise CoreError(code="unknown_entry_command")
    value = value.strip()
    match verb:
        case "add":
            if not value:
                composing = "memory" if command == "/memory" else "journal"
                return CommandResult(
                    state.model_copy(update={"composing": composing, "editing": None}),
                    "Enter text. /cancel — cancel.",
                )
            limit = (
                REMEMBERED_CHARACTERS if command == "/memory" else CONTEXT_CHARACTERS
            )
            if len(value) > limit or len(entries) >= ENTRY_CAPACITY:
                raise CoreError(code="entry_capacity")
            if all(entry.text.casefold() != value.casefold() for entry in entries):
                entries = (*entries, Entry(text=value))
            return CommandResult(state.model_copy(update={field: entries}), "Saved.")
        case "delete":
            try:
                identifier = UUID(value)
            except ValueError:
                raise CoreError(code="invalid_entry_id") from None
            filtered = tuple(entry for entry in entries if entry.id != identifier)
            if len(filtered) == len(entries):
                raise CoreError(code="entry_not_found")
            return CommandResult(state.model_copy(update={field: filtered}), "Deleted.")
        case "":
            lines = [f"{entry.id}: {entry.text}" for entry in entries]
            return CommandResult(state, "\n".join(lines) or "Empty for now.")
        case _:
            assert_never(verb)


def _settings_command(state: TelegramState, tail: str) -> CommandResult:
    if not tail.strip():
        return CommandResult(
            state.model_copy(update={"settings_menu_token": None, "editing": None}),
            settings_text(state.settings),
            tuple(f"set_{field}" for field in FIELDS),
        )
    name, separator, value = tail.strip().partition(" ")
    field = field_named(name)
    if not separator:
        return CommandResult(
            state.model_copy(
                update={
                    "editing": field,
                    "composing": None,
                    "settings_menu_token": None,
                }
            ),
            field_prompt(field),
            preset_actions(field),
            field if preset_actions(field) else None,
        )
    updated = update_setting(state.settings, field, value)
    return CommandResult(
        state.model_copy(
            update={
                "settings": updated,
                "editing": None,
                "composing": None,
                "settings_menu_token": None,
            }
        ),
        "Setting saved.",
    )


def _slash_command(
    state: TelegramState, command: SlashCommand, tail: str
) -> CommandResult:
    match command:
        case "/start" | "/help":
            return CommandResult(
                state, HELP, ("settings", "memory", "journal_list", "gmail")
            )
        case "/cancel":
            return CommandResult(
                cancelled_state(state),
                "Input and waiting canceled. Unknown requests remain under review.",
            )
        case "/new":
            return CommandResult(
                state.model_copy(update={"new_confirmation_token": None}),
                NEW_PROMPT,
                ("new_confirm", "new_stay"),
            )
        case "/gmail":
            return CommandResult(state, GMAIL)
        case "/memory" | "/journal":
            return _entries(state, command, tail)
        case "/settings":
            return _settings_command(state, tail)
        case _:
            assert_never(command)


def local_command(state: TelegramState, text: str) -> CommandResult | None:
    """Route slash commands before conversational or pending-field input."""
    command, _, tail = text.strip().partition(" ")
    if command.startswith("/"):
        parsed: SlashCommand | None = next(
            (item for item in COMMANDS if item == command), None
        )
        return (
            CommandResult(state, HELP)
            if parsed is None
            else _slash_command(state, parsed, tail)
        )
    if state.editing is not None:
        updated = update_setting(state.settings, state.editing, text)
        return CommandResult(
            state.model_copy(
                update={
                    "settings": updated,
                    "editing": None,
                    "settings_menu_token": None,
                }
            ),
            "Setting saved.",
        )
    if state.composing is not None:
        command = "/memory" if state.composing == "memory" else "/journal"
        result = _entries(state, command, f"add {text}")
        return CommandResult(
            result.state.model_copy(update={"composing": None}), result.text
        )
    return None


def cancelled_state(state: TelegramState) -> TelegramState:
    """Durable tombstones suppress late results; they never authorize a resend."""
    return state.model_copy(
        update={
            "context": state.context.model_copy(
                update={"epoch": state.context.epoch + 1}
            ),
            "editing": None,
            "composing": None,
            "new_confirmation_token": None,
            "settings_menu_token": None,
            "voice_inputs": tuple(
                item.model_copy(update={"cancelled": True})
                if f"text:{item.update_id}" not in state.seen_updates
                else item
                for item in state.voice_inputs
            ),
            "requests": tuple(
                item.model_copy(update={"cancelled": True})
                if item.response is None
                else item
                for item in state.requests
            ),
            "replies": tuple(
                item.model_copy(update={"voice_cancelled": True})
                if item.voice_request is not None
                and (not item.voice_delivery_attempted)
                else item
                for item in state.replies
            ),
        }
    )
