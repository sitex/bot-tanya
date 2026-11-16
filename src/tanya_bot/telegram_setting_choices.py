"""Local setting choices; callers validate values and bind menu callbacks."""

from dataclasses import dataclass
from typing import Final, Literal, assert_never

from tanya_bot.telegram_models import SettingField

type PresetField = Literal["style", "voice", "language", "voice_replies"]


@dataclass(frozen=True, slots=True)
class _Choice:
    """One local action; a missing value requests existing manual editing."""

    action: str
    field: PresetField
    label: str
    value: str | None


_CHOICES: Final[tuple[_Choice, ...]] = (
    _Choice("style_warm", "style", "Warm", "warm"),
    _Choice("style_calm", "style", "Calm", "calm"),
    _Choice("style_playful", "style", "Playful", "playful"),
    _Choice("style_direct", "style", "Direct", "direct"),
    _Choice("manual_style", "style", "Enter manually", None),
    _Choice("voice_alloy", "voice", "Alloy", "alloy"),
    _Choice("voice_ash", "voice", "Ash", "ash"),
    _Choice("voice_ballad", "voice", "Ballad", "ballad"),
    _Choice("voice_coral", "voice", "Coral", "coral"),
    _Choice("voice_echo", "voice", "Echo", "echo"),
    _Choice("voice_sage", "voice", "Sage", "sage"),
    _Choice("voice_shimmer", "voice", "Shimmer", "shimmer"),
    _Choice("voice_verse", "voice", "Verse", "verse"),
    _Choice("voice_marin", "voice", "Marin", "marin"),
    _Choice("voice_cedar", "voice", "Cedar", "cedar"),
    _Choice("manual_voice", "voice", "Enter manually", None),
    _Choice("language_ru", "language", "Russian", "ru"),
    _Choice("language_en", "language", "English", "en"),
    _Choice("manual_language", "language", "Other language", None),
    _Choice("voice_replies_on", "voice_replies", "Turn on", "on"),
    _Choice("voice_replies_off", "voice_replies", "Turn off", "off"),
    _Choice("manual_voice_replies", "voice_replies", "Enter manually", None),
)


def preset_actions(field: SettingField) -> tuple[str, ...]:
    """Offer bounded preset IDs and manual input, or no choices for text fields."""
    return tuple(choice.action for choice in _CHOICES if choice.field == field)


def preset_label(action: str) -> str | None:
    """Resolve the visible label for a known preset or manual-input action."""
    return next((choice.label for choice in _CHOICES if choice.action == action), None)


def preset_value(action: str) -> tuple[SettingField, str] | None:
    """Resolve preset input for update_setting; manual/unknown IDs have no value."""
    return next(
        (
            (choice.field, choice.value)
            for choice in _CHOICES
            if choice.action == action and choice.value is not None
        ),
        None,
    )


def manual_field(action: str) -> SettingField | None:
    """Resolve only explicit manual-input IDs, never arbitrary field suffixes."""
    return next(
        (
            choice.field
            for choice in _CHOICES
            if choice.action == action and choice.value is None
        ),
        None,
    )


def field_prompt(field: SettingField) -> str:
    """Explain existing manual validation without limiting free-text fields."""
    match field:
        case "name":
            prompt = "Enter the companion's name (1-80 characters)."
        case "user_name":
            prompt = "Enter your name (up to 80 characters)."
        case "language":
            prompt = "Choose a language or enter another one (up to 40 characters)."
        case "style":
            prompt = "Choose a personality or enter: warm, calm, playful, direct."
        case "about":
            prompt = "Describe the companion you want (up to 2000 characters)."
        case "interests":
            prompt = "Enter the companion's interests (up to 1000 characters)."
        case "boundaries":
            prompt = "Describe your boundaries (up to 1000 characters)."
        case "voice":
            prompt = (
                "Choose a voice or enter: alloy, ash, ballad, coral, echo, sage, "
                "shimmer, verse, marin, cedar."
            )
        case "voice_replies":
            prompt = "Turn voice replies on or off; manually: on or off."
        case _:
            assert_never(field)
    instructions = "Clear field: - (if allowed); cancel: /cancel."
    return f"{prompt}\n{instructions}"
