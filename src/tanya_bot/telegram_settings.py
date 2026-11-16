"""Bounded profile editing with the public companion schema as authority."""

from typing import Final, assert_never

from pydantic import ValidationError

from tanya_bot.i18n import localize
from tanya_bot.schemas import CompanionProfile
from tanya_bot.telegram_models import SettingField, Settings
from tanya_bot.telegram_store import CoreError

FIELDS: Final[tuple[SettingField, ...]] = (
    "name",
    "user_name",
    "language",
    "style",
    "about",
    "interests",
    "boundaries",
    "voice",
    "voice_replies",
)
LABELS: Final = (
    "Companion's name",
    "Your name",
    "Language",
    "Personality",
    "Character background",
    "Character interests",
    "Boundaries",
    "Voice",
    "Voice replies",
)


def field_named(value: str) -> SettingField:
    """Parse a field identifier without accepting arbitrary model keys."""
    for field in FIELDS:
        if field == value:
            return field
    raise CoreError(code="unknown_setting")


def update_setting(settings: Settings, field: SettingField, text: str) -> Settings:
    """Reject invalid input instead of silently truncating or changing the profile."""
    value = "" if text == "-" else text.strip()
    data = settings.model_dump()
    profile = settings.profile.model_dump()
    match field:
        case "name" | "language":
            data[field] = value
        case "voice_replies":
            if value.casefold() in {"on", "yes", "вкл", "да"}:
                data[field] = True
            elif value.casefold() in {"off", "no", "выкл", "нет"}:
                data[field] = False
            else:
                raise CoreError(code="use_on_off")
        case "user_name" | "style" | "about" | "interests" | "boundaries" | "voice":
            profile[field] = value
        case _:
            assert_never(field)
    try:
        data["profile"] = CompanionProfile.model_validate(profile)
        return Settings.model_validate(data)
    except ValidationError:
        raise CoreError(code="invalid_setting") from None


def settings_text(settings: Settings) -> str:
    """Show actual current values and the available editing commands."""
    profile = settings.profile
    values = (
        settings.name,
        profile.user_name,
        settings.language,
        profile.style,
        profile.about,
        profile.interests,
        profile.boundaries,
        profile.voice,
        localize("on" if settings.voice_replies else "off", settings.language),
    )
    lines = [localize("Settings (/settings field value):", settings.language)]
    lines.extend(
        (
            f"{localize(label, settings.language)} ({field}): {value or '—'}"
            for field, label, value in zip(FIELDS, LABELS, values, strict=True)
        )
    )
    lines.append(
        localize(
            "style: warm / calm / playful / direct. Empty field: -; cancel: /cancel.",
            settings.language,
        )
    )
    return "\n".join(lines)
