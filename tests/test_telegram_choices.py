from typing import Literal, assert_never
from uuid import UUID

import pytest

from tanya_bot.telegram_models import SettingField, Settings
from tanya_bot.telegram_setting_choices import (
    manual_field,
    preset_actions,
    preset_label,
    preset_value,
)
from tanya_bot.telegram_settings import FIELDS, update_setting
from tanya_bot.telegram_store import CoreError


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("style", "warm"),
        ("style", "calm"),
        ("style", "playful"),
        ("style", "direct"),
        ("voice", "alloy"),
        ("voice", "ash"),
        ("voice", "ballad"),
        ("voice", "coral"),
        ("voice", "echo"),
        ("voice", "sage"),
        ("voice", "shimmer"),
        ("voice", "verse"),
        ("voice", "marin"),
        ("voice", "cedar"),
        ("language", "ru"),
        ("language", "en"),
    ],
)
def test_presets_resolve_to_schema_valid_settings(
    field: Literal["style", "voice", "language"], value: str
) -> None:
    action = f"{field}_{value}"
    assert action in preset_actions(field)
    resolved = preset_value(action)
    assert resolved == (field, value)
    assert resolved is not None

    updated = update_setting(Settings(), *resolved)

    match field:
        case "style":
            assert updated.profile.style == value
        case "voice":
            assert updated.profile.voice == value
        case "language":
            assert updated.language == value
        case _:
            assert_never(field)
    assert Settings.model_validate_json(updated.model_dump_json()) == updated
    assert manual_field(action) is None


@pytest.mark.parametrize(
    ("field", "preset_count"),
    [("style", 4), ("voice", 10), ("language", 2), ("voice_replies", 2)],
)
def test_choice_menus_are_bounded_with_one_manual_action(
    field: SettingField, preset_count: int
) -> None:
    actions = preset_actions(field)

    assert len(actions) == preset_count + 1
    assert sum(preset_value(action) is not None for action in actions) == preset_count
    assert [manual_field(action) for action in actions if manual_field(action)] == [
        field
    ]
    assert manual_field(actions[-1]) == field
    assert preset_value(actions[-1]) is None


def test_all_offered_callback_ids_fit_telegram_with_uuid_token() -> None:
    token = UUID("00000000-0000-4000-8000-000000000001").hex
    actions = tuple(action for field in FIELDS for action in preset_actions(field))

    assert len(actions) == len(set(actions))
    for action in actions:
        assert action.isascii()
        assert 1 <= len(action.encode()) <= 31
        assert len(f"{token}:{action}".encode()) <= 64
        assert preset_label(action)


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize(("value", "expected"), [("on", True), ("off", False)])
def test_voice_reply_presets_assign_instead_of_toggle(
    *, initial: bool, value: str, expected: bool
) -> None:
    resolved = preset_value(f"voice_replies_{value}")
    assert resolved == ("voice_replies", value)
    assert resolved is not None
    settings = Settings(voice_replies=initial)

    updated = update_setting(settings, *resolved)
    repeated = update_setting(updated, *resolved)

    assert updated.voice_replies is expected
    assert repeated == updated


@pytest.mark.parametrize(
    ("field", "text"),
    [
        ("name", "Луна"),
        ("user_name", "Анна"),
        ("about", "Учусь рисовать и люблю прогулки"),
        ("interests", "Музыка, книги и путешествия"),
        ("boundaries", "Без советов, пока я не попрошу"),
    ],
)
def test_text_fields_keep_existing_free_form_editing(
    field: SettingField, text: str
) -> None:
    assert preset_actions(field) == ()

    updated = update_setting(Settings(), field, text)

    match field:
        case "name":
            assert updated.name == text
        case "user_name":
            assert updated.profile.user_name == text
        case "about":
            assert updated.profile.about == text
        case "interests":
            assert updated.profile.interests == text
        case "boundaries":
            assert updated.profile.boundaries == text
        case "style" | "voice" | "voice_replies" | "language":
            pytest.fail("Expected a free-text profile field")
        case _:
            assert_never(field)
    assert Settings.model_validate_json(updated.model_dump_json()) == updated


def test_manual_language_accepts_custom_text_and_keeps_schema_limits() -> None:
    field = manual_field("manual_language")
    assert field == "language"
    assert field is not None

    updated = update_setting(Settings(), field, "  Português brasileiro  ")

    assert updated.language == "Português brasileiro"
    assert Settings.model_validate_json(updated.model_dump_json()) == updated
    for invalid in ("-", "x" * 41):
        with pytest.raises(CoreError, match="invalid_setting"):
            _ = update_setting(updated, field, invalid)
    assert updated.language == "Português brasileiro"


@pytest.mark.parametrize(
    "action",
    ["", "style_invalid", "voice_unknown", "language_es", "manual_about", "STYLE_WARM"],
)
def test_unknown_actions_do_not_resolve_arbitrary_settings(action: str) -> None:
    assert preset_label(action) is None
    assert preset_value(action) is None
    assert manual_field(action) is None


@pytest.mark.parametrize(
    ("action", "invalid"), [("manual_style", "invented"), ("manual_voice", "invented")]
)
def test_manual_choice_entry_does_not_bypass_profile_schema(
    action: str, invalid: str
) -> None:
    field = manual_field(action)
    assert field is not None
    settings = Settings()

    with pytest.raises(CoreError, match="invalid_setting"):
        _ = update_setting(settings, field, invalid)

    assert settings == Settings()
