from pathlib import Path
from typing import assert_never

import pytest

from tanya_bot.telegram_setting_choices import PresetField
from tests.test_telegram_core_support import Rig

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize(
    ("field", "action", "expected"),
    [
        ("style", "style_calm", "calm"),
        ("voice", "voice_cedar", "cedar"),
        ("language", "language_en", "en"),
        ("voice_replies", "voice_replies_on", True),
    ],
)
async def test_preset_is_message_bound_consumed_and_durable(
    tmp_path: Path, field: PresetField, action: str, *, expected: str | bool
) -> None:
    rig = Rig(tmp_path / "state")
    await rig.say("/settings")
    root = rig.disk.load().replies[-1]
    await rig.controller.on_callback(rig.callback(root, f"set_{field}"))
    choices = rig.disk.load().replies[-1]
    assert choices.setting_field == field
    assert action in choices.actions
    await rig.controller.on_callback(rig.callback(choices, action))
    state = rig.disk.load()
    match field:
        case "style":
            assert state.settings.profile.style == expected
        case "voice":
            assert state.settings.profile.voice == expected
        case "language":
            assert state.settings.language == expected
        case "voice_replies":
            assert state.settings.voice_replies == expected
        case _:
            assert_never(field)
    assert state.editing is None
    assert state.settings_menu_token is None
    assert (
        next(item for item in state.replies if item.token == choices.token).actions
        == ()
    )
    assert choices.message_id is not None
    assert rig.outbox.keyboards[choices.message_id] == ()
    await rig.controller.on_callback(rig.callback(choices, action))
    assert rig.disk.load().settings == state.settings
    assert Rig(tmp_path / "state").disk.load().settings == state.settings
    assert rig.gateway.calls == []
    assert rig.voice.calls == []


async def test_old_menu_after_field_switch_or_new_conversation_is_inert(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state")
    await rig.say("/settings style")
    old = rig.disk.load().replies[-1]
    await rig.say("/settings voice")
    current = rig.disk.load().replies[-1]
    await rig.controller.on_callback(rig.callback(old, "style_direct"))
    assert rig.disk.load().settings.profile.style == "warm"
    assert rig.disk.load().editing == "voice"
    await rig.say("/new")
    confirmation = rig.disk.load().replies[-1]
    await rig.controller.on_callback(rig.callback(confirmation, "new_confirm"))
    await rig.controller.on_callback(rig.callback(current, "voice_cedar"))
    assert rig.disk.load().settings.profile.voice == "marin"
    assert rig.disk.load().conversation_id == 1
    assert rig.gateway.calls == []


async def test_manual_language_remains_free_and_invalid_input_keeps_old_value(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state")
    await rig.say("/settings language")
    menu = rig.disk.load().replies[-1]
    await rig.controller.on_callback(rig.callback(menu, "manual_language"))
    assert rig.disk.load().editing == "language"
    assert rig.disk.load().replies[-1].actions == ()
    await rig.say("x" * 41)
    assert rig.disk.load().settings.language == "en"
    assert rig.disk.load().editing == "language"
    await rig.say("Hindi")
    assert rig.disk.load().settings.language == "Hindi"
    assert rig.disk.load().editing is None
    await rig.controller.on_callback(rig.callback(menu, "language_en"))
    assert rig.disk.load().settings.language == "Hindi"
    assert rig.gateway.calls == []


async def test_forged_preset_and_cancelled_menu_cannot_change_profile(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state")
    await rig.say("/settings style")
    menu = rig.disk.load().replies[-1]
    await rig.controller.on_callback(rig.callback(menu, "voice_cedar"))
    assert rig.disk.load().settings.profile.voice == "marin"
    await rig.say("/cancel")
    await rig.controller.on_callback(rig.callback(menu, "style_direct"))
    assert rig.disk.load().settings.profile.style == "warm"
    assert rig.gateway.calls == []
