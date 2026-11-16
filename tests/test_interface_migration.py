from pathlib import Path

import pytest

from tanya_bot.i18n import localize
from tanya_bot.telegram_models import Settings
from tanya_bot.telegram_settings import update_setting
from tanya_bot.telegram_store import TelegramStore
from tests.test_telegram_core_support import Rig


def test_fresh_english_and_legacy_omitted_language_preserve_profile(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "state"
    store = TelegramStore(directory, 42)
    with store.exclusive():
        fresh = store.load()
        assert fresh.settings.name == "Tanya"
        assert fresh.settings.language == "en"
        path = directory / "state.json"
        legacy = (
            '{"version":1,"owner_id":42,"settings":'
            '{"name":"Старое имя","profile":{"voice":"cedar"}}}'
        )
        _ = path.write_text(legacy, encoding="utf-8")
        path.chmod(0o600)
    with store.exclusive():
        restored = store.load()
        store.save(restored)
    assert restored.settings.name == "Старое имя"
    assert restored.settings.language == "ru"
    assert restored.settings.profile.voice == "cedar"
    assert store.load().settings == restored.settings


@pytest.mark.parametrize(
    ("value", "enabled"),
    [
        ("вкл", True),
        ("да", True),
        ("выкл", False),
        ("нет", False),
    ],
)
def test_russian_voice_aliases(value: str, *, enabled: bool) -> None:
    settings = Settings(voice_replies=not enabled)
    assert update_setting(settings, "voice_replies", value).voice_replies is enabled


@pytest.mark.anyio
async def test_russian_menu_callback_binding_and_untranslated_ai_reply(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state")
    await rig.say("/settings language ru")
    await rig.say("/settings")
    menu = rig.disk.load().replies[-1]
    assert menu.message_id is not None
    labels = {
        button.data: button.text
        for row in rig.outbox.keyboards[menu.message_id]
        for button in row
    }
    assert labels[f"{menu.token}:set_about"] == "\u041e персонаже"
    assert labels[f"{menu.token}:set_interests"] == "Интересы персонажа"
    await rig.controller.on_callback(rig.callback(menu, "set_about"))
    assert rig.outbox.acks[-1] == "Принято."
    assert "персонажа" in rig.outbox.texts[-1]
    await rig.say("/cancel")
    rig.gateway.reply = "Saved."
    await rig.say("Question")
    assert rig.outbox.texts[-1] == "Saved."
    assert localize("Input error: entry_not_found. /cancel — cancel.", "ru") == (
        "Ошибка ввода: entry_not_found. /cancel — отмена."
    )
