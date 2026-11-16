from pathlib import Path

import pytest

from tanya_bot.telegram_models import GatewayFailure, VoiceResult
from tests.test_telegram_core_support import Rig

pytestmark = pytest.mark.anyio


async def test_buttons_bound_to_message_and_real_memory_journal_actions(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state")
    await rig.say("Question")
    reply = rig.disk.load().replies[-1]
    assert "voice" not in reply.actions
    bad = rig.callback(reply, "journal").model_copy(update={"message_id": 1000})
    await rig.controller.on_callback(bad)
    foreign = rig.callback(reply, "journal").model_copy(update={"sender_id": 7})
    await rig.controller.on_callback(foreign)
    assert rig.disk.load().journal == ()
    remember = rig.callback(reply, "remember")
    await rig.controller.on_callback(remember)
    journal = rig.callback(reply, "journal")
    await rig.controller.on_callback(journal)
    await rig.controller.on_callback(journal)
    await rig.controller.on_callback(rig.callback(reply, "journal"))
    state = rig.disk.load()
    assert [entry.text for entry in state.memories] == [rig.gateway.reply]
    assert [entry.text for entry in state.journal] == [rig.gateway.reply]
    assert state.replies[-1].actions == ("more",)
    assert len(rig.gateway.calls) == 1
    await rig.say(f"/memory delete {state.memories[0].id}")
    await rig.say(f"/journal delete {state.journal[0].id}")
    assert rig.disk.load().memories == ()
    assert rig.disk.load().journal == ()


async def test_settings_buttons_switch_fields_cancel_and_survive_restart(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state")
    await rig.say("/settings")
    menu = rig.disk.load().replies[-1]
    await rig.controller.on_callback(rig.callback(menu, "set_about"))
    assert rig.disk.load().editing == "about"
    await rig.say("/settings user_name")
    await rig.say("Synthetic owner")
    await rig.say("/settings voice cedar")
    await rig.say("/settings voice_replies on")
    await rig.say("/settings boundaries")
    await rig.say("/cancel")
    await rig.say("Question after cancel")
    state = rig.disk.load()
    assert state.settings.profile.user_name == "Synthetic owner"
    assert state.settings.profile.about == ""
    assert state.settings.profile.voice == "cedar"
    assert state.settings.voice_replies
    assert state.editing is None
    assert len(rig.gateway.calls) == 1
    resumed = Rig(tmp_path / "state")
    assert resumed.disk.load().settings == state.settings


async def test_invalid_profile_keeps_pending_field_and_old_value(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state")
    await rig.say("/settings style")
    await rig.say("invalid-style")
    assert rig.disk.load().editing == "style"
    assert rig.disk.load().settings.profile.style == "warm"
    await rig.say("calm")
    assert rig.disk.load().settings.profile.style == "calm"
    assert rig.gateway.calls == []


async def test_more_menu_gmail_truthful_and_local_composing(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "state")
    await rig.say("Question")
    reply = rig.disk.load().replies[-1]
    await rig.controller.on_callback(rig.callback(reply, "more"))
    expanded = rig.disk.load().replies[-1]
    assert "gmail" in expanded.actions
    await rig.controller.on_callback(rig.callback(expanded, "gmail"))
    assert "Gmail is not set up" in rig.outbox.texts[-1]
    await rig.say("/memory add")
    await rig.say("Synthetic explicit fact")
    await rig.say("/journal add")
    await rig.say("Synthetic local note")
    await rig.say("/memory")
    assert "Synthetic explicit fact" in rig.outbox.texts[-1]
    await rig.say("/journal")
    assert "Synthetic local note" in rig.outbox.texts[-1]
    assert len(rig.gateway.calls) == 1


async def test_voice_unknown_never_regenerates_and_recovers_same_artifact(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state", voice=True)
    await rig.say("/settings voice cedar")
    await rig.say("Question")
    reply = rig.disk.load().replies[-1]
    assert "voice" in reply.actions
    event = rig.callback(reply, "voice")
    rig.voice.fail = True
    with pytest.raises(GatewayFailure, match="synthetic_voice_unknown"):
        await rig.controller.on_callback(event)
    await rig.controller.on_callback(event)
    await rig.controller.on_callback(rig.callback(reply, "voice"))
    assert len(rig.voice.calls) == 1
    request = rig.voice.calls[0]
    assert request.voice == "cedar"
    await rig.say("/recover")
    assert rig.outbox.voices == []
    rig.voice.cached = VoiceResult(request_id=request.request_id, artifact_id="cached")
    await rig.say("/recover")
    await rig.say("/recover")
    assert rig.outbox.voices == [rig.voice.cached]
    assert all(identifier == request.request_id for identifier in rig.voice.recoveries)
    assert len(rig.voice.calls) == 1


async def test_automatic_voice_off_on_and_long_reply_hides_memory_button(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state", voice=True)
    await rig.say("Text only")
    assert rig.voice.calls == []
    await rig.say("/settings voice_replies on")
    rig.gateway.reply = "😀" * 4500
    await rig.say("Speak")
    assert rig.voice.calls == []
    assert rig.disk.load().replies[-1].voice_request is None
    assert "voice" not in rig.disk.load().replies[-1].actions
    assert "remember" not in rig.disk.load().replies[-1].actions
    assert rig.outbox.voices == []
    rig.gateway.reply = "Short spoken answer"
    await rig.say("Speak shorter")
    assert len(rig.voice.calls) == 1
    assert len(rig.outbox.voices) == 1
    assert all(len(text.encode("utf-16-le")) // 2 <= 4096 for text in rig.outbox.texts)
