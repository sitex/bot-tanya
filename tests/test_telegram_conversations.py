from pathlib import Path

import pytest

from tanya_bot.schemas import MemoryUpdate
from tanya_bot.telegram_models import GatewayFailure
from tests.test_telegram_core_support import Rig

pytestmark = pytest.mark.anyio


async def new_conversation(rig: Rig) -> None:
    await rig.say("/new")
    prompt = rig.disk.load().replies[-1]
    await rig.controller.on_callback(rig.callback(prompt, "new_confirm"))


async def test_new_requires_current_confirmation_and_preserves_personal_state(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state")
    await rig.say("/settings name Tanya")
    await rig.say("/memory add Keep this")
    await rig.say("/journal add Existing note")
    await rig.say("First conversation")
    before = rig.disk.load()
    await rig.say("/new")
    first = rig.disk.load().replies[-1]
    assert rig.disk.load().history == before.history
    await rig.say("/new")
    current = rig.disk.load().replies[-1]
    await rig.controller.on_callback(rig.callback(first, "new_confirm"))
    await rig.say("No confirmation yet")
    assert len(rig.gateway.calls) == 1
    assert rig.disk.load().conversation_id == 0
    confirmation = rig.callback(current, "new_confirm")
    await rig.controller.on_callback(confirmation)
    await rig.controller.on_callback(confirmation)
    await rig.controller.on_callback(rig.callback(current, "new_confirm"))
    after = rig.disk.load()
    assert after.conversation_id == 1
    assert after.history == ()
    assert after.settings == before.settings
    assert after.memories == before.memories
    assert after.journal == before.journal
    assert after.requests == before.requests
    assert set(before.seen_updates) <= set(after.seen_updates)
    assert len(rig.gateway.calls) == 1


@pytest.mark.parametrize("stay", ["new_stay", "/cancel"])
async def test_stay_or_cancel_invalidates_confirmation_without_erasing_history(
    tmp_path: Path, stay: str
) -> None:
    rig = Rig(tmp_path / "state")
    await rig.say("Existing conversation")
    original = rig.disk.load().history
    await rig.say("/new")
    prompt = rig.disk.load().replies[-1]
    if stay == "/cancel":
        await rig.say(stay)
    else:
        await rig.controller.on_callback(rig.callback(prompt, stay))
    await rig.controller.on_callback(rig.callback(prompt, "new_confirm"))
    assert rig.disk.load().conversation_id == 0
    assert rig.disk.load().history == original
    assert rig.disk.load().new_confirmation_token is None


async def test_new_chat_after_unknown_never_resends_old_request(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "state")
    rig.gateway.fail = True
    with pytest.raises(GatewayFailure):
        await rig.say("Unknown old question")
    old = rig.disk.load().requests[0]
    await new_conversation(rig)
    assert len(rig.gateway.calls) == 1
    rig.gateway.fail = False
    await rig.say("Independent new question")
    state = rig.disk.load()
    assert len(rig.gateway.calls) == 2
    assert state.requests[0].request == old.request
    assert state.requests[0].response is None
    assert state.requests[0].cancelled
    assert state.requests[1].conversation_id == 1
    assert state.requests[1].request.request_id != old.request.request_id
    assert [message.content for message in rig.gateway.calls[1].messages] == [
        "Independent new question"
    ]


async def test_late_old_completion_is_archived_without_memory_history_or_speech(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state", voice=True)
    await rig.say("/memory add Keep this")
    rig.gateway.fail = True
    with pytest.raises(GatewayFailure):
        await rig.say("Old question")
    old = rig.disk.load().requests[0]
    response = rig.gateway.response(old.request).model_copy(
        update={
            "memory_updates": (
                MemoryUpdate(action="forget", text="Keep this"),
                MemoryUpdate(action="remember", text="Old-only fact"),
            )
        }
    )
    await new_conversation(rig)
    rig.gateway.fail = False
    await rig.say("New question")
    before = rig.disk.load()
    sent = tuple(rig.outbox.texts)
    await rig.controller.complete_text(old.request.request_id, response)
    after = rig.disk.load()
    assert after.requests[0].response == response
    assert after.history == before.history
    assert after.memories == before.memories
    assert tuple(rig.outbox.texts) == sent
    assert rig.voice.calls == []
    archived = next(
        item for item in after.replies if item.token == after.requests[0].reply_token
    )
    assert archived.conversation_id == 0
    assert archived.actions == ()
    assert archived.message_id is None
    assert archived.voice_request is None


async def test_archived_cache_recovery_never_delivers_into_new_conversation(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state", voice=True)
    rig.gateway.fail = True
    with pytest.raises(GatewayFailure):
        await rig.say("Old question")
    old = rig.disk.load().requests[0]
    await new_conversation(rig)
    rig.gateway.cached = rig.gateway.response(old.request)
    await rig.say("/recover")
    await rig.say("/recover")
    state = rig.disk.load()
    assert state.requests[0].response == rig.gateway.cached
    assert state.history == ()
    assert rig.gateway.reply not in rig.outbox.texts
    assert rig.gateway.recoveries == [old.request.request_id]
    assert len(rig.gateway.calls) == 1
    assert rig.voice.calls == []


async def test_old_voice_hold_and_buttons_cannot_generate_in_new_chat(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state", voice=True)
    await rig.say("First question")
    old_reply = rig.disk.load().replies[-1]
    rig.voice.fail = True
    with pytest.raises(GatewayFailure):
        await rig.controller.on_callback(rig.callback(old_reply, "voice"))
    old_request = rig.voice.calls[0]
    await new_conversation(rig)
    await rig.controller.on_callback(rig.callback(old_reply, "remember"))
    await rig.controller.on_callback(rig.callback(old_reply, "voice"))
    assert rig.disk.load().memories == ()
    assert len(rig.voice.calls) == 1
    rig.voice.fail = False
    await rig.say("New question")
    fresh = rig.disk.load().replies[-1]
    assert "voice" in fresh.actions
    await rig.controller.on_callback(rig.callback(fresh, "voice"))
    assert len(rig.voice.calls) == 2
    assert rig.voice.calls[0] == old_request
    await rig.say("/recover")
    assert old_request.request_id not in rig.voice.recoveries


async def test_confirmation_survives_restart_and_late_result_after_restart_is_archived(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state"
    rig = Rig(path)
    rig.gateway.fail = True
    with pytest.raises(GatewayFailure):
        await rig.say("Old request")
    old = rig.disk.load().requests[0]
    await rig.say("/new")
    prompt = rig.disk.load().replies[-1]
    resumed = Rig(path)
    resumed.sequence = 100
    await resumed.controller.on_callback(resumed.callback(prompt, "new_confirm"))
    restarted = Rig(path)
    restarted.sequence = 200
    await restarted.controller.on_callback(restarted.callback(prompt, "new_confirm"))
    await restarted.controller.complete_text(
        old.request.request_id, rig.gateway.response(old.request)
    )
    assert restarted.disk.load().conversation_id == 1
    assert restarted.disk.load().history == ()
    assert restarted.gateway.calls == []
    assert restarted.outbox.texts == []
