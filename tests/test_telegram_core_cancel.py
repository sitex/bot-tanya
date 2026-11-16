from pathlib import Path

import anyio
import pytest

from tanya_bot.schemas import MemoryUpdate
from tanya_bot.telegram_models import VoiceResult
from tests.test_telegram_core_support import Rig

pytestmark = pytest.mark.anyio


async def test_post_gateway_checkpoint_honors_already_cancelled_scope(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state")
    with anyio.CancelScope() as scope:
        rig.gateway.cancel_on_return = scope
        await rig.say("Question")
    assert scope.cancelled_caught
    assert rig.disk.load().requests[0].response is None
    assert rig.outbox.texts == []


async def test_cancel_voice_keeps_uuid_and_requires_explicit_cached_recovery(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state", voice=True)
    await rig.say("Question")
    event = rig.callback(rig.disk.load().replies[-1], "voice")
    rig.voice.block = True
    async with anyio.create_task_group() as group:
        _ = group.start_soon(rig.controller.on_callback, event)
        await rig.voice.entered.wait()
        group.cancel_scope.cancel()
    await rig.controller.cancel_pending()
    reply = rig.disk.load().replies[-1]
    assert reply.voice_cancelled
    assert reply.voice_request is not None
    assert rig.outbox.voices == []
    await rig.controller.on_callback(event)
    assert len(rig.voice.calls) == 1
    rig.voice.cached = VoiceResult(
        request_id=reply.voice_request.request_id, artifact_id="cached-after-cancel"
    )
    await rig.say("/recover")
    assert rig.outbox.voices == [rig.voice.cached]
    assert len(rig.voice.calls) == 1


async def test_forget_cannot_remove_fact_not_supplied_in_same_request(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state")
    await rig.say("/memory add Supplied fact")
    rig.sequence += 1
    request = await rig.controller.begin_text(rig.event("Correction"))
    assert request is not None
    await rig.say("/memory add Later fact")
    rig.gateway.updates = (
        MemoryUpdate(action="forget", text="Supplied fact"),
        MemoryUpdate(action="forget", text="Later fact"),
    )
    await rig.controller.complete_text(
        request.request_id, rig.gateway.response(request)
    )
    assert [entry.text for entry in rig.disk.load().memories] == ["Later fact"]
