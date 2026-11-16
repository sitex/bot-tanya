from pathlib import Path

import httpx2
import pytest

from tanya_bot.memory_runtime import MemoryCoordinator
from tanya_bot.telegram_gateway import LocalCompanionGateway
from tests.conftest import Harness
from tests.test_telegram_core_support import Rig
from tests.test_turn_api import provider_answer


@pytest.mark.anyio
async def test_enhanced_gateway_recovery_after_failed_local_commit_is_cache_only(
    harness: Harness,
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "bot", voice=True)
    rig.controller.memory = MemoryCoordinator(tmp_path / "vectors")
    rig.controller.gateway = LocalCompanionGateway(
        harness.directory,
        httpx2.ASGITransport(app=harness.app),
        core=rig.disk,
    )
    harness.wire.outcomes.append(provider_answer())
    rig.disk.fail_completion = True
    with pytest.raises(OSError, match="Synthetic disk failure"):
        await rig.say("A v2 request")
    pending = rig.disk.load().requests[0]
    assert pending.response is None
    assert rig.outbox.texts == []

    def forbidden(_request: httpx2.Request) -> httpx2.Response:
        pytest.fail("Recovery issued an HTTP request")

    resumed = Rig(tmp_path / "bot", voice=True)
    resumed.sequence = 10
    resumed.controller.gateway = LocalCompanionGateway(
        harness.directory,
        httpx2.MockTransport(forbidden),
        core=resumed.disk,
    )
    await resumed.say("/recover")
    await resumed.say("/recover")
    state = resumed.disk.load()
    assert state.context.summary.text == "New summary"
    assert state.replies[0].delivery is not None
    assert resumed.outbox.texts.count("Current reply") == 1
    assert len(harness.wire.calls) == 1
    assert resumed.voice.calls == []
