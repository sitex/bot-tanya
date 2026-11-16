from datetime import UTC, date, datetime, tzinfo
from pathlib import Path

import anyio
import pytest
from tanya_memory import MemoryOperationError

from tanya_bot import runtime_assembly
from tanya_bot.memory_runtime import MemoryCoordinator
from tanya_bot.telegram_models import TelegramState
from tests.test_telegram_core_support import Rig

pytestmark = pytest.mark.anyio


async def test_assembly_preserves_profile_and_enables_v2_and_local_gmail_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TANYA_MEMORY_MODEL_DIR", raising=False)
    monkeypatch.setenv("TANYA_GMAIL_CONFIG", str(tmp_path / "missing.json"))
    monkeypatch.setenv("TANYA_GMAIL_TOKEN_DIR", str(tmp_path / "tokens"))
    rig = Rig(tmp_path / "state")
    await rig.say("/settings name Existing")
    await rig.say("/settings language ru")
    await rig.say("/settings voice cedar")
    before = rig.disk.load().settings
    await runtime_assembly.assemble(rig.controller, tmp_path)
    await rig.say("/gmail status")
    assert rig.outbox.texts[-1].startswith("Gmail не настроен.")
    await rig.say("A new chat")
    state = rig.disk.load()
    assert state.settings == before
    assert state.requests[-1].context is not None
    assert len(rig.gateway.calls) == 1
    assert not (tmp_path / "tokens").exists()


class Clock:
    def __init__(self) -> None:
        self.index: int = 0

    def now(self, zone: tzinfo) -> datetime:
        assert zone is UTC
        return datetime(2026, 11, 16 + (self.index // 3), tzinfo=UTC)


async def test_startup_hourly_retry_and_once_per_utc_day(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = Rig(tmp_path / "state")
    memory = MemoryCoordinator(tmp_path / "vectors")
    rig.controller.memory = memory
    clock = Clock()
    marks: list[date | None] = []
    calls: list[int] = []
    original = memory.maintenance

    async def maintain(state: TelegramState) -> tuple[()]:
        calls.append(clock.index)
        if clock.index == 0:
            raise MemoryOperationError(code="synthetic_cache_failure")
        assert await original(state) == ()
        return ()

    with anyio.CancelScope() as scope:

        async def pause(seconds: float) -> None:
            assert seconds == 3600
            marks.append(rig.disk.load().context.maintenance_day)
            clock.index += 1
            if clock.index == 4:
                scope.cancel()
            await anyio.lowlevel.checkpoint()

        monkeypatch.setattr(runtime_assembly, "datetime", clock)
        monkeypatch.setattr(anyio, "sleep", pause)
        monkeypatch.setattr(memory, "maintenance", maintain)
        await runtime_assembly.daily_maintenance(rig.controller)
    assert marks == [None, date(2026, 11, 16), date(2026, 11, 16), date(2026, 11, 17)]
    assert calls == [0, 1, 3]
