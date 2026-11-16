from datetime import date
from pathlib import Path
from typing import override

import anyio
import pytest
from tanya_memory import RelatedPair, TurnContext

from tanya_bot.memory_runtime import MemoryCoordinator
from tanya_bot.schemas import ChatRequest
from tanya_bot.telegram_models import TelegramState
from tests.test_telegram_core_support import Rig

pytestmark = pytest.mark.anyio


class PausedMemory(MemoryCoordinator):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.prepared: anyio.Event = anyio.Event()
        self.release: anyio.Event = anyio.Event()
        self.maintaining: anyio.Event = anyio.Event()
        self.finish: anyio.Event = anyio.Event()

    @override
    async def prepare(
        self,
        state: TelegramState,
        request: ChatRequest,
    ) -> tuple[ChatRequest, TurnContext]:
        result = await super().prepare(state, request)
        self.prepared.set()
        await self.release.wait()
        return result

    @override
    async def maintenance(self, state: TelegramState) -> tuple[RelatedPair, ...]:
        pairs = await super().maintenance(state)
        self.maintaining.set()
        await self.finish.wait()
        return pairs


async def test_daily_metadata_does_not_drop_prepared_chat(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "state")
    memory = PausedMemory(tmp_path / "vectors")
    rig.controller.memory = memory
    memory.finish.set()
    with anyio.fail_after(2):
        async with anyio.create_task_group() as tasks:
            _ = tasks.start_soon(rig.say, "Question during maintenance")
            await memory.prepared.wait()
            await rig.controller.maintain_memory(date(2026, 11, 16))
            memory.release.set()
    assert len(rig.gateway.calls) == 1
    assert rig.disk.load().context.maintenance_day == date(2026, 11, 16)


@pytest.mark.parametrize("change", ["source", "new"])
async def test_source_or_conversation_change_discards_prepared_chat(
    tmp_path: Path,
    change: str,
) -> None:
    rig = Rig(tmp_path / "state")
    memory = PausedMemory(tmp_path / "vectors")
    rig.controller.memory = memory
    with anyio.fail_after(2):
        async with anyio.create_task_group() as tasks:
            _ = tasks.start_soon(rig.say, "Question during edit")
            await memory.prepared.wait()
            if change == "source":
                await rig.say("/memory add New manual fact")
            else:
                await rig.say("/new")
                menu = rig.disk.load().replies[-1]
                await rig.controller.on_callback(rig.callback(menu, "new_confirm"))
            memory.release.set()
    assert rig.gateway.calls == []
    assert rig.disk.load().requests == ()


async def test_daily_maintenance_edit_race_retries_without_marking_stale_day(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state")
    memory = PausedMemory(tmp_path / "vectors")
    rig.controller.memory = memory
    day = date(2026, 11, 16)
    with anyio.fail_after(2):
        async with anyio.create_task_group() as tasks:
            _ = tasks.start_soon(rig.controller.maintain_memory, day)
            await memory.maintaining.wait()
            await rig.say("/memory add A new fact")
            memory.finish.set()
    assert rig.disk.load().context.maintenance_day is None
    await rig.controller.maintain_memory(day)
    assert rig.disk.load().context.maintenance_day == day
    before = (tmp_path / "state/state.json").read_bytes()
    resumed = Rig(tmp_path / "state")
    resumed.controller.memory = MemoryCoordinator(tmp_path / "vectors")
    await resumed.controller.maintain_memory(day)
    assert (tmp_path / "state/state.json").read_bytes() == before


async def test_corrupt_disposable_cache_does_not_reset_sources(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "state")
    await rig.say("/memory add Current source")
    directory = tmp_path / "vectors"
    directory.mkdir(mode=0o700)
    cache = directory / "vectors.sqlite3"
    _ = cache.write_bytes(b"corrupt sqlite")
    cache.chmod(0o600)
    rig.controller.memory = MemoryCoordinator(directory)
    await rig.say("Current source")
    assert rig.gateway.calls[-1].memories == ("Current source",)
    assert cache.read_bytes() == b"corrupt sqlite"
