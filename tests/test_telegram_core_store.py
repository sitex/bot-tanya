import stat
from pathlib import Path

import pytest

from tanya_bot.telegram_models import Entry, TelegramState
from tanya_bot.telegram_store import CoreError, TelegramStore
from tests.test_telegram_core_support import Rig

pytestmark = pytest.mark.anyio


async def test_corrupt_state_fails_closed_before_any_gateway_or_delivery(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state")
    await rig.say("/help")
    path = rig.disk.disk.directory / "state.json"
    _ = path.write_bytes(b'{"owner_id":42,"requests":[')
    with pytest.raises(CoreError, match="state_corrupt"):
        await rig.say("Question")
    assert rig.gateway.calls == []
    assert len(rig.outbox.texts) == 1


async def test_unsafe_state_owner_and_process_lock_fail_closed(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "state")
    await rig.say("/help")
    directory = rig.disk.disk.directory
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE((directory / "state.json").stat().st_mode) == 0o600
    foreign = TelegramStore(directory, 7)
    with pytest.raises(CoreError, match="state_owner"):
        _ = foreign.load()
    with (
        rig.disk.exclusive(),
        pytest.raises(CoreError, match="store_busy"),
        TelegramStore(directory, 42).exclusive(),
    ):
        pytest.fail("Concurrent transaction acquired the store")
    (directory / "state.json").chmod(0o644)
    with pytest.raises(CoreError, match="unsafe_file"):
        await rig.say("Question")
    assert rig.gateway.calls == []


async def test_state_symlink_never_reads_or_changes_target(tmp_path: Path) -> None:
    directory = tmp_path / "state"
    directory.mkdir(mode=0o700)
    target = tmp_path / "target"
    _ = target.write_bytes(b"PRIVATE SYNTHETIC TARGET")
    (directory / "state.json").symlink_to(target)
    rig = Rig(directory)
    with pytest.raises(OSError, match="Too many levels of symbolic links"):
        await rig.say("Question")
    assert target.read_bytes() == b"PRIVATE SYNTHETIC TARGET"
    assert rig.gateway.calls == []


async def test_failed_reservation_prevents_paid_call(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "state")
    rig.disk.fail_all = True
    with pytest.raises(OSError, match="Synthetic disk failure"):
        await rig.say("Question")
    assert rig.gateway.calls == []
    assert rig.outbox.texts == []


async def test_semantically_corrupt_history_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "state"
    directory.mkdir(mode=0o700)
    state = TelegramState(owner_id=42).model_copy(
        update={"history": (Entry(text="Bad"),)}
    )
    path = directory / "state.json"
    _ = path.write_text(state.model_dump_json())
    path.chmod(0o600)
    with pytest.raises(CoreError, match="state_corrupt"):
        _ = TelegramStore(directory, 42).load()
