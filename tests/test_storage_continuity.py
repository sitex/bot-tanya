import sqlite3
from contextlib import closing
from functools import partial
from pathlib import Path
from typing import Literal, assert_never

import httpx2
import pytest

from tanya_bot import turn_receipts
from tanya_bot.app import create_app
from tanya_bot.errors import ServiceError
from tanya_bot.receipts import exclusive_store, mark_sent
from tanya_bot.schemas import ChatRequest, ErrorResponse
from tanya_bot.telegram_models import TelegramState
from tanya_bot.telegram_store import CoreError, TelegramStore
from tests.conftest import Harness
from tests.test_telegram_core_support import Rig
from tests.test_turn_api import envelope, provider_answer


@pytest.mark.anyio
async def test_missing_reserved_telegram_state_blocks_replay_before_gateway(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "core")
    event = rig.event("Reserved question")
    assert await rig.controller.begin_text(event) is not None
    (rig.disk.disk.directory / "state.json").unlink()
    resumed = Rig(rig.disk.disk.directory)

    with pytest.raises(CoreError, match="state_missing"):
        await resumed.controller.on_text(event)

    assert resumed.gateway.calls == []
    assert resumed.outbox.texts == []


@pytest.mark.parametrize("operation", ["load", "save"])
def test_missing_telegram_authority_cannot_be_recreated_directly(
    tmp_path: Path, operation: Literal["load", "save"]
) -> None:
    store = TelegramStore(tmp_path / "core", 42)
    with store.exclusive():
        saved = store.load()
        store.save(saved)
    (store.directory / "state.json").unlink()

    match operation:
        case "load":
            action = TelegramStore(store.directory, 42).load
        case "save":
            action = partial(store.save, saved)
        case _:
            assert_never(operation)
    with pytest.raises(CoreError, match="state_missing"):
        _ = action()
    assert not (store.directory / "state.json").exists()


@pytest.mark.anyio
@pytest.mark.parametrize("remnant", ["state.lock", ".state-interrupted"])
async def test_partial_telegram_initialization_blocks_generation(
    tmp_path: Path, remnant: str
) -> None:
    directory = tmp_path / "core"
    directory.mkdir(mode=0o700)
    path = directory / remnant
    _ = path.write_bytes(b"")
    path.chmod(0o600)
    rig = Rig(directory)

    with pytest.raises(CoreError, match="state_missing"):
        await rig.say("Question")

    assert rig.gateway.calls == []


@pytest.mark.parametrize("precreated", [False, True])
def test_fresh_telegram_initialization_is_durable_before_first_save(
    tmp_path: Path, *, precreated: bool
) -> None:
    directory = tmp_path / "core"
    if precreated:
        directory.mkdir(mode=0o700)
    with TelegramStore(directory, 42).exclusive():
        pass

    assert TelegramState.model_validate_json(
        (directory / "state.json").read_bytes()
    ) == TelegramState(owner_id=42)


@pytest.mark.parametrize("locked", [False, True])
def test_legacy_telegram_adoption_preserves_authority_bytes(
    tmp_path: Path, *, locked: bool
) -> None:
    directory = tmp_path / "core"
    directory.mkdir(mode=0o700)
    legacy = b'{"version":1,"owner_id":42,"settings":{"name":"Legacy"}}'
    state = directory / "state.json"
    _ = state.write_bytes(legacy)
    state.chmod(0o600)
    if locked:
        lock = directory / "state.lock"
        _ = lock.write_bytes(b"")
        lock.chmod(0o600)

    with TelegramStore(directory, 42).exclusive():
        restored = TelegramStore(directory, 42).load()

    assert restored.settings.name == "Legacy"
    assert restored.settings.language == "ru"
    assert state.read_bytes() == legacy


@pytest.mark.anyio
@pytest.mark.parametrize(
    "scenario",
    [("v1", "database"), ("v2", "database"), ("v1", "table"), ("v2", "table")],
)
async def test_missing_receipt_authority_blocks_replayed_hold_before_provider(
    harness: Harness,
    chat_request: ChatRequest,
    scenario: tuple[Literal["v1", "v2"], Literal["database", "table"]],
) -> None:
    version, damage = scenario
    turn = envelope(chat_request)
    directory = harness.directory / "turns-v2" if version == "v2" else harness.directory
    with exclusive_store(directory) as connection:
        match version:
            case "v1":
                mark_sent(connection, chat_request)
            case "v2":
                turn_receipts.reserve(connection, turn)
            case _:
                assert_never(version)
    database = directory / "receipts.sqlite3"
    match damage:
        case "database":
            database.unlink()
        case "table":
            with closing(sqlite3.connect(database)) as connection:
                _ = connection.execute("DROP TABLE receipts")
                connection.commit()
        case _:
            assert_never(damage)
    damaged = database.read_bytes() if database.exists() else None
    restarted = Harness(
        harness.directory,
        harness.wire,
        create_app(harness.directory, httpx2.MockTransport(harness.wire.handle)),
    )

    async with restarted.client() as client:
        response = await client.post(
            "/v2/telegram/chat" if version == "v2" else "/v1/chat",
            content=turn.model_dump_json()
            if version == "v2"
            else chat_request.model_dump_json(),
        )

    assert response.status_code == 503
    assert (
        ErrorResponse.model_validate_json(response.content).detail.code
        == "storage_unavailable"
    )
    assert harness.wire.calls == []
    assert (database.read_bytes() if database.exists() else None) == damaged


@pytest.mark.parametrize(
    "remnant", ["request.lock", "receipts.sqlite3", "unknown", "turns-v2"]
)
def test_partial_receipt_initialization_is_rejected(
    tmp_path: Path, remnant: str
) -> None:
    directory = tmp_path / "api"
    directory.mkdir(mode=0o700)
    path = directory / remnant
    _ = path.write_bytes(b"")
    path.chmod(0o600)

    with (
        pytest.raises(ServiceError, match="storage_unavailable"),
        exclusive_store(directory),
    ):
        pytest.fail("Partial receipt authority was accepted")


@pytest.mark.parametrize("locked", [False, True])
def test_legacy_receipt_adoption_preserves_rows_and_database_bytes(
    tmp_path: Path, chat_request: ChatRequest, *, locked: bool
) -> None:
    directory = tmp_path / "api"
    with exclusive_store(directory) as connection:
        mark_sent(connection, chat_request)
    if not locked:
        (directory / "request.lock").unlink()
    database = directory / "receipts.sqlite3"
    before = database.read_bytes()

    with exclusive_store(directory) as connection:
        rows = connection.execute(
            "SELECT request_id, response FROM receipts"
        ).fetchall()

    assert rows == [(str(chat_request.request_id), None)]
    assert database.read_bytes() == before


@pytest.mark.anyio
async def test_first_v1_request_after_v2_initializes_its_independent_ledger(
    harness: Harness, chat_request: ChatRequest
) -> None:
    harness.directory.mkdir(mode=0o700)
    harness.wire.outcomes.append(provider_answer())
    async with harness.client() as client:
        v2 = await client.post(
            "/v2/telegram/chat", content=envelope(chat_request).model_dump_json()
        )
        v1 = await client.post("/v1/chat", content=chat_request.model_dump_json())

    assert v2.status_code == v1.status_code == 200
    assert len(harness.wire.calls) == 2
