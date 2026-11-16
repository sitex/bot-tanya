import json
from pathlib import Path
from uuid import uuid4

import httpx2
import pytest
from telethon import TelegramClient
from telethon.tl.custom.message import Message as SDKMessage
from telethon.tl.types import KeyboardInlineButton, PeerUser

from tanya_bot.errors import ServiceError
from tanya_bot.receipts import exclusive_store, mark_sent, save_success
from tanya_bot.schemas import ChatRequest, ChatResponse, Message
from tanya_bot.telegram_config import identity
from tanya_bot.telegram_gateway import LocalCompanionGateway
from tanya_bot.telegram_models import Button, VoiceResult
from tanya_bot.telegram_store import CoreError
from tanya_bot.telegram_transport import TelethonTransport


class NoArtifacts:
    def resolve(self, result: VoiceResult) -> Path:
        raise AssertionError(result)


@pytest.mark.anyio
async def test_gateway_sends_exact_request_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TOLAN_CLIENT_TOKEN", "offline-client")
    request = ChatRequest(
        request_id=uuid4(), messages=(Message(role="user", content="Hello"),)
    )
    answer = ChatResponse(
        request_id=request.request_id,
        reply="Hi",
        model="offline",
        input_tokens=3,
        output_tokens=1,
    )
    calls: list[httpx2.Request] = []

    def wire(incoming: httpx2.Request) -> httpx2.Response:
        calls.append(incoming)
        assert incoming.headers["authorization"] == "Bearer offline-client"
        assert ChatRequest.model_validate_json(incoming.content) == request
        return httpx2.Response(200, content=answer.model_dump_json())

    gateway = LocalCompanionGateway(tmp_path / "receipts", httpx2.MockTransport(wire))
    assert await gateway.chat(request) == answer
    assert len(calls) == 1
    assert await gateway.recover(request) is None
    assert len(calls) == 1


@pytest.mark.anyio
async def test_recovery_of_held_and_completed_receipts_is_local(tmp_path: Path) -> None:
    request = ChatRequest(
        request_id=uuid4(), messages=(Message(role="user", content="Hello"),)
    )
    answer = ChatResponse(
        request_id=request.request_id,
        reply="Saved",
        model="offline",
        input_tokens=3,
        output_tokens=1,
    )

    def forbidden(_incoming: httpx2.Request) -> httpx2.Response:
        pytest.fail("Recovery must never access HTTP")

    directory = tmp_path / "receipts"
    gateway = LocalCompanionGateway(directory, httpx2.MockTransport(forbidden))
    with exclusive_store(directory) as connection:
        mark_sent(connection, request)
    assert await gateway.recover(request) is None
    with exclusive_store(directory) as connection:
        save_success(connection, answer)
    assert await gateway.recover(request) == answer
    changed = request.model_copy(update={"language": "en"})
    with pytest.raises(ServiceError, match="request_conflict"):
        _ = await gateway.recover(changed)


@pytest.mark.anyio
async def test_transport_owner_gate_precedes_sdk_and_preserves_plain_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = TelegramClient(str(tmp_path / "test-sdk"), 12345, "offline")
    transport = TelethonTransport(client, 42, NoArtifacts())
    texts: list[str] = []

    async def send(
        entity: int,
        message: str,
        *,
        buttons: list[list[KeyboardInlineButton]] | None,
        parse_mode: None,
        link_preview: bool,
    ) -> SDKMessage:
        assert entity == 42
        assert parse_mode is None
        assert not link_preview
        assert buttons is not None
        assert len(buttons) == 1
        texts.append(message)
        return SDKMessage(77, peer_id=PeerUser(42))

    monkeypatch.setattr(client, "send_message", send)
    with pytest.raises(CoreError, match="telegram_owner_required"):
        _ = await transport.send_text(43, "private", ())
    assert texts == []
    assert (
        await transport.send_text(
            42, "**plain** <text>", ((Button(text="Save", data="opaque"),),)
        )
        == 77
    )
    assert texts == ["**plain** <text>"]
    await client.disconnect()


def test_identity_requires_private_completed_receipt(tmp_path: Path) -> None:
    path = tmp_path / "creation.json"
    with pytest.raises(CoreError, match="bot_creation_required"):
        _ = identity(tmp_path)
    _ = path.write_text(
        json.dumps(
            {
                "stage": "created",
                "owner_id": 42,
                "bot_id": 12345,
                "username": "test_bot",
            }
        )
    )
    path.chmod(0o600)
    assert identity(tmp_path).owner_id == 42
    path.chmod(0o644)
    with pytest.raises(ServiceError, match="storage_unavailable"):
        _ = identity(tmp_path)
