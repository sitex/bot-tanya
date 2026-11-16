from pathlib import Path

import anyio
import pytest
from telethon import TelegramClient, events
from telethon.tl.custom.message import Message as SDKMessage
from telethon.tl.types import KeyboardInlineButton, PeerUser

from tanya_bot.schemas import ChatRequest, ChatResponse
from tanya_bot.telegram_controller import TelegramController
from tanya_bot.telegram_ingress import TelegramIngress
from tanya_bot.telegram_models import EVENT_CAPACITY, StoredReply
from tanya_bot.telegram_store import TelegramStore
from tanya_bot.telegram_transport import TelethonTransport
from tanya_bot.telegram_voice import TelegramVoiceGateway


class BlockingGateway:
    def __init__(self) -> None:
        self.started: anyio.Event = anyio.Event()
        self.calls: int = 0

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        self.started.set()
        await anyio.sleep_forever()
        return ChatResponse(
            request_id=request.request_id,
            reply="late reply",
            model="offline",
            input_tokens=1,
            output_tokens=1,
        )

    async def recover(self, request: ChatRequest) -> ChatResponse | None:
        assert request.request_id is not None
        return None


def incoming(sender: int, number: int, text: str) -> events.NewMessage.Event:
    message = SDKMessage(number, PeerUser(42), from_id=PeerUser(sender), message=text)
    return events.NewMessage.Event(message)


@pytest.mark.anyio
async def test_foreign_sdk_event_never_opens_private_core(tmp_path: Path) -> None:
    client = TelegramClient(str(tmp_path / "sdk"), 12345, "offline")
    voice = TelegramVoiceGateway(tmp_path / "audio")
    store = TelegramStore(tmp_path / "core", 42)
    transport = TelethonTransport(client, 42, voice)
    gateway = BlockingGateway()
    controller = TelegramController(42, store, transport, gateway, voice)
    ingress = TelegramIngress(client, controller, transport, voice, 12345)
    try:
        with anyio.fail_after(1):
            await ingress.on_message(incoming(7, 100, "private input"))
        assert gateway.calls == 0
        assert not store.directory.exists()
    finally:
        await client.disconnect()


@pytest.mark.anyio
async def test_cancel_sdk_message_cancels_work_and_saves_tombstone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = TelegramClient(str(tmp_path / "sdk"), 12345, "offline")
    voice = TelegramVoiceGateway(tmp_path / "audio")
    store = TelegramStore(tmp_path / "core", 42)
    transport = TelethonTransport(client, 42, voice)
    gateway = BlockingGateway()
    controller = TelegramController(42, store, transport, gateway, voice)
    ingress = TelegramIngress(client, controller, transport, voice, 12345)
    notices: list[str] = []

    async def send(
        entity: int,
        message: str,
        **_options: bool | list[list[KeyboardInlineButton]] | None,
    ) -> SDKMessage:
        assert entity == 42
        notices.append(message)
        return SDKMessage(77, PeerUser(42))

    monkeypatch.setattr(client, "send_message", send)
    try:
        with anyio.fail_after(2):
            async with anyio.create_task_group() as group:
                _ = group.start_soon(ingress.on_message, incoming(42, 100, "hello"))
                await gateway.started.wait()
                await ingress.on_message(incoming(42, 101, "/cancel"))
        with store.exclusive():
            state = store.load()
        assert len(state.requests) == 1
        assert state.requests[0].cancelled
        assert state.requests[0].response is None
        assert ingress.active is None
        assert gateway.calls == 1
        assert "late reply" not in notices
    finally:
        await client.disconnect()


@pytest.mark.anyio
@pytest.mark.parametrize("text", ["Question", "/memory add Rejected fact"])
async def test_capacity_sdk_notice_does_not_offer_unsent_request_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    client = TelegramClient(str(tmp_path / "sdk"), 12345, "offline")
    voice = TelegramVoiceGateway(tmp_path / "audio")
    store = TelegramStore(tmp_path / "core", 42)
    transport = TelethonTransport(client, 42, voice)
    gateway = BlockingGateway()
    controller = TelegramController(42, store, transport, gateway, voice)
    ingress = TelegramIngress(client, controller, transport, voice, 12345)
    notices: list[str] = []

    async def send(
        entity: int,
        message: str,
        **_options: bool | list[list[KeyboardInlineButton]] | None,
    ) -> SDKMessage:
        assert entity == 42
        notices.append(message)
        return SDKMessage(77, PeerUser(42))

    monkeypatch.setattr(client, "send_message", send)
    with store.exclusive():
        state = store.load()
        store.save(
            state.model_copy(
                update={
                    "replies": tuple(
                        StoredReply(text="Saved") for _ in range(EVENT_CAPACITY)
                    )
                }
            )
        )
    try:
        with anyio.fail_after(2):
            await ingress.on_message(incoming(42, 100, text))
        assert gateway.calls == 0
        with store.exclusive():
            saved = store.load()
        assert saved.requests == ()
        assert saved.memories == ()
        assert len(saved.replies) == EVENT_CAPACITY
        assert len(notices) == 1
        assert "/recover" not in notices[0]
        assert ingress.active is None
    finally:
        await client.disconnect()
