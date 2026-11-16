from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import anyio
import httpx2
import pytest
from telethon import TelegramClient, events
from telethon.tl.custom.message import Message as SDKMessage
from telethon.tl.types import (
    Document,
    DocumentAttributeAudio,
    KeyboardInlineButton,
    MessageMediaDocument,
    PeerUser,
)

from tanya_bot.telegram_ingress import TelegramIngress
from tanya_bot.telegram_transport import TelethonTransport
from tanya_bot.telegram_voice import TelegramVoiceGateway
from tests.test_telegram_core_support import Rig
from tests.test_telegram_ingress import incoming

pytestmark = pytest.mark.anyio


async def test_new_cancels_active_tts_before_showing_confirmation(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "core", voice=True)
    await rig.say("Synthetic question")
    reply = rig.disk.load().replies[-1]
    rig.voice.block = True
    client = TelegramClient(str(tmp_path / "sdk"), 12345, "offline")
    voice = TelegramVoiceGateway(tmp_path / "audio")
    transport = TelethonTransport(client, 42, voice)
    ingress = TelegramIngress(client, rig.controller, transport, voice, 12345)

    async def speak() -> None:
        scope = anyio.CancelScope()
        ingress.active = scope
        try:
            with scope:
                await rig.controller.on_callback(rig.callback(reply, "voice"))
        finally:
            if ingress.active is scope:
                ingress.active = None

    try:
        with anyio.fail_after(2):
            async with anyio.create_task_group() as group:
                _ = group.start_soon(speak)
                await rig.voice.entered.wait()
                await ingress.on_message(incoming(42, 101, "/new"))
        state = rig.disk.load()
        old = next(item for item in state.replies if item.token == reply.token)
        assert old.voice_cancelled
        assert old.voice_request == rig.voice.calls[0]
        assert old.voice_result is None
        assert state.new_confirmation_token is not None
        assert state.conversation_id == 0
        assert ingress.active is None
        assert rig.outbox.voices == []
        assert len(rig.voice.calls) == 1
    finally:
        await client.disconnect()


async def test_new_cancels_stt_with_durable_hold_and_no_chat_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-offline")
    entered = anyio.Event()
    calls: list[httpx2.Request] = []

    async def wire(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        entered.set()
        await anyio.sleep_forever()
        return httpx2.Response(200, json={"text": "Late transcript"})

    audio = b"synthetic-voice-bytes"
    rig = Rig(tmp_path / "core")
    client = TelegramClient(str(tmp_path / "sdk"), 12345, "offline")
    voice = TelegramVoiceGateway(tmp_path / "audio", httpx2.MockTransport(wire))
    transport = TelethonTransport(client, 42, voice)
    ingress = TelegramIngress(client, rig.controller, transport, voice, 12345)

    async def download(_message: SDKMessage, _file: type[bytes]) -> bytes:
        return audio

    monkeypatch.setattr(client, "download_media", download)
    document = Document(
        1,
        0,
        b"offline",
        None,
        "audio/ogg",
        len(audio),
        1,
        [DocumentAttributeAudio(2, voice=True)],
    )
    message = SDKMessage(
        100,
        PeerUser(42),
        from_id=PeerUser(42),
        media=MessageMediaDocument(document=document, voice=True),
    )
    event = events.NewMessage.Event(message)
    try:
        with anyio.fail_after(3):
            async with anyio.create_task_group() as group:
                _ = group.start_soon(ingress.on_message, event)
                await entered.wait()
                await ingress.on_message(incoming(42, 101, "/new"))
        request_id = uuid5(NAMESPACE_URL, "tolan:12345:voice:42:100")
        assert await voice.recover_transcript(request_id, audio) is None
        assert len(calls) == 1
        assert rig.gateway.calls == []
        assert rig.disk.load().new_confirmation_token is not None
        assert ingress.active is None
    finally:
        await client.disconnect()


async def test_new_control_barrier_blocks_generation_while_saving_tombstones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path / "core")
    client = TelegramClient(str(tmp_path / "sdk"), 12345, "offline")
    voice = TelegramVoiceGateway(tmp_path / "audio")
    transport = TelethonTransport(client, 42, voice)
    ingress = TelegramIngress(client, rig.controller, transport, voice, 12345)
    entered = anyio.Event()
    release = anyio.Event()
    original = rig.controller.cancel_pending
    notices: list[str] = []

    async def suspend_cancel() -> None:
        entered.set()
        await release.wait()
        await original()

    async def send(
        entity: int,
        message: str,
        **_options: bool | list[list[KeyboardInlineButton]] | None,
    ) -> SDKMessage:
        assert entity == 42
        notices.append(message)
        return SDKMessage(77, PeerUser(42))

    monkeypatch.setattr(rig.controller, "cancel_pending", suspend_cancel)
    monkeypatch.setattr(client, "send_message", send)
    try:
        with anyio.fail_after(3):
            async with anyio.create_task_group() as group:
                _ = group.start_soon(ingress.on_message, incoming(42, 100, "/new"))
                await entered.wait()
                await ingress.on_message(
                    incoming(42, 101, "Do not generate during handoff")
                )
                assert rig.gateway.calls == []
                assert ingress.active is not None
                release.set()
        assert len(notices) == 1
        assert rig.disk.load().new_confirmation_token is not None
        assert rig.gateway.calls == []
    finally:
        await client.disconnect()
