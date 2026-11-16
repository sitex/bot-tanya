from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING

import anyio

from tanya_bot.schemas import ChatRequest, ChatResponse, MemoryUpdate
from tanya_bot.telegram_controller import TelegramController
from tanya_bot.telegram_models import (
    GatewayFailure,
    IncomingCallback,
    IncomingText,
    Keyboard,
    StoredReply,
    TelegramState,
    VoiceRequest,
    VoiceResult,
)
from tanya_bot.telegram_store import TelegramStore

if TYPE_CHECKING:
    from uuid import UUID


class FaultDisk:
    def __init__(self, path: Path) -> None:
        self.disk: TelegramStore = TelegramStore(path, 42)
        self.fail_completion: bool = False
        self.fail_all: bool = False

    def exclusive(self) -> AbstractContextManager[None]:
        return self.disk.exclusive()

    def load(self) -> TelegramState:
        return self.disk.load()

    def save(self, state: TelegramState) -> None:
        if self.fail_all or (
            self.fail_completion
            and any(item.response is not None for item in state.requests)
        ):
            message = "Synthetic disk failure"
            raise OSError(message)
        self.disk.save(state)


class Outbox:
    def __init__(self, disk: FaultDisk) -> None:
        self.disk: FaultDisk = disk
        self.texts: list[str] = []
        self.keyboards: dict[int, Keyboard] = {}
        self.acks: list[str] = []
        self.voices: list[VoiceResult] = []
        self.fail_send: bool = False

    async def send_text(self, chat_id: int, text: str, keyboard: Keyboard) -> int:
        assert chat_id == 42
        assert keyboard == ()
        assert any(reply.delivery_attempted for reply in self.disk.load().replies)
        if self.fail_send:
            message = "Synthetic Telegram timeout"
            raise TimeoutError(message)
        self.texts.append(text)
        return len(self.texts)

    async def edit_keyboard(
        self, chat_id: int, message_id: int, keyboard: Keyboard
    ) -> None:
        assert chat_id == 42
        assert any(reply.message_id == message_id for reply in self.disk.load().replies)
        assert all(
            len(button.data.encode()) <= 64 for row in keyboard for button in row
        )
        self.keyboards[message_id] = keyboard

    async def ack(self, callback_id: str, text: str) -> None:
        assert callback_id
        self.acks.append(text)

    async def send_voice(self, chat_id: int, result: VoiceResult) -> int:
        assert chat_id == 42
        assert any(
            reply.voice_result == result and reply.voice_delivery_attempted
            for reply in self.disk.load().replies
        )
        self.voices.append(result)
        return 100


class Gateway:
    def __init__(self, disk: FaultDisk) -> None:
        self.disk: FaultDisk = disk
        self.calls: list[ChatRequest] = []
        self.recoveries: list[UUID] = []
        self.reply: str = "Synthetic companion reply"
        self.updates: tuple[MemoryUpdate, ...] = ()
        self.fail: bool = False
        self.block: bool = False
        self.entered: anyio.Event = anyio.Event()
        self.cached: ChatResponse | None = None
        self.cancel_on_return: anyio.CancelScope | None = None

    def response(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(
            request_id=request.request_id,
            reply=self.reply,
            model="fake",
            input_tokens=10,
            output_tokens=5,
            memory_updates=self.updates,
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        assert any(
            item.request == request and item.response is None
            for item in self.disk.load().requests
        )
        self.calls.append(request)
        self.entered.set()
        if self.block:
            await anyio.sleep_forever()
        if self.fail:
            raise GatewayFailure(code="synthetic_unknown")
        if self.cancel_on_return is not None:
            self.cancel_on_return.cancel()
        return self.response(request)

    async def recover(self, request: ChatRequest) -> ChatResponse | None:
        self.recoveries.append(request.request_id)
        return self.cached


class Voice:
    def __init__(self, disk: FaultDisk) -> None:
        self.disk: FaultDisk = disk
        self.calls: list[VoiceRequest] = []
        self.recoveries: list[UUID] = []
        self.fail: bool = False
        self.cached: VoiceResult | None = None
        self.block: bool = False
        self.entered: anyio.Event = anyio.Event()

    async def generate(self, request: VoiceRequest) -> VoiceResult:
        assert any(
            reply.voice_request == request and reply.voice_result is None
            for reply in self.disk.load().replies
        )
        self.calls.append(request)
        self.entered.set()
        if self.block:
            await anyio.sleep_forever()
        if self.fail:
            raise GatewayFailure(code="synthetic_voice_unknown")
        return VoiceResult(
            request_id=request.request_id, artifact_id="synthetic-artifact"
        )

    async def recover(self, request: VoiceRequest) -> VoiceResult | None:
        self.recoveries.append(request.request_id)
        return self.cached


class Rig:
    def __init__(self, path: Path, *, voice: bool = False) -> None:
        self.disk: FaultDisk = FaultDisk(path)
        self.outbox: Outbox = Outbox(self.disk)
        self.gateway: Gateway = Gateway(self.disk)
        self.voice: Voice = Voice(self.disk)
        self.controller: TelegramController = TelegramController(
            42,
            self.disk,
            self.outbox,
            self.gateway,
            self.voice if voice else None,
        )
        self.sequence: int = 0

    async def say(self, text: str) -> None:
        self.sequence += 1
        await self.controller.on_text(self.event(text))

    def event(self, text: str) -> IncomingText:
        return IncomingText(
            update_id=str(self.sequence),
            sender_id=42,
            chat_id=42,
            private=True,
            text=text,
        )

    def callback(self, reply: StoredReply, action: str) -> IncomingCallback:
        self.sequence += 1
        assert reply.message_id is not None
        return IncomingCallback(
            update_id=str(self.sequence),
            sender_id=42,
            chat_id=42,
            private=True,
            callback_id=f"cb{self.sequence}",
            message_id=reply.message_id,
            data=f"{reply.token}:{action}",
        )
