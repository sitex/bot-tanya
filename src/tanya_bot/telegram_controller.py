"""Owner-only durable Telegram controller independent of the Telethon SDK."""

from datetime import date
from typing import TYPE_CHECKING
from uuid import UUID

import anyio
from tanya_memory import maintenance_due

from tanya_bot.memory_runtime import MemoryCoordinator, maintained
from tanya_bot.schemas import ChatRequest, ChatResponse
from tanya_bot.telegram_completion import ReplyCompletion
from tanya_bot.telegram_controls import handle_callback, recover_results
from tanya_bot.telegram_core_commands import cancelled_state, local_command
from tanya_bot.telegram_core_context import build_request
from tanya_bot.telegram_core_session import CoreSession
from tanya_bot.telegram_models import (
    REQUEST_CAPACITY,
    CompanionGateway,
    IncomingCallback,
    IncomingText,
    IncomingVoice,
    PendingChat,
    TelegramState,
    Transport,
    VoiceGateway,
    VoiceInput,
)
from tanya_bot.telegram_store import CoreError, StateStore
from tanya_bot.telegram_voice_inputs import reserve_input, voice_text_allowed

if TYPE_CHECKING:
    from tanya_bot.telegram_gmail import GmailFlow


class TelegramController:
    """Serializes local mutations; durable reservations gate every paid call."""

    def __init__(
        self,
        owner_id: int,
        store: StateStore,
        transport: Transport,
        gateway: CompanionGateway,
        voice: VoiceGateway | None = None,
    ) -> None:
        """Inject owner-specific persistence, transport and optional voice gateway."""
        self.owner_id: int = owner_id
        self.store: StateStore = store
        self.transport: Transport = transport
        self.gateway: CompanionGateway = gateway
        self.voice: VoiceGateway | None = voice
        self.memory: MemoryCoordinator | None = None
        self.lock: anyio.Lock = anyio.Lock()
        self.gmail: GmailFlow | None = None

    def accepts(self, sender_id: int, chat_id: int, *, private: bool) -> bool:
        """Root can use this same gate before downloading or transcribing media."""
        return private and sender_id == self.owner_id and (chat_id == self.owner_id)

    def _session(self) -> CoreSession:
        session = CoreSession(self.store, self.transport, voice=self.voice is not None)
        if session.state.owner_id != self.owner_id:
            raise CoreError(code="state_owner")
        return session

    async def on_text(self, event: IncomingText) -> None:
        """Transcribed voice uses this same stable-update reservation path."""
        if not self.accepts(event.sender_id, event.chat_id, private=event.private):
            return
        if self.gmail is not None and await self.gmail.text(event):
            return
        request = await self.begin_text(event)
        if request is not None:
            response = await self.gateway.chat(request)
            await anyio.lowlevel.checkpoint()
            await self.complete_text(request.request_id, response)

    async def cancel_pending(self) -> None:
        """Root calls after cancelling active scopes; persist late-result tombstones."""
        async with self.lock:
            with self.store.exclusive():
                session = self._session()
                session.save(cancelled_state(session.state))

    async def begin_text(self, event: IncomingText) -> ChatRequest | None:
        """Reserve BEFORE a root-managed external call; None means do not generate."""
        if not self.accepts(event.sender_id, event.chat_id, private=event.private):
            return None
        snapshot: TelegramState | None = None
        request: ChatRequest | None = None
        async with self.lock:
            with self.store.exclusive():
                session = self._session()
                if not voice_text_allowed(session.state, event) or not session.claim(
                    f"text:{event.update_id}"
                ):
                    return None
                if event.text.strip() == "/recover":
                    await recover_results(session, self.gateway, self.voice)
                else:
                    request = await self._begin(session, event)
                    if request is not None:
                        if self.memory is None:
                            self._reserve(
                                session,
                                PendingChat(
                                    conversation_id=session.state.conversation_id,
                                    update_id=event.update_id,
                                    request=request,
                                ),
                            )
                            return request
                        snapshot = session.state
        if snapshot is not None and request is not None and (self.memory is not None):
            prepared, turn = await self.memory.prepare(snapshot, request)
            await anyio.lowlevel.checkpoint()
            async with self.lock:
                with self.store.exclusive():
                    session = self._session()
                    if (
                        session.state.conversation_id != snapshot.conversation_id
                        or session.state.context.epoch != snapshot.context.epoch
                        or session.state.context.memory_revision
                        != snapshot.context.memory_revision
                        or session.state.context.summary != snapshot.context.summary
                        or session.state.settings != snapshot.settings
                        or (session.state.requests != snapshot.requests)
                        or session.state.history != snapshot.history
                        or session.state.history_barrier != snapshot.history_barrier
                        or session.state.new_confirmation_token
                        != snapshot.new_confirmation_token
                    ):
                        return None
                    self._reserve(
                        session,
                        PendingChat(
                            conversation_id=session.state.conversation_id,
                            update_id=event.update_id,
                            request=prepared,
                            context=turn,
                        ),
                    )
                    return prepared
        return None

    async def reserve_voice_input(
        self, event: IncomingVoice, request_id: UUID
    ) -> tuple[VoiceInput, bool] | None:
        """Bind authenticated media before downloading or transcribing it."""
        if not self.accepts(event.sender_id, event.chat_id, private=event.private):
            return None
        async with self.lock:
            with self.store.exclusive():
                return reserve_input(self._session(), event, request_id)

    async def _begin(
        self, session: CoreSession, event: IncomingText
    ) -> ChatRequest | None:
        try:
            command = local_command(session.state, event.text)
            if command is not None:
                session.save(command.state)
                await session.say(
                    command.text, command.actions, setting_field=command.setting_field
                )
                return None
            if session.state.new_confirmation_token is not None:
                await session.say("First choose “Start new” or “Stay”.")
                return None
            if any(
                item.response is None
                and item.conversation_id == session.state.conversation_id
                for item in session.state.requests
            ):
                notice = (
                    "The request outcome is unknown. "
                    "/recover will check the cache without repeating it."
                )
                await session.say(notice)
                return None
            request = build_request(session.state, event.text)
        except CoreError as error:
            await session.say(f"Input error: {error.code}. /cancel — cancel.")
            return None
        return request

    @staticmethod
    def _reserve(session: CoreSession, pending: PendingChat) -> None:
        if len(session.state.requests) >= REQUEST_CAPACITY:
            raise CoreError(code="request_capacity")
        session.save(
            session.state.model_copy(
                update={"requests": (*session.state.requests, pending)}
            )
        )

    async def maintain_memory(self, day: date) -> None:
        """Compute locally, then compare revisions before the atomic daily mark."""
        if self.memory is None:
            return
        async with self.lock:
            with self.store.exclusive():
                snapshot = self._session().state
                if not maintenance_due(snapshot.context, day):
                    return
        pairs = await self.memory.maintenance(snapshot)
        async with self.lock:
            with self.store.exclusive():
                session = self._session()
                if (
                    session.state.context.epoch == snapshot.context.epoch
                    and session.state.context.memory_revision
                    == snapshot.context.memory_revision
                    and (session.state.memories == snapshot.memories)
                    and maintenance_due(session.state.context, day)
                ):
                    session.save(
                        session.state.model_copy(
                            update={
                                "context": maintained(session.state.context, pairs, day)
                            }
                        )
                    )

    async def complete_text(self, request_id: UUID, response: ChatResponse) -> None:
        """Trusted root hook: match a saved reservation and commit before sending."""
        async with self.lock:
            with self.store.exclusive():
                await ReplyCompletion(self._session(), self.voice).complete(
                    request_id, response
                )

    async def on_callback(self, event: IncomingCallback) -> None:
        """Reject foreign/private mismatches before reading any private state."""
        if not self.accepts(event.sender_id, event.chat_id, private=event.private):
            await self.transport.ack(event.callback_id, "Unavailable.")
            return
        if self.gmail is not None and await self.gmail.callback(event):
            return
        async with self.lock:
            with self.store.exclusive():
                await handle_callback(self._session(), event, self.voice)
