"""Explicit Gmail operations release bot state locks while the adapter works."""

from collections.abc import Awaitable, Callable
from typing import Final

import anyio

from tanya_bot.gmail_commands import GmailCommands
from tanya_bot.telegram_controls import claim_callback
from tanya_bot.telegram_core_session import CoreSession
from tanya_bot.telegram_models import IncomingCallback, IncomingText, Transport
from tanya_bot.telegram_store import StateStore
from tanya_bot.telegram_voice_inputs import voice_text_allowed

GMAIL_ACTIONS: Final = {
    "gmail": "/gmail status",
    "gmail_status": "/gmail status",
    "gmail_connect": "/gmail connect",
    "gmail_list": "/gmail list",
    "gmail_disconnect": "/gmail disconnect",
}


class GmailFlow:
    """The controller owner gate must precede either entry point."""

    def __init__(
        self,
        store: StateStore,
        transport: Transport,
        lock: anyio.Lock,
        commands: GmailCommands,
        *,
        cancel: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Share the controller transaction lock and one account adapter instance."""
        self.store: StateStore = store
        self.transport: Transport = transport
        self.lock: anyio.Lock = lock
        self.commands: GmailCommands = commands
        self.cancel: Callable[[], Awaitable[None]] | None = cancel

    def _session(self) -> CoreSession:
        return CoreSession(self.store, self.transport, voice=False)

    async def text(self, event: IncomingText) -> bool:
        """Claim every explicit command before executing it outside the lock."""
        if not self.commands.is_command(event.text):
            return False
        async with self.lock:
            with self.store.exclusive():
                session = self._session()
                if not voice_text_allowed(session.state, event) or not session.claim(
                    f"text:{event.update_id}"
                ):
                    return True
                conversation = session.state.conversation_id
                language = session.state.settings.language
                session.require_reply_capacity()
        await self._execute(event.text, conversation, language)
        return True

    async def callback(self, event: IncomingCallback) -> bool:
        """Only saved static Gmail action IDs can invoke the adapter."""
        command = GMAIL_ACTIONS.get(event.data.partition(":")[2])
        if command is None:
            return False
        async with self.lock:
            with self.store.exclusive():
                session = self._session()
                if await claim_callback(session, event) is None:
                    return True
                conversation = session.state.conversation_id
                language = session.state.settings.language
                session.require_reply_capacity()
        await self._execute(command, conversation, language)
        return True

    async def _execute(self, command: str, conversation: int, language: str) -> None:
        if command == "/gmail disconnect" and self.cancel is not None:
            await self.cancel()
        output = await self.commands.handle(command, language)
        async with self.lock:
            with self.store.exclusive():
                session = self._session()
                if session.state.conversation_id == conversation:
                    await session.say_plain(output.text, output.actions)
