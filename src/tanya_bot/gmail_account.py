"""One lazy account and lock shared by commands and owner dialogue tools."""

from pathlib import Path

import anyio
import httpx2
from tanya_gmail import (
    ConnectionStatus,
    FileTokenStore,
    GmailClient,
    MessagePage,
    MessagePreview,
)

from tanya_bot.gmail_config import load_config


class GmailAccount:
    """Construction and local status never refresh tokens or read the mailbox."""

    def __init__(
        self,
        config: Path,
        tokens: Path,
        transport: httpx2.BaseTransport | None = None,
    ) -> None:
        """Retain external paths without opening them before owner authorization."""
        self.config: Path = config
        self.tokens: Path = tokens
        self.transport: httpx2.BaseTransport | None = transport
        self.cached: GmailClient | None = None
        self.lock: anyio.Lock = anyio.Lock()

    def client(self) -> GmailClient:
        """Caller holds the account lock in a worker for all client operations."""
        if self.cached is None:
            self.cached = GmailClient(
                load_config(self.config),
                FileTokenStore(self.tokens),
                transport=self.transport,
            )
        return self.cached

    def restore(self) -> GmailClient:
        """An explicit mailbox gesture may restore and reverify the saved owner."""
        client = self.client()
        if client.status() is not ConnectionStatus.CONNECTED:
            _ = client.connect(interactive=False)
        return client

    async def search(self, query: str, limit: int) -> MessagePage:
        """Run exactly one page request off the event loop under the account lock."""

        def search() -> MessagePage:
            return self.restore().list_messages(query=query, page_size=limit)

        async with self.lock:
            return await anyio.to_thread.run_sync(search)

    async def read(self, message_id: str, limit: int) -> MessagePreview:
        """Read one bounded inert preview without attachments or remote resources."""

        def read() -> MessagePreview:
            return self.restore().read_message(message_id, preview_chars=limit)

        async with self.lock:
            return await anyio.to_thread.run_sync(read)
