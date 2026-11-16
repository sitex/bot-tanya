"""Narrow Telethon adapter: plain text, opaque buttons and saved voice files."""

from pathlib import Path
from typing import Protocol

from telethon import Button, TelegramClient
from telethon.errors import MessageNotModifiedError
from telethon.tl.functions.messages import SetBotCallbackAnswerRequest
from telethon.tl.types import KeyboardInlineButton

from tanya_bot.telegram_models import Keyboard, VoiceResult
from tanya_bot.telegram_store import CoreError


class ArtifactResolver(Protocol):
    """Resolve a validated durable artifact without generating any audio."""

    def resolve(self, result: VoiceResult) -> Path:
        """Refuse missing, unsafe or changed artifacts."""
        ...


def buttons(keyboard: Keyboard) -> list[list[KeyboardInlineButton]] | None:
    """Translate the exact opaque data produced by the core."""
    return (
        [
            [Button.inline(button.text, button.data.encode("utf-8")) for button in row]
            for row in keyboard
        ]
        if keyboard
        else None
    )


class TelethonTransport:
    """Send only to the configured personal owner, even if core is misused."""

    def __init__(
        self, client: TelegramClient, owner_id: int, artifacts: ArtifactResolver
    ) -> None:
        """Keep the bot SDK and audio resolver outside the SDK-independent core."""
        self.client: TelegramClient = client
        self.owner_id: int = owner_id
        self.artifacts: ArtifactResolver = artifacts

    def _owner(self, chat_id: int) -> None:
        if chat_id != self.owner_id:
            raise CoreError(code="telegram_owner_required")

    async def send_text(self, chat_id: int, text: str, keyboard: Keyboard) -> int:
        """Disable formatting and link previews; never retry a failed send."""
        self._owner(chat_id)
        message = await self.client.send_message(
            chat_id,
            text,
            buttons=buttons(keyboard),
            parse_mode=None,
            link_preview=False,
        )
        return message.id

    async def edit_keyboard(
        self, chat_id: int, message_id: int, keyboard: Keyboard
    ) -> None:
        """An already identical keyboard is a successful idempotent edit."""
        self._owner(chat_id)
        try:
            _ = await self.client.edit_message(
                chat_id, message_id, buttons=buttons(keyboard)
            )
        except MessageNotModifiedError:
            return

    async def ack(self, callback_id: str, text: str) -> None:
        """Acknowledge only a callback already accepted by the owner gate."""
        _ = await self.client(
            SetBotCallbackAnswerRequest(
                int(callback_id), cache_time=0, message=text, alert=False
            )
        )

    async def send_voice(self, chat_id: int, result: VoiceResult) -> int:
        """Upload a saved Ogg/Opus file as a native playable Telegram voice note."""
        self._owner(chat_id)
        message = await self.client.send_file(
            chat_id, self.artifacts.resolve(result), voice_note=True, parse_mode=None
        )
        return message.id
