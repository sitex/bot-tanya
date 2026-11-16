"""Validate owner events before media access and cancel active work explicitly."""

import logging
from typing import Final
from uuid import NAMESPACE_URL, uuid5

import anyio
import httpx2
from pydantic import ValidationError
from telethon import TelegramClient, events
from telethon.errors import RPCError

from tanya_bot.errors import ServiceError
from tanya_bot.i18n import localize
from tanya_bot.telegram_controller import TelegramController
from tanya_bot.telegram_models import (
    GatewayFailure,
    IncomingCallback,
    IncomingText,
    IncomingVoice,
)
from tanya_bot.telegram_store import CoreError
from tanya_bot.telegram_transport import TelethonTransport
from tanya_bot.telegram_voice import TelegramVoiceGateway

MAX_VOICE_BYTES: Final = 5 * 1024 * 1024
MAX_VOICE_SECONDS: Final = 120
MAX_TEXT: Final = 16000
LOGGER: Final = logging.getLogger(__name__)
CAPACITY_NOTICE: Final = (
    "Storage is full or reserved for unresolved requests. No new chat request was sent."
)
MESSAGE_HELD: Final = "/recover will check the cache. The request will not be repeated."
CALLBACK_HELD: Final = "Action stopped. /recover — check the cache without repeating."


def _failure_notice(code: str, fallback: str) -> str:
    return (
        CAPACITY_NOTICE
        if code in {"reply_capacity_admission", "request_capacity", "update_capacity"}
        else fallback
    )


class TelegramIngress:
    """One cancellable owner operation; foreign events perform no other action."""

    def __init__(
        self,
        client: TelegramClient,
        controller: TelegramController,
        transport: TelethonTransport,
        voice: TelegramVoiceGateway,
        bot_id: int,
    ) -> None:
        """Retain only the new bot's dependencies, with no user-session access."""
        self.client: TelegramClient = client
        self.controller: TelegramController = controller
        self.transport: TelethonTransport = transport
        self.voice: TelegramVoiceGateway = voice
        self.bot_id: int = bot_id
        self.active: anyio.CancelScope | None = None

    async def _notice(self, text: str) -> None:
        try:
            with self.controller.store.exclusive():
                language = self.controller.store.load().settings.language
            _ = await self.transport.send_text(
                self.controller.owner_id, localize(text, language), ()
            )
        except (RPCError, OSError, TimeoutError):
            LOGGER.warning("telegram_notice_delivery_unknown")

    async def _message(self, event: events.NewMessage.Event, key: str) -> None:
        text = event.raw_text
        conversation_id: int | None = None
        if event.message.voice is not None:
            metadata = event.message.file
            if (
                metadata is None
                or metadata.size is None
                or (not 0 < metadata.size <= MAX_VOICE_BYTES)
                or (metadata.duration is None)
                or (not 0 < metadata.duration <= MAX_VOICE_SECONDS)
            ):
                await self._notice(
                    "Voice message: no longer than 2 minutes and no larger than 5 MB."
                )
                return
            request_id = uuid5(NAMESPACE_URL, f"tolan:{self.bot_id}:voice:{key}")
            reservation = await self.controller.reserve_voice_input(
                IncomingVoice(
                    update_id=key,
                    sender_id=self.controller.owner_id,
                    chat_id=self.controller.owner_id,
                    private=True,
                ),
                request_id,
            )
            if reservation is None:
                return
            reserved, fresh = reservation
            conversation_id = reserved.conversation_id
            with anyio.fail_after(40):
                audio = await self.client.download_media(event.message, bytes)
            if audio is None or len(audio) != metadata.size:
                raise CoreError(code="voice_download_invalid")
            transcript = (
                await self.voice.transcribe(request_id, audio)
                if fresh
                else await self.voice.recover_transcript(request_id, audio)
            )
            if transcript is None:
                await self._notice(
                    "The recognition outcome is unknown. It will not be repeated."
                )
                return
            text = transcript.text
        elif not text:
            await self._notice("Write a message or send a voice message.")
            return
        if len(text) > MAX_TEXT:
            await self._notice("Message too long. Shorten it to 16,000 characters.")
            return
        await anyio.lowlevel.checkpoint()
        await self.controller.on_text(
            IncomingText(
                update_id=key,
                conversation_id=conversation_id,
                sender_id=self.controller.owner_id,
                chat_id=self.controller.owner_id,
                private=True,
                text=text,
            )
        )

    async def on_message(self, event: events.NewMessage.Event) -> None:
        """Stable chat/message identity also binds the transcription reservation."""
        if not self.controller.accepts(
            event.sender_id or 0, event.chat_id or 0, private=event.is_private
        ):
            return
        command = event.raw_text.split(maxsplit=1)[0] if event.raw_text else ""
        control = command in {"/cancel", "/new"}
        if self.active is not None and (not control):
            await self._notice("Finishing the previous request. /cancel — stop.")
            return
        scope = anyio.CancelScope()
        previous = self.active
        self.active = scope
        try:
            if control:
                if previous is not None:
                    previous.cancel()
                await self.controller.cancel_pending()
            with scope:
                await self._message(event, f"{event.chat_id}:{event.id}")
        except CoreError as error:
            LOGGER.warning("telegram_operation_held:%s", type(error).__name__)
            await self._notice(_failure_notice(error.code, MESSAGE_HELD))
        except (
            ServiceError,
            GatewayFailure,
            httpx2.HTTPError,
            RPCError,
            OSError,
            TimeoutError,
            ValidationError,
        ) as error:
            LOGGER.warning("telegram_operation_held:%s", type(error).__name__)
            await self._notice(MESSAGE_HELD)
        finally:
            if self.active is scope:
                self.active = None

    async def on_callback(self, event: events.CallbackQuery.Event) -> None:
        """Reject foreign and malformed callback bytes before invoking the core."""
        if not self.controller.accepts(
            event.sender_id or 0, event.chat_id or 0, private=event.is_private
        ):
            return
        if event.data is None:
            return
        if self.active is not None:
            with self.controller.store.exclusive():
                language = self.controller.store.load().settings.language
            await self.transport.ack(
                str(event.id),
                localize("Finishing the request; /cancel — cancel.", language),
            )
            return
        scope = anyio.CancelScope()
        self.active = scope
        try:
            with scope:
                await self.controller.on_callback(
                    IncomingCallback(
                        update_id=f"callback:{event.id}",
                        callback_id=str(event.id),
                        sender_id=self.controller.owner_id,
                        chat_id=self.controller.owner_id,
                        private=True,
                        message_id=event.message_id,
                        data=event.data.decode("utf-8"),
                    )
                )
        except CoreError as error:
            LOGGER.warning("telegram_callback_held:%s", type(error).__name__)
            await self._notice(_failure_notice(error.code, CALLBACK_HELD))
        except (
            ServiceError,
            GatewayFailure,
            httpx2.HTTPError,
            RPCError,
            OSError,
            TimeoutError,
            ValidationError,
            UnicodeError,
        ) as error:
            LOGGER.warning("telegram_callback_held:%s", type(error).__name__)
            await self._notice(CALLBACK_HELD)
        finally:
            if self.active is scope:
                self.active = None
