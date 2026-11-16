"""Run the new personal bot using its dedicated locked Telethon session."""

import fcntl
import logging
import os
from typing import Final

import anyio
from pydantic import ValidationError
from telethon import TelegramClient, events
from telethon.errors import RPCError
from telethon.tl.functions.bots import (
    SetBotCommandsRequest,
    SetBotInfoRequest,
    SetBotMenuButtonRequest,
)
from telethon.tl.types import (
    BotCommand,
    BotCommandScopeDefault,
    BotMenuButtonCommands,
    InputUserEmpty,
)

from tanya_bot.config import STATE_DIRECTORY
from tanya_bot.i18n import localize
from tanya_bot.receipts import private_file
from tanya_bot.runtime_assembly import assemble, daily_maintenance
from tanya_bot.telegram_config import (
    BOT_DIRECTORY,
    BotIdentity,
    SDKCredentials,
    bot_token,
    identity,
)
from tanya_bot.telegram_controller import TelegramController
from tanya_bot.telegram_gateway import LocalCompanionGateway
from tanya_bot.telegram_ingress import TelegramIngress
from tanya_bot.telegram_store import CoreError, TelegramStore
from tanya_bot.telegram_transport import TelethonTransport
from tanya_bot.telegram_voice import TelegramVoiceGateway

COMMANDS: Final = (
    ("start", "Open assistant"),
    ("settings", "Personality, voice, and settings"),
    ("memory", "Character memory"),
    ("journal", "Personal journal"),
    ("new", "Start an independent conversation"),
    ("gmail", "Personal email: setup status"),
    ("cancel", "Stop current action"),
    ("recover", "Check the saved result without repeating"),
    ("help", "Help and commands"),
)
LOGGER: Final = logging.getLogger(__name__)


async def authenticate_bot(config: BotIdentity, credentials: SDKCredentials) -> None:
    """Allow SDK DC migration only during unpaid login; receive no chat updates."""
    client = TelegramClient(
        str(BOT_DIRECTORY / "sdk"),
        credentials.telegram_api_id,
        credentials.telegram_api_hash.get_secret_value(),
        request_retries=2,
        flood_sleep_threshold=0,
        raise_last_call_error=True,
        receive_updates=False,
    )
    await client.connect()
    try:
        if not await client.is_user_authorized():
            _ = await client.sign_in(bot_token=bot_token().get_secret_value())
        me = await client.get_me()
        if me is None or not me.bot or me.id != config.bot_id:
            raise CoreError(code="bot_identity_mismatch")
    finally:
        with anyio.CancelScope(shield=True):
            await client.disconnect()


async def run() -> None:
    """A separate bot process cannot alter iOS data or another bot's session."""
    config = identity()
    credentials = SDKCredentials.model_validate({})
    await authenticate_bot(config, credentials)
    voice = TelegramVoiceGateway(BOT_DIRECTORY / "audio")
    client = TelegramClient(
        str(BOT_DIRECTORY / "sdk"),
        credentials.telegram_api_id,
        credentials.telegram_api_hash.get_secret_value(),
        request_retries=0,
        flood_sleep_threshold=0,
        raise_last_call_error=True,
        catch_up=True,
    )
    transport = TelethonTransport(client, config.owner_id, voice)
    store = TelegramStore(BOT_DIRECTORY / "core", config.owner_id)
    with store.exclusive():
        settings = store.load().settings
        display_name = settings.name[:32]
    controller = TelegramController(
        config.owner_id,
        store,
        transport,
        LocalCompanionGateway(STATE_DIRECTORY, core=store),
        voice,
    )
    await assemble(controller, BOT_DIRECTORY)
    ingress = TelegramIngress(client, controller, transport, voice, config.bot_id)
    client.add_event_handler(ingress.on_message, events.NewMessage(incoming=True))
    client.add_event_handler(ingress.on_callback, events.CallbackQuery())
    await client.connect()
    try:
        if not await client.is_user_authorized():
            _ = await client.sign_in(bot_token=bot_token().get_secret_value())
        me = await client.get_me()
        if me is None or not me.bot or me.id != config.bot_id:
            raise CoreError(code="bot_identity_mismatch")
        _ = await client(
            SetBotInfoRequest(
                "",
                name=display_name,
                about=localize(
                    "Personal AI assistant. Personality, memory, synthesized voice.",
                    settings.language,
                ),
            )
        )
        _ = await client(
            SetBotCommandsRequest(
                BotCommandScopeDefault(),
                "",
                [
                    BotCommand(command, localize(description, settings.language))
                    for command, description in COMMANDS
                ],
            )
        )
        _ = await client(
            SetBotMenuButtonRequest(InputUserEmpty(), BotMenuButtonCommands())
        )
        LOGGER.warning("telegram_ready:%s", config.username)
        async with anyio.create_task_group() as tasks:
            _ = tasks.start_soon(daily_maintenance, controller)
            await client.disconnected
            tasks.cancel_scope.cancel()
    finally:
        with anyio.CancelScope(shield=True):
            await client.disconnect()


def main() -> None:
    """Fail closed on an absent identity or concurrent runtime process."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(message)s")
    logging.getLogger("telethon").setLevel(logging.CRITICAL)
    _ = os.umask(0o077)
    try:
        _ = identity()
        with os.fdopen(private_file(BOT_DIRECTORY / "runtime.lock"), "rb") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            anyio.run(run)
    except (CoreError, RPCError, OSError, ValidationError) as error:
        LOGGER.warning("telegram_start_failed:%s", type(error).__name__)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
