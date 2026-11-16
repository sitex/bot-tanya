"""Assemble external configuration without embedding private deployment paths."""

import logging
import os
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import anyio
from pydantic import ValidationError
from tanya_memory import Encoder, MemoryOperationError
from tanya_memory.manifest import ModelManifest, pinned_manifest

from tanya_bot.dialogue_gateway import LocalPhaseGateway
from tanya_bot.dialogue_runner import DialogueRunner, DialogueServices
from tanya_bot.dialogue_sources import prune_sources
from tanya_bot.dialogue_store import DialogueStore
from tanya_bot.errors import ServiceError
from tanya_bot.gmail_commands import GmailCommands
from tanya_bot.memory_runtime import MemoryCoordinator
from tanya_bot.receipts import exclusive_store
from tanya_bot.telegram_controller import TelegramController
from tanya_bot.telegram_gateway import LocalCompanionGateway
from tanya_bot.telegram_gmail import GmailFlow
from tanya_bot.telegram_store import CoreError

_encoder_factory: Callable[[Path, ModelManifest], Encoder] | None
try:
    from tanya_memory.encoder import SentenceEncoder
except ImportError:
    _encoder_factory = None
else:
    _encoder_factory = SentenceEncoder

ENCODER_FACTORY: Final = _encoder_factory

LOGGER: Final = logging.getLogger(__name__)
MAINTENANCE_SECONDS: Final = 3600
DIALOGUE_CONTEXT_CHARACTERS: Final = 8000


def configured_memory(directory: Path) -> MemoryCoordinator:
    """Only an explicit verified local snapshot enables CPU semantic inference."""
    location = os.environ.get("TANYA_MEMORY_MODEL_DIR")
    encoder: Encoder | None = None
    if location:
        if ENCODER_FACTORY is None:
            LOGGER.warning("memory_lexical:local_extra_missing")
        else:
            try:
                encoder = ENCODER_FACTORY(Path(location), pinned_manifest())
            except (MemoryOperationError, OSError, ValueError, RuntimeError):
                LOGGER.warning("memory_lexical:local_model_unavailable")
    else:
        LOGGER.warning("memory_lexical:model_not_configured")
    return MemoryCoordinator(directory, encoder)


async def assemble(controller: TelegramController, directory: Path) -> None:
    """Attach local memory and one lazy Gmail adapter without changing profiles."""
    controller.memory = await anyio.to_thread.run_sync(
        configured_memory, directory / "memory-index"
    )
    config = Path(os.environ.get("TANYA_GMAIL_CONFIG", str(directory / "gmail.json")))
    tokens = Path(
        os.environ.get("TANYA_GMAIL_TOKEN_DIR", str(directory / "gmail-tokens"))
    )
    commands = GmailCommands(config, tokens)
    controller.gmail = GmailFlow(
        controller.store,
        controller.transport,
        controller.lock,
        commands,
        cancel=controller.cancel_pending,
    )
    if isinstance(controller.gateway, LocalCompanionGateway):
        controller.memory.context_budget = DIALOGUE_CONTEXT_CHARACTERS
        gateway = controller.gateway
        with exclusive_store(gateway.directory):
            pass
        dialogue = DialogueRunner(
            controller.store,
            DialogueServices(
                DialogueStore(gateway.directory / "dialogue-v3"),
                commands.account,
                LocalPhaseGateway(gateway.transport),
            ),
        )
        gateway.dialogue = dialogue
        controller.dialogue = dialogue


async def daily_maintenance(controller: TelegramController) -> None:
    """Check at startup and hourly; failed or stale passes never mark success."""
    while True:
        try:
            if controller.dialogue is not None:
                prune_sources(
                    controller.dialogue.services.store, datetime.now(UTC).timestamp()
                )
            await controller.maintain_memory(datetime.now(UTC).date())
        except (
            CoreError,
            MemoryOperationError,
            sqlite3.Error,
            OSError,
            ValidationError,
            ServiceError,
        ):
            LOGGER.warning("memory_maintenance_deferred")
        await anyio.sleep(MAINTENANCE_SECONDS)
