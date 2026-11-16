"""CPU inference and daily maintenance stay outside the controller state lock."""

import logging
import sqlite3
from datetime import date
from pathlib import Path
from typing import Final
from uuid import uuid4

import anyio
from tanya_memory import (
    ContextState,
    Encoder,
    MemoryIndex,
    MemoryRecord,
    RelatedPair,
    TurnContext,
    VectorCache,
    maintain,
)

from tanya_bot.schemas import CONTEXT_CHARACTERS, ChatRequest
from tanya_bot.telegram_core_context import build_request, context_history
from tanya_bot.telegram_models import TelegramState

LOGGER: Final = logging.getLogger(__name__)


class MemoryCoordinator:
    """Serialize model/cache work; immutable snapshots cross the worker boundary."""

    def __init__(self, directory: Path, encoder: Encoder | None = None) -> None:
        """Bind an optional local encoder and a rebuildable private vector cache."""
        try:
            cache = VectorCache(directory)
        except sqlite3.Error:
            LOGGER.warning("memory_cache_rebuild:corrupt_sqlite")
            cache = VectorCache(directory / f"rebuild-{uuid4().hex}")
        self.index: MemoryIndex = MemoryIndex(encoder, cache)
        self.limiter: anyio.CapacityLimiter = anyio.CapacityLimiter(1)

    async def prepare(
        self,
        state: TelegramState,
        request: ChatRequest,
    ) -> tuple[ChatRequest, TurnContext]:
        """No reservation or provider operation occurs during local retrieval."""
        text = request.messages[-1].content
        records = tuple(
            MemoryRecord(id=item.id, text=item.text) for item in state.memories
        )
        selected = await anyio.to_thread.run_sync(
            self.index.search,
            text,
            records,
            limiter=self.limiter,
        )
        if selected.diagnostic is not None:
            LOGGER.warning("memory_retrieval:%s:%s", selected.mode, selected.diagnostic)
        fixed = len(text) + len(state.settings.name) + len(state.settings.language)
        fixed += state.settings.profile.context_size()
        summary = state.context.summary
        allowance = max(0, CONTEXT_CHARACTERS - fixed)
        if len(summary.text) > allowance:
            summary = summary.model_copy(
                update={
                    "text": summary.text[:allowance],
                    "coverage_gap": True,
                }
            )
        context_state = state.model_copy(
            update={
                "context": state.context.model_copy(update={"summary": summary}),
            }
        )
        prepared = build_request(
            context_state, text, tuple(item.text for item in selected.records)
        )
        prepared = prepared.model_copy(update={"request_id": request.request_id})
        amount = len(prepared.messages) - 1
        history = context_history(state)
        sources = tuple(item.id for item, _ in history[-amount:]) if amount else ()
        covered = {*sources, *state.context.summary.source_ids}
        gap = any(item.id not in covered for item, _ in history)
        turn = TurnContext(
            epoch=state.context.epoch,
            memory_revision=state.context.memory_revision,
            summary=summary,
            source_ids=sources,
            coverage_gap=gap or summary.coverage_gap,
            user_entry_id=uuid4(),
            assistant_entry_id=uuid4(),
        )
        return prepared, turn

    async def maintenance(self, state: TelegramState) -> tuple[RelatedPair, ...]:
        """An adapter commits these candidates only if its snapshot is still current."""
        records = tuple(
            MemoryRecord(id=item.id, text=item.text) for item in state.memories
        )
        return await anyio.to_thread.run_sync(
            maintain,
            state.context,
            records,
            self.index,
            limiter=self.limiter,
        )


def maintained(
    context: ContextState, pairs: tuple[RelatedPair, ...], day: date
) -> ContextState:
    """This local mark is persisted atomically with the exact candidate set."""
    return context.model_copy(update={"maintenance_day": day, "related": pairs})
