"""An owner-bound turn uses one decision and at most one mail finalization."""

import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

import anyio

from tanya_bot.config import Settings
from tanya_bot.dialogue_authority import authority
from tanya_bot.dialogue_models import (
    DialoguePhase,
    DialogueRoot,
    PhaseAddress,
    PhaseReceipt,
)
from tanya_bot.dialogue_results import assembled
from tanya_bot.dialogue_sources import cancel_roots, prune_sources, recent_selection
from tanya_bot.dialogue_store import DialogueStore, phase_for
from tanya_bot.dialogue_tools import MailTools
from tanya_bot.errors import ServiceError
from tanya_bot.gmail_account import GmailAccount
from tanya_bot.schemas import ChatRequest
from tanya_bot.telegram_models import PendingChat
from tanya_bot.telegram_store import StateStore
from tanya_bot.turns import TurnRequest, TurnResponse


class PhaseGateway(Protocol):
    """The API executes an already committed phase; no implicit recovery sends."""

    async def phase(self, address: PhaseAddress) -> PhaseReceipt:
        """Return one saved provider result or preserve its unknown hold."""
        ...


@dataclass(frozen=True, slots=True)
class DialogueServices:
    """The bot retains Gmail credentials; the phase API receives only source data."""

    store: DialogueStore
    account: GmailAccount
    gateway: PhaseGateway


class DialogueRunner:
    """Legacy pending requests never pass through this new protocol."""

    def __init__(self, core: StateStore, services: DialogueServices) -> None:
        """Share core authority without holding its lock across external work."""
        self.core: StateStore = core
        self.services: DialogueServices = services
        self.tools: MailTools = MailTools(
            services.store, services.account, self.authority
        )

    def authority(self, root: DialogueRoot) -> AbstractContextManager[None]:
        """Every nested acquisition follows core then sidecar ordering."""
        return authority(self.core, root)

    def guard(self, root: DialogueRoot) -> None:
        """Reject cancelled, archived, changed or mismatched owner authority."""
        with self.authority(root):
            pass

    def _prepare(self, request: ChatRequest) -> DialogueRoot:
        store = self.services.store
        now = time.time()
        prune_sources(store, now)
        with self.core.exclusive():
            state = self.core.load()
            pending = next(
                (item for item in state.requests if item.request == request),
                None,
            )
            if (
                pending is None
                or pending.context is None
                or pending.dialogue_version is None
            ):
                raise ServiceError(409, "dialogue_missing", "Missing v3 reservation.")
            selection = recent_selection(
                store, state.owner_id, state.conversation_id, now
            )
            root = DialogueRoot(
                turn=TurnRequest(chat=request, context=pending.context),
                owner_id=state.owner_id,
                conversation_id=pending.conversation_id,
                update_id=pending.update_id,
                model=Settings().model,
                created_at=now,
                selection=selection,
                tainted=selection is not None,
                phases=(DialoguePhase(kind="decide"),),
            )
            if pending.cancelled or state.conversation_id != pending.conversation_id:
                raise ServiceError(409, "dialogue_cancelled", "Turn is cancelled.")
            if root.tainted:
                self._taint_locked(pending)
            store.create(root)
        return root

    def _taint_locked(self, pending: PendingChat) -> None:
        state = self.core.load()
        self.core.save(
            state.model_copy(
                update={
                    "requests": tuple(
                        item.model_copy(update={"mail_tainted": True})
                        if item.request.request_id == pending.request.request_id
                        else item
                        for item in state.requests
                    ),
                }
            )
        )

    def _taint(self, root: DialogueRoot) -> DialogueRoot:
        self.guard(root)
        with self.core.exclusive():
            state = self.core.load()
            pending = next(
                item for item in state.requests if item.request == root.turn.chat
            )
            self._taint_locked(pending)
        return self.services.store.mutate(
            root.turn.chat.request_id,
            lambda current: current.model_copy(update={"tainted": True}),
        )

    async def chat(self, request: ChatRequest) -> TurnResponse:
        """A duplicate or crash never starts an unexecuted follow-up phase."""
        root = self._prepare(request)
        self.guard(root)
        first = await self.services.gateway.phase(
            PhaseAddress(root_id=request.request_id, kind="decide")
        )
        self.guard(root)
        await anyio.lowlevel.checkpoint()
        if first.tool is None:
            return self._finish(request.request_id, first)
        root = self._taint(root)
        result = await self.tools.execute(root, first.tool)
        self.guard(root)
        await anyio.lowlevel.checkpoint()
        root = self.services.store.mutate(
            request.request_id,
            lambda current: current.model_copy(
                update={
                    "tool_result": result,
                    "phases": (*current.phases, DialoguePhase(kind="final")),
                }
            ),
        )
        self.guard(root)
        final = await self.services.gateway.phase(
            PhaseAddress(root_id=request.request_id, kind="final")
        )
        self.guard(root)
        await anyio.lowlevel.checkpoint()
        return self._finish(request.request_id, final)

    def _finish(self, root_id: UUID, receipt: PhaseReceipt) -> TurnResponse:
        def finish(root: DialogueRoot) -> DialogueRoot:
            if root.cancelled:
                raise ServiceError(409, "dialogue_cancelled", "Turn is cancelled.")
            return assembled(root, receipt)

        saved = self.services.store.load(root_id)
        if saved is None:
            raise ServiceError(409, "dialogue_missing", "Dialogue is unavailable.")
        with self.authority(saved):
            root = self.services.store.mutate(root_id, finish)
        if root.final is None:
            raise ServiceError(
                502, "dialogue_invalid_final", "Final answer is unavailable."
            )
        return root.final

    def recover(self, request: ChatRequest) -> TurnResponse | None:
        """Cache only: never call Google or advance from a cached tool proposal."""
        prune_sources(self.services.store, time.time())
        root = self.services.store.load(request.request_id)
        if root is None:
            return None
        if root.turn.chat != request:
            raise ServiceError(409, "dialogue_conflict", "Frozen request changed.")
        if root.final is not None:
            return root.final
        phase = (
            phase_for(root, "final")
            if any(item.kind == "final" for item in root.phases)
            else phase_for(root, "decide")
        )
        if phase.receipt is not None and phase.receipt.answer is not None:
            receipt = phase.receipt
            cached = self.services.store.mutate(
                request.request_id, lambda current: assembled(current, receipt)
            )
            return cached.final
        return None

    def cancel(self) -> None:
        """Called under core cancellation after its durable tombstones are saved."""
        state = self.core.load()
        ids = tuple(
            item.request.request_id
            for item in state.requests
            if item.dialogue_version is not None
        )
        cancel_roots(self.services.store, ids)
