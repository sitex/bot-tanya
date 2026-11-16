"""Execute at most one owner-bound readonly tool with durable sub-operation holds."""

import hashlib
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import assert_never
from uuid import UUID

import anyio
from tanya_gmail import GmailError

from tanya_bot.dialogue_models import (
    DialogueRoot,
    MailCall,
    MailOperation,
    MailRead,
    MailResult,
    MailSearch,
    MailSource,
)
from tanya_bot.dialogue_references import bound_refs
from tanya_bot.dialogue_sources import selection_valid
from tanya_bot.dialogue_store import DialogueStore
from tanya_bot.errors import ServiceError
from tanya_bot.gmail_account import GmailAccount


class MailTools:
    """The guard checks current core authority before every Google operation."""

    def __init__(
        self,
        store: DialogueStore,
        account: GmailAccount,
        authority: Callable[[DialogueRoot], AbstractContextManager[None]],
    ) -> None:
        """Keep mailbox execution in the bot, outside its primary state lock."""
        self.store: DialogueStore = store
        self.account: GmailAccount = account
        self.authority: Callable[[DialogueRoot], AbstractContextManager[None]] = (
            authority
        )

    def _guard(self, root: DialogueRoot) -> None:
        with self.authority(root):
            pass

    def _reserve(self, root_id: UUID, key: str) -> DialogueRoot:
        def reserve(root: DialogueRoot) -> DialogueRoot:
            if root.cancelled:
                raise ServiceError(409, "dialogue_cancelled", "Turn is cancelled.")
            if any(item.key == key for item in root.operations):
                raise ServiceError(
                    409, "mail_operation_held", "Mail operation is held."
                )
            return root.model_copy(
                update={"operations": (*root.operations, MailOperation(key=key))}
            )

        saved = self.store.load(root_id)
        if saved is None:
            raise ServiceError(409, "dialogue_missing", "Dialogue is unavailable.")
        with self.authority(saved):
            return self.store.mutate(root_id, reserve)

    def _commit(self, root_id: UUID, operation: MailOperation) -> None:
        _ = self.store.mutate(
            root_id,
            lambda root: root.model_copy(
                update={
                    "operations": tuple(
                        operation if item.key == operation.key else item
                        for item in root.operations
                    )
                }
            ),
        )

    async def execute(self, root: DialogueRoot, call: MailCall) -> MailResult:
        """Validate selection membership; no model-supplied raw ID is executable."""
        self._guard(root)
        sources: list[MailSource] = []
        query = ""
        more = False
        try:
            match call.action:
                case MailSearch(query=chosen, limit=limit):
                    query = chosen.strip() or "newer_than:7d"
                    _ = self._reserve(root.turn.chat.request_id, "list")
                    await anyio.lowlevel.checkpoint()
                    page = await self.account.search(query, limit)
                    ids = tuple(item.id for item in page.messages)
                    more = page.next_page_token is not None
                    self._commit(
                        root.turn.chat.request_id,
                        MailOperation(
                            key="list", completed=True, ids=ids, has_more=more
                        ),
                    )
                    preview_limit = 2400
                case MailRead(refs=refs):
                    self._read_current(root)
                    selected = root.selection.result.sources if root.selection else ()
                    mapping = {item.ref: item.message_id for item in selected}
                    resolved = bound_refs(
                        root.turn.chat.messages[-1].content, refs, selected
                    )
                    if len(set(resolved)) != len(resolved) or any(
                        ref not in mapping for ref in resolved
                    ):
                        raise ServiceError(
                            422, "mail_reference_invalid", "Select a current message."
                        )
                    ids = tuple(mapping[ref] for ref in resolved)
                    preview_limit = 4000
                case _:
                    assert_never(call.action)
            for index, message_id in enumerate(ids):
                if isinstance(call.action, MailRead):
                    self._read_current(root)
                current = self._reserve(root.turn.chat.request_id, f"read:{index}")
                self._guard(current)
                await anyio.lowlevel.checkpoint()
                preview = await self.account.read(message_id, preview_limit)
                source = MailSource(
                    ref="M" + hashlib.sha256(message_id.encode()).hexdigest()[:12],
                    message_id=message_id,
                    subject=preview.subject,
                    sender=preview.sender,
                    date=preview.date,
                    text=preview.text,
                    truncated=preview.truncated,
                    position=next(
                        (
                            item.position or number
                            for number, item in enumerate(
                                root.selection.result.sources, start=1
                            )
                            if item.message_id == message_id
                        ),
                        index + 1,
                    )
                    if isinstance(call.action, MailRead) and root.selection
                    else index + 1,
                )
                remaining = max(
                    0,
                    16000
                    - sum(len(item.model_dump_json()) for item in sources)
                    - len(source.model_dump_json(exclude={"text"}))
                    - 256,
                )
                source = source.model_copy(
                    update={
                        "text": source.text[:remaining],
                        "truncated": source.truncated or len(source.text) > remaining,
                    }
                )
                self._commit(
                    root.turn.chat.request_id,
                    MailOperation(key=f"read:{index}", completed=True, source=source),
                )
                sources.append(source)
                self._guard(current)
                await anyio.lowlevel.checkpoint()
        except GmailError as error:
            return MailResult(
                sources=tuple(sources),
                query=query,
                has_more=more,
                error=error.code.value,
                retrieved_at=time.time(),
            )
        except FileNotFoundError:
            return MailResult(
                error="not_configured",
                retrieved_at=time.time(),
            )
        return MailResult(
            sources=tuple(sources),
            query=query,
            has_more=more,
            retrieved_at=time.time(),
        )

    @staticmethod
    def _read_current(root: DialogueRoot) -> None:
        if root.selection is None:
            raise ServiceError(
                422, "mail_reference_invalid", "Select a current message."
            )
        if not selection_valid(root.selection, time.time()):
            raise ServiceError(
                409, "mail_reference_expired", "The mail selection expired."
            )
