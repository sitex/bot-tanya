"""Execute at most one owner-bound readonly tool with durable sub-operation holds."""

import hashlib
import time
from collections.abc import Callable
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
from tanya_bot.dialogue_store import DialogueStore
from tanya_bot.errors import ServiceError
from tanya_bot.gmail_account import GmailAccount


class MailTools:
    """The guard checks current core authority before every Google operation."""

    def __init__(
        self,
        store: DialogueStore,
        account: GmailAccount,
        guard: Callable[[DialogueRoot], None],
    ) -> None:
        """Keep mailbox execution in the bot, outside its primary state lock."""
        self.store: DialogueStore = store
        self.account: GmailAccount = account
        self.guard: Callable[[DialogueRoot], None] = guard

    def _reserve(self, root_id: UUID, key: str) -> DialogueRoot:
        def reserve(root: DialogueRoot) -> DialogueRoot:
            self.guard(root)
            if any(item.key == key for item in root.operations):
                raise ServiceError(
                    409, "mail_operation_held", "Mail operation is held."
                )
            return root.model_copy(
                update={"operations": (*root.operations, MailOperation(key=key))}
            )

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
        self.guard(root)
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
                    selected = root.selection.result.sources if root.selection else ()
                    mapping = {item.ref: item.message_id for item in selected}
                    if len(set(refs)) != len(refs) or any(
                        ref not in mapping for ref in refs
                    ):
                        raise ServiceError(
                            422, "mail_reference_invalid", "Select a current message."
                        )
                    ids = tuple(mapping[ref] for ref in refs)
                    preview_limit = 4000
                case _:
                    assert_never(call.action)
            for index, message_id in enumerate(ids):
                current = self._reserve(root.turn.chat.request_id, f"read:{index}")
                self.guard(current)
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
                self.guard(current)
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
