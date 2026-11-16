"""Expire transient mail sources while retaining permanent replay tombstones."""

from typing import Final
from uuid import UUID

from tanya_bot.dialogue_models import DialogueRoot, MailSelection
from tanya_bot.dialogue_store import SOURCE_RETENTION, DialogueStore

IDLE_SECONDS: Final = 1800
SELECTION_SECONDS: Final = 7200


def recent_selection(
    store: DialogueStore, owner: int, conversation: int, now: float
) -> MailSelection | None:
    """References expire after 30 idle minutes or two hours from selection."""
    roots = store.roots()
    candidates = [
        root.selection
        for root in roots
        if root.final is not None
        and not root.cancelled
        and root.selection is not None
        and root.selection.owner_id == owner
        and root.selection.conversation_id == conversation
        and selection_valid(root.selection, now)
    ]
    return max(candidates, key=lambda item: item.accessed_at, default=None)


def selection_valid(selection: MailSelection, now: float) -> bool:
    """Apply the same deadline both at admission and before selected reads."""
    return (
        0 <= now - selection.accessed_at <= IDLE_SECONDS
        and 0 <= now - selection.created_at <= SELECTION_SECONDS
    )


def prune_sources(store: DialogueStore, now: float) -> None:
    """Delete source payloads after 24 hours, never identities or final answers."""
    for root in store.roots():
        if now - root.created_at > SOURCE_RETENTION and (
            root.selection
            or root.tool_result
            or root.operations
            or any(phase.payload or phase.receipt for phase in root.phases)
        ):
            _ = store.mutate(root.turn.chat.request_id, _pruned)


def _pruned(root: DialogueRoot) -> DialogueRoot:
    return root.model_copy(
        update={
            "selection": None,
            "tool_result": None,
            "operations": (),
            "cancelled": root.cancelled or root.final is None,
            "phases": tuple(
                phase.model_copy(
                    update={
                        "payload": None,
                        "receipt": None,
                        "failure": "source_expired",
                    }
                )
                for phase in root.phases
            ),
        }
    )


def cancel_roots(store: DialogueStore, ids: tuple[UUID, ...]) -> None:
    """Cancellation keeps every source and phase hold until normal expiry."""
    for root_id in ids:
        if store.load(root_id) is not None:
            _ = store.mutate(
                root_id, lambda root: root.model_copy(update={"cancelled": True})
            )
