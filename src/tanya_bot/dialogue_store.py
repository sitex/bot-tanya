"""Short v3 transactions preserve holds without locking across external I/O."""

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Final
from uuid import UUID

from pydantic import TypeAdapter

from tanya_bot.conversation_memory import JSONValue
from tanya_bot.dialogue_models import DialoguePhase, DialogueRoot, PhaseKind
from tanya_bot.errors import ServiceError
from tanya_bot.receipts import exclusive_store

ROWS: Final = TypeAdapter(tuple[tuple[str], ...])
MAX_ROOTS: Final = 1024
MAX_RECORD_BYTES: Final = 1048576
SOURCE_RETENTION: Final = 86400


class DialogueStore:
    """Only the owner bot creates roots; the API claims authorized phases."""

    def __init__(self, directory: Path) -> None:
        """Use the existing safe, continuity-checked SQLite namespace boundary."""
        self.directory: Path = directory

    def load(self, root_id: UUID) -> DialogueRoot | None:
        """Read a root without advancing or recreating any missing operation."""
        with exclusive_store(self.directory) as connection:
            rows = ROWS.validate_python(
                connection.execute(
                    "SELECT response FROM receipts WHERE request_id=?",
                    (str(root_id),),
                ).fetchall()
            )
        return DialogueRoot.model_validate_json(rows[0][0]) if rows else None

    def create(self, root: DialogueRoot) -> None:
        """A duplicate root remains a hold even if no phase has been sent."""
        data = self._serialize(root)
        with exclusive_store(self.directory) as connection:
            existing = ROWS.validate_python(
                connection.execute(
                    "SELECT request_id FROM receipts WHERE request_id=?",
                    (str(root.turn.chat.request_id),),
                ).fetchall()
            )
            counts = TypeAdapter(tuple[tuple[int], ...]).validate_python(
                connection.execute("SELECT COUNT(*) FROM receipts").fetchall()
            )
            if existing or counts[0][0] >= MAX_ROOTS:
                raise ServiceError(409, "dialogue_held", "Dialogue remains held.")
            digest = hashlib.sha256(root.turn.model_dump_json().encode()).hexdigest()
            _ = connection.execute(
                "INSERT INTO receipts(request_id,digest,response) VALUES(?,?,?)",
                (str(root.turn.chat.request_id), digest, data),
            )
            connection.commit()

    def mutate(
        self, root_id: UUID, change: Callable[[DialogueRoot], DialogueRoot]
    ) -> DialogueRoot:
        """Read, validate and replace one root inside a single short transaction."""
        with exclusive_store(self.directory) as connection:
            rows = ROWS.validate_python(
                connection.execute(
                    "SELECT response FROM receipts WHERE request_id=?",
                    (str(root_id),),
                ).fetchall()
            )
            if not rows:
                raise ServiceError(409, "dialogue_missing", "Dialogue is unavailable.")
            original = DialogueRoot.model_validate_json(rows[0][0])
            updated = change(original)
            if (
                updated.turn != original.turn
                or updated.owner_id != original.owner_id
                or updated.conversation_id != original.conversation_id
                or updated.update_id != original.update_id
                or updated.model != original.model
                or updated.created_at != original.created_at
            ):
                raise ServiceError(409, "dialogue_conflict", "Frozen turn changed.")
            _ = connection.execute(
                "UPDATE receipts SET response=? WHERE request_id=?",
                (self._serialize(updated), str(root_id)),
            )
            connection.commit()
            return updated

    def claim(
        self, root_id: UUID, kind: PhaseKind, payload: dict[str, JSONValue]
    ) -> DialogueRoot:
        """Commit the exact provider payload before its sole network dispatch."""

        def claim(root: DialogueRoot) -> DialogueRoot:
            if root.cancelled:
                raise ServiceError(409, "dialogue_cancelled", "Dialogue is cancelled.")
            phase = phase_for(root, kind)
            if phase.sent or phase.receipt is not None or phase.failure is not None:
                raise ServiceError(409, "dialogue_held", "Phase remains held.")
            claimed = phase.model_copy(update={"sent": True, "payload": payload})
            return replace_phase(root, claimed)

        return self.mutate(root_id, claim)

    def roots(self) -> tuple[DialogueRoot, ...]:
        """Inspect bounded local records for selection expiry and source pruning."""
        with exclusive_store(self.directory) as connection:
            rows = ROWS.validate_python(
                connection.execute("SELECT response FROM receipts").fetchall()
            )
        return tuple(DialogueRoot.model_validate_json(row[0]) for row in rows)

    @staticmethod
    def _serialize(root: DialogueRoot) -> str:
        data = DialogueRoot.model_validate_json(
            root.model_dump_json()
        ).model_dump_json()
        if len(data.encode()) > MAX_RECORD_BYTES:
            raise ServiceError(409, "dialogue_capacity", "Dialogue storage is full.")
        return data


def phase_for(root: DialogueRoot, kind: PhaseKind) -> DialoguePhase:
    """Missing authorization never creates a replacement phase."""
    phase = next((item for item in root.phases if item.kind == kind), None)
    if phase is None:
        raise ServiceError(409, "phase_missing", "Phase is not authorized.")
    return phase


def replace_phase(root: DialogueRoot, phase: DialoguePhase) -> DialogueRoot:
    """Replace the matching phase without allocating another identity."""
    _ = phase_for(root, phase.kind)
    return root.model_copy(
        update={
            "phases": tuple(
                phase if item.kind == phase.kind else item for item in root.phases
            )
        }
    )
