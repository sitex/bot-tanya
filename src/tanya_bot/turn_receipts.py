"""Separate v2 fingerprints preserve every legacy request receipt unchanged."""

import hashlib
import sqlite3
from typing import Final

from pydantic import TypeAdapter

from tanya_bot.errors import ServiceError
from tanya_bot.turns import TurnRequest, TurnResponse

ROWS: Final = TypeAdapter(tuple[tuple[str, str | None], ...])


def fingerprint(request: TurnRequest) -> str:
    """Include the complete frozen derived envelope in the v2 ledger identity."""
    return hashlib.sha256(
        request.model_dump_json(exclude_none=True).encode()
    ).hexdigest()


def lookup(connection: sqlite3.Connection, request: TurnRequest) -> TurnResponse | None:
    """A held UUID fails closed; this operation cannot generate or retry."""
    rows = ROWS.validate_python(
        connection.execute(
            "SELECT digest, response FROM receipts WHERE request_id=?",
            (str(request.chat.request_id),),
        ).fetchall()
    )
    if not rows:
        return None
    digest, data = rows[0]
    if digest != fingerprint(request):
        raise ServiceError(409, "request_conflict", "The frozen turn changed.")
    if data is None:
        raise ServiceError(409, "request_outcome_unknown", "The turn remains held.")
    return TurnResponse.model_validate_json(data)


def reserve(connection: sqlite3.Connection, request: TurnRequest) -> None:
    """Commit the hold before the single billable provider call."""
    _ = connection.execute(
        "INSERT INTO receipts (request_id,digest) VALUES (?,?)",
        (str(request.chat.request_id), fingerprint(request)),
    )
    connection.commit()


def complete(connection: sqlite3.Connection, response: TurnResponse) -> None:
    """Reply, summary and delivery become recoverable in one atomic commit."""
    _ = connection.execute(
        "UPDATE receipts SET response=? WHERE request_id=?",
        (response.model_dump_json(), str(response.request_id)),
    )
    connection.commit()
