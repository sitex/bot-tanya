"""Private SQLite receipts and an interprocess concurrency gate."""

import fcntl
import hashlib
import os
import sqlite3
import stat
from collections.abc import Generator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Final

from pydantic import TypeAdapter

from tanya_bot.config import SECRET_MODE
from tanya_bot.errors import ServiceError
from tanya_bot.schemas import ChatRequest, ChatResponse, FrozenModel


class Receipt(FrozenModel):
    """Only a digest and successful output cross the persistence boundary."""

    digest: str
    response: ChatResponse | None


type ReceiptRow = tuple[str, str | None]
ROWS_ADAPTER: Final = TypeAdapter(tuple[ReceiptRow, ...])
DIRECTORY_MODE: Final = 0o700


def fingerprint(request: ChatRequest) -> str:
    """Hash the parsed input, including explicit defaults, without persisting it."""
    return hashlib.sha256(
        request.model_dump_json(exclude_none=True).encode()
    ).hexdigest()


def private_file(path: Path, *, flags: int = os.O_CREAT) -> int:
    """Create 0600 state files and refuse unsafe existing files."""
    descriptor = os.open(
        path, os.O_RDWR | flags | os.O_NOFOLLOW | os.O_NONBLOCK, SECRET_MODE
    )
    metadata = os.fstat(descriptor)
    if (
        metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != SECRET_MODE
        or not stat.S_ISREG(metadata.st_mode)
    ):
        os.close(descriptor)
        raise ServiceError(
            503, "storage_unavailable", "Receipt requires owned 0600 file."
        )
    return descriptor


def _sync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fresh_directory(directory: Path) -> bool:
    return all(
        p.name == "turns-v2" and stat.S_ISDIR(p.lstat().st_mode)
        for p in directory.iterdir()
        if p.name != "request.lock"
    )


def _prepare(connection: sqlite3.Connection, *, fresh: bool) -> None:
    """Existing receipt authority must be readable before any schema creation."""
    try:
        if not fresh:
            _ = connection.execute(
                "SELECT request_id, digest, response FROM receipts LIMIT 0"
            )
        _ = connection.execute("PRAGMA journal_mode=DELETE")
        _ = connection.execute("PRAGMA synchronous=FULL")
        if fresh:
            _ = connection.execute(
                """CREATE TABLE receipts (
                    request_id TEXT PRIMARY KEY, digest TEXT NOT NULL, response TEXT)"""
            )
            connection.commit()
    except sqlite3.Error:
        raise ServiceError(
            503, "storage_unavailable", "Private request receipts are unavailable."
        ) from None


@contextmanager
def exclusive_store(directory: Path) -> Generator[sqlite3.Connection]:
    """Hold a nonblocking cross-process lock through commit and the HTTP call.

    The sent receipt is committed before network I/O. A process crash therefore
    leaves a durable hold rather than making the next request spend again.
    """
    directory.mkdir(mode=DIRECTORY_MODE, parents=True, exist_ok=True)
    metadata = directory.lstat()
    if (
        metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != DIRECTORY_MODE
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise ServiceError(
            503, "storage_unavailable", "State requires owned 0700 folder."
        )
    # V2 may create the shared API parent before its independent V1 ledger exists.
    fresh = _fresh_directory(directory)
    try:
        descriptor = private_file(
            directory / "request.lock", flags=os.O_CREAT | os.O_EXCL
        )
    except FileExistsError:
        descriptor = private_file(directory / "request.lock", flags=0)
        fresh = False
    with os.fdopen(descriptor, "rb") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ServiceError(
                503, "server_busy", "Another chat request is in progress."
            ) from None
        os.fsync(lock.fileno())
        _sync_directory(directory)
        fresh = fresh and _fresh_directory(directory)
        database = directory / "receipts.sqlite3"
        try:
            descriptor = private_file(
                database, flags=os.O_CREAT | os.O_EXCL if fresh else 0
            )
        except FileNotFoundError:
            raise ServiceError(
                503, "storage_unavailable", "Private request receipts are unavailable."
            ) from None
        with (
            os.fdopen(descriptor, "rb"),
            closing(
                sqlite3.connect(
                    f"{database.absolute().as_uri()}?mode=rw", uri=True, timeout=0.1
                )
            ) as connection,
        ):
            _prepare(connection, fresh=fresh)
            _sync_directory(directory)
            yield connection


def lookup(connection: sqlite3.Connection, request: ChatRequest) -> ChatResponse | None:
    """Replay successes, rejecting changed input and every unresolved send."""
    rows = ROWS_ADAPTER.validate_python(
        connection.execute(
            "SELECT digest, response FROM receipts WHERE request_id = ?",
            (str(request.request_id),),
        ).fetchall()
    )
    if not rows:
        return None
    digest, serialized = rows[0]
    receipt = Receipt(
        digest=digest,
        response=ChatResponse.model_validate_json(serialized) if serialized else None,
    )
    if receipt.digest != fingerprint(request):
        raise ServiceError(
            409, "request_conflict", "This request_id belongs to different input."
        )
    if receipt.response is None:
        raise ServiceError(
            409,
            "request_outcome_unknown",
            "This request was marked sent. Check its outcome before sending again.",
        )
    return receipt.response


def mark_sent(connection: sqlite3.Connection, request: ChatRequest) -> None:
    """Durably claim an ID before a potentially billable request."""
    _ = connection.execute(
        "INSERT INTO receipts (request_id, digest) VALUES (?, ?)",
        (str(request.request_id), fingerprint(request)),
    )
    connection.commit()


def save_success(connection: sqlite3.Connection, response: ChatResponse) -> None:
    """Commit the reply for replay without storing the input dialogue."""
    _ = connection.execute(
        "UPDATE receipts SET response = ? WHERE request_id = ?",
        (response.model_dump_json(), str(response.request_id)),
    )
    connection.commit()
