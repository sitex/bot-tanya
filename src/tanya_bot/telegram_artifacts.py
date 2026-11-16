"""Owner-only atomic audio ledger with local artifact integrity checks."""

import fcntl
import hashlib
import os
import stat
from collections.abc import Generator
from contextlib import contextmanager, suppress
from itertools import islice
from pathlib import Path
from typing import Final
from uuid import UUID, uuid4

from pydantic import ValidationError

from tanya_bot.telegram_audio_models import (
    MAX_AUDIO_BYTES,
    MAX_DIRECTORY_BYTES,
    MAX_REQUESTS,
    MAX_STATE_BYTES,
    AudioFailure,
    AudioReceipt,
    AudioState,
)
from tanya_bot.telegram_models import VoiceResult

_DIR_FLAGS: Final = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_FILE_FLAGS: Final = os.O_NOFOLLOW | os.O_NONBLOCK
_DIRECTORY_MODE: Final = 0o700
_FILE_MODE: Final = 0o600


def _names(fd: int) -> list[str]:
    with os.scandir(fd) as entries:
        names = [entry.name for entry in islice(entries, MAX_REQUESTS * 2 + 5)]
    if len(names) > MAX_REQUESTS * 2 + 4:
        raise AudioFailure(code="audio_capacity")
    return names


@contextmanager
def _directory(path: Path, *, create: bool = False) -> Generator[int]:
    """Traverse every component using directory descriptors, never a symlink."""
    fd = os.open("/", _DIR_FLAGS)
    try:
        for index, part in enumerate(path.parts[1:], start=1):
            if create and index == len(path.parts) - 1:
                with suppress(FileExistsError):
                    os.mkdir(part, 0o700, dir_fd=fd)
            next_fd = os.open(part, _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        info = os.fstat(fd)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != _DIRECTORY_MODE:
            raise AudioFailure(code="audio_unsafe_directory")
        yield fd
    finally:
        os.close(fd)


def _check_file(info: os.stat_result) -> None:
    """Only owned private single-link regular files may enter the ledger."""
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != _FILE_MODE
        or info.st_nlink != 1
    ):
        raise AudioFailure(code="audio_unsafe_file")


def _read(fd: int, name: str, limit: int) -> bytes:
    with os.fdopen(os.open(name, os.O_RDONLY | _FILE_FLAGS, dir_fd=fd), "rb") as f:
        _check_file(os.fstat(f.fileno()))
        data = f.read(limit + 1)
    if len(data) > limit:
        raise AudioFailure(code="audio_capacity")
    return data


def _atomic(fd: int, name: str, data: bytes) -> None:
    """Persist new bytes then replace and sync the containing directory."""
    temporary = f"{uuid4()}.tmp"
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _FILE_FLAGS
        with os.fdopen(os.open(temporary, flags, 0o600, dir_fd=fd), "wb") as f:
            _ = f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, name, src_dir_fd=fd, dst_dir_fd=fd)
        os.fsync(fd)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary, dir_fd=fd)


class ArtifactTransaction:
    """One exclusive ledger transaction; its in-memory state changes on commit."""

    def __init__(self, fd: int) -> None:
        """Load validated durable state; absence after initialization is corruption."""
        self.fd: int = fd
        try:
            self.state: AudioState = AudioState.model_validate_json(
                _read(fd, "state.json", MAX_STATE_BYTES)
            )
        except (FileNotFoundError, ValidationError):
            raise AudioFailure(code="audio_corrupt") from None

    def find(self, candidate: AudioReceipt) -> AudioReceipt | None:
        """Bind UUID to exact digest and operation, including unresolved receipts."""
        receipt = next(
            (r for r in self.state.receipts if r.request_id == candidate.request_id),
            None,
        )
        if receipt is not None and (
            receipt.kind != candidate.kind or receipt.digest != candidate.digest
        ):
            raise AudioFailure(code="audio_conflict")
        return receipt

    def save(self, receipt: AudioReceipt) -> None:
        """Append a reservation or replace only its matching completion."""
        receipts = (
            *(r for r in self.state.receipts if r.request_id != receipt.request_id),
            receipt,
        )
        state = AudioState(receipts=receipts)
        data = state.model_dump_json().encode()
        if len(data) > MAX_STATE_BYTES:
            raise AudioFailure(code="audio_capacity")
        _atomic(self.fd, "state.json", data)
        self.state = state

    def reserve(self, receipt: AudioReceipt) -> None:
        """Account for pending worst-case audio space before any paid operation."""
        promised = sum(
            r.audio_size or MAX_AUDIO_BYTES
            for r in self.state.receipts
            if r.kind == "speech"
        )
        used = 0
        names = _names(self.fd)
        for name in names:
            info = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
            _check_file(info)
            used += info.st_size
        if (
            len(self.state.receipts) >= MAX_REQUESTS
            or used + MAX_AUDIO_BYTES > MAX_DIRECTORY_BYTES
            or promised + MAX_AUDIO_BYTES > MAX_DIRECTORY_BYTES - MAX_STATE_BYTES
        ):
            raise AudioFailure(code="audio_capacity")
        self.save(receipt)

    def audio(self, receipt: AudioReceipt) -> bytes:
        """Verify content binding before recovery or exposing a sendable path."""
        data = _read(self.fd, f"{receipt.artifact_id}.ogg", MAX_AUDIO_BYTES)
        if (
            len(data) != receipt.audio_size
            or hashlib.sha256(data).hexdigest() != receipt.audio_digest
        ):
            raise AudioFailure(code="audio_corrupt")
        return data

    def complete_audio(self, receipt: AudioReceipt, audio: bytes) -> None:
        """Audio is durable before its completed receipt becomes visible."""
        _atomic(self.fd, f"{receipt.artifact_id}.ogg", audio)
        self.save(
            receipt.model_copy(
                update={
                    "audio_digest": hashlib.sha256(audio).hexdigest(),
                    "audio_size": len(audio),
                }
            )
        )


class ArtifactStore:
    """Fresh bot-specific directory; every public operation fails closed."""

    def __init__(self, directory: Path) -> None:
        """Create only the final directory; initialize under an exclusive lock."""
        if not directory.is_absolute() or ".." in directory.parts:
            raise AudioFailure(code="audio_unsafe_directory")
        self.directory: Path = directory
        try:
            with _directory(directory, create=True) as fd:
                fresh = "state.lock" not in _names(fd)
                flags = os.O_RDWR | _FILE_FLAGS
                if fresh:
                    if _names(fd):
                        raise AudioFailure(code="audio_corrupt")
                    flags |= os.O_CREAT | os.O_EXCL
                with os.fdopen(
                    os.open("state.lock", flags, 0o600, dir_fd=fd), "rb"
                ) as lock:
                    _check_file(os.fstat(lock.fileno()))
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if fresh:
                        _atomic(
                            fd, "state.json", AudioState().model_dump_json().encode()
                        )
                    _ = ArtifactTransaction(fd)
        except OSError:
            raise AudioFailure(code="audio_storage") from None

    @contextmanager
    def exclusive(self) -> Generator[ArtifactTransaction]:
        """Hold a nonblocking process lock across reserve, POST and completion."""
        try:
            with (
                _directory(self.directory) as fd,
                os.fdopen(
                    os.open("state.lock", os.O_RDWR | _FILE_FLAGS, dir_fd=fd), "rb"
                ) as lock,
            ):
                _check_file(os.fstat(lock.fileno()))
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise AudioFailure(code="audio_busy") from None
                yield ArtifactTransaction(fd)
        except OSError:
            raise AudioFailure(code="audio_storage") from None

    def resolve(self, result: VoiceResult) -> Path:
        """Resolve canonical UUID only, bound to this exact completed request."""
        try:
            artifact_id = UUID(result.artifact_id)
        except ValueError:
            raise AudioFailure(code="audio_invalid_artifact") from None
        if str(artifact_id) != result.artifact_id:
            raise AudioFailure(code="audio_invalid_artifact")
        with self.exclusive() as transaction:
            receipt = next(
                (
                    r
                    for r in transaction.state.receipts
                    if r.request_id == result.request_id
                ),
                None,
            )
            if (
                receipt is None
                or receipt.artifact_id != artifact_id
                or receipt.audio_digest is None
            ):
                raise AudioFailure(code="audio_invalid_artifact")
            _ = transaction.audio(receipt)
        return self.directory / f"{artifact_id}.ogg"
