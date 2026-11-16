"""Private atomic state with a nonblocking interprocess transaction boundary."""

import fcntl
import json
import os
import stat
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final, Protocol, assert_never, override

from pydantic import JsonValue, TypeAdapter, ValidationError

from tanya_bot.telegram_models import TelegramState

DIRECTORY_MODE: Final = 0o700
FILE_MODE: Final = 0o600
STATE_BYTES: Final = 32_000_000
JSON_STATE: Final = TypeAdapter(JsonValue)


class CoreError(Exception):
    """Safe machine-readable failure, without submitted content."""

    code: str

    def __init__(self, code: str) -> None:
        """Permit context managers to attach standard exception traceback metadata."""
        self.code = code
        super().__init__(code)

    @override
    def __str__(self) -> str:
        """Return only the safe code to callers and runtime logs."""
        return self.code


class StateStore(Protocol):
    """Transaction held by controller through reservation, call and completion."""

    def exclusive(self) -> AbstractContextManager[None]:
        """Implement as a context manager; concurrent users must fail closed."""
        ...

    def load(self) -> TelegramState:
        """Read validated state or return a fresh owner-bound state."""
        ...

    def save(self, state: TelegramState) -> None:
        """Durably commit before returning; never publish partial state."""
        ...


class TelegramStore:
    """Only the designated bot directory is ever created or changed."""

    def __init__(self, directory: Path, owner_id: int) -> None:
        """Bind a new bot-specific state directory to its sole owner."""
        self.directory: Path = directory
        self.owner_id: int = owner_id

    def _directory(self) -> None:
        self.directory.mkdir(mode=DIRECTORY_MODE, parents=True, exist_ok=True)
        info = self.directory.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or stat.S_IMODE(info.st_mode) != DIRECTORY_MODE
            or info.st_uid != os.getuid()
        ):
            raise CoreError(code="unsafe_directory")

    @staticmethod
    def _open(path: Path, *, create: bool = False) -> int:
        flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
        if create:
            flags |= os.O_CREAT
        fd = os.open(path, flags, FILE_MODE)
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != FILE_MODE
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
        ):
            os.close(fd)
            raise CoreError(code="unsafe_file")
        return fd

    @contextmanager
    def exclusive(self) -> Generator[None]:
        """Keep the file lock across awaits; it never waits on another process."""
        self._directory()
        with os.fdopen(
            self._open(self.directory / "state.lock", create=True), "rb"
        ) as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise CoreError(code="store_busy") from None
            yield

    def load(self) -> TelegramState:
        """Missing state is fresh; invalid or unsafe state never resets silently."""
        self._directory()
        try:
            fd = self._open(self.directory / "state.json")
        except FileNotFoundError:
            return TelegramState(owner_id=self.owner_id)
        with os.fdopen(fd, "rb") as source:
            data = source.read(STATE_BYTES + 1)
        if len(data) > STATE_BYTES:
            raise CoreError(code="state_capacity")
        try:
            parsed = JSON_STATE.validate_json(data)
            match parsed:
                case dict():
                    old_settings = parsed.get("settings", {})
                    match old_settings:
                        case dict():
                            if "language" not in old_settings:
                                parsed = {
                                    **parsed,
                                    "settings": {**old_settings, "language": "ru"},
                                }
                        case list() | str() | int() | float() | None:
                            raise CoreError(code="state_corrupt")
                        case _:
                            assert_never(old_settings)
                case list() | str() | int() | float() | None:
                    raise CoreError(code="state_corrupt")
                case _:
                    assert_never(parsed)
            state = TelegramState.model_validate_json(json.dumps(parsed))
        except ValidationError:
            raise CoreError(code="state_corrupt") from None
        if state.owner_id != self.owner_id or len(state.history) != len(
            state.history_roles
        ):
            raise CoreError(code="state_owner_or_history")
        return state

    def save(self, state: TelegramState) -> None:
        """Validate, fsync, replace, then fsync directory before any publication."""
        self._directory()
        checked = TelegramState.model_validate_json(state.model_dump_json())
        if checked.owner_id != self.owner_id:
            raise CoreError(code="state_owner")
        serialized = checked.model_dump_json().encode()
        if len(serialized) > STATE_BYTES:
            raise CoreError(code="state_capacity")
        path: Path | None = None
        try:
            with NamedTemporaryFile(
                dir=self.directory, prefix=".state-", delete=False
            ) as out:
                path = Path(out.name)
                os.fchmod(out.fileno(), FILE_MODE)
                _ = out.write(serialized)
                out.flush()
                os.fsync(out.fileno())
            _ = path.replace(self.directory / "state.json")
            fd = os.open(self.directory, os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            if path is not None:
                path.unlink(missing_ok=True)
