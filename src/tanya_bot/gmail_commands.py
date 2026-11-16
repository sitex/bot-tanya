"""Explicit owner-gated Gmail commands, independent of Telegram session state."""

from __future__ import annotations

import os
import stat
import unicodedata
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, assert_never, final

import anyio
from pydantic import ValidationError
from tanya_gmail import (
    ConnectionStatus,
    ErrorCode,
    FileTokenStore,
    GmailClient,
    GmailConfig,
    GmailError,
)

from .i18n import localize

if TYPE_CHECKING:
    import httpx2

_ACTIONS: Final = (
    "gmail_status",
    "gmail_connect",
    "gmail_list",
    "gmail_disconnect",
)
_SETUP: Final = "Gmail is not configured. Ask the host owner to set up Gmail."
_AUTHORIZE: Final = (
    "Gmail authorization is required. The host owner must explicitly authorize "
    "Google in the host browser, then use /gmail connect."
)
_HELP: Final = (
    "Gmail commands: /gmail status, /gmail connect, /gmail list [query], "
    "/gmail read ID, /gmail disconnect."
)
_MAX_CONFIG: Final = 16384
_PRIVATE_MODE: Final = 0o600
_ARGUMENT_PARTS: Final = 3
type GmailAction = Literal["connect", "disconnect", "list", "read"]
_COMMANDS: Final[tuple[GmailAction, ...]] = ("connect", "disconnect", "list", "read")


@dataclass(frozen=True, slots=True)
class GmailOutput:
    """Plain text for a renderer with formatting disabled, plus local actions."""

    text: str
    actions: tuple[str, ...] = ()


def _config(path: Path) -> GmailConfig:
    """Read bounded private JSON through descriptors without following symlinks."""
    if not path.is_absolute() or ".." in path.parts or path == Path("/"):
        raise GmailError(ErrorCode.UNSAFE_STORAGE)
    try:
        with ExitStack() as stack:
            directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
            _ = stack.callback(os.close, directory)
            for part in path.parts[1:-1]:
                directory = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory,
                )
                _ = stack.callback(os.close, directory)
            descriptor = os.open(
                path.name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory,
            )
            stream = stack.enter_context(os.fdopen(descriptor, "rb"))
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != _PRIVATE_MODE
                or info.st_nlink != 1
                or info.st_size > _MAX_CONFIG
            ):
                raise GmailError(ErrorCode.UNSAFE_STORAGE)
            data = stream.read(_MAX_CONFIG + 1)
            if len(data) > _MAX_CONFIG:
                raise GmailError(ErrorCode.UNSAFE_STORAGE)
            return GmailConfig.model_validate_json(data)
    except FileNotFoundError:
        raise
    except OSError:
        raise GmailError(ErrorCode.UNSAFE_STORAGE) from None
    except ValidationError:
        raise GmailError(ErrorCode.INVALID_INPUT) from None


def _plain(text: str, limit: int) -> str:
    """Remove control/format characters and bound display text by UTF-16 units."""
    safe = "".join(
        char
        for char in text
        if char in "\n\t" or not unicodedata.category(char).startswith("C")
    )
    return safe.encode("utf-16-le")[: limit * 2].decode("utf-16-le", errors="ignore")


@final
class GmailCommands:
    """Keep one instance per account; its lock serializes all worker operations.

    Construction performs no configuration, token-store or Google I/O. The caller
    must enforce owner authorization and callback claims before calling handle.
    """

    def __init__(
        self,
        config_path: Path,
        token_directory: Path,
        *,
        transport: httpx2.BaseTransport | None = None,
    ) -> None:
        """Store paths and a transport without opening configuration or credentials."""
        self._config_path = config_path
        self._token_directory = token_directory
        self._transport = transport
        self._client: GmailClient | None = None
        self._lock = anyio.Lock()

    @staticmethod
    def is_command(text: str) -> bool:
        """Recognize only the standalone /gmail command prefix."""
        return text.split(maxsplit=1)[:1] == ["/gmail"]

    async def handle(self, text: str, language: str) -> GmailOutput:
        """Execute one explicit gesture off the event loop under the account lock."""
        async with self._lock:
            return await anyio.to_thread.run_sync(self._handle, text, language)

    def _handle(self, text: str, language: str) -> GmailOutput:
        try:
            return self._execute(text, language)
        except FileNotFoundError:
            return GmailOutput(localize(_SETUP, language), _ACTIONS)
        except GmailError as error:
            if error.code is ErrorCode.AUTH_REQUIRED:
                return GmailOutput(localize(_AUTHORIZE, language), _ACTIONS)
            return GmailOutput(
                f"{localize('Gmail request failed:', language)} {error.code.value}",
                _ACTIONS,
            )

    def _execute(self, text: str, language: str) -> GmailOutput:
        parts = text.split(maxsplit=2)
        if not self.is_command(text):
            return GmailOutput(localize(_HELP, language), _ACTIONS)
        command = parts[1] if len(parts) > 1 else "status"
        argument = parts[2] if len(parts) == _ARGUMENT_PARTS else ""
        if (
            command not in {"status", "connect", "list", "read", "disconnect"}
            or (command not in {"list", "read"} and argument)
            or (command == "read" and not argument)
        ):
            return GmailOutput(localize(_HELP, language), _ACTIONS)
        if command == "status":
            return self._status(language)
        if self._client is None:
            self._client = GmailClient(
                _config(self._config_path),
                FileTokenStore(self._token_directory),
                transport=self._transport,
            )
        client = self._client
        action: GmailAction | None = next(
            (item for item in _COMMANDS if item == command),
            None,
        )
        if action is None:
            raise GmailError(ErrorCode.INVALID_INPUT)
        match action:
            case "connect":
                _ = client.connect(interactive=False)
                message = "Gmail is connected."
            case "disconnect":
                client.disconnect(revoke=False)
                message = "Gmail local credentials were removed."
            case "list":
                return self._list(client, argument, language)
            case "read":
                preview = client.read_message(argument, preview_chars=2400)
                lines = [
                    localize("Gmail plain-text preview (untrusted content):", language)
                ]
                lines.extend(
                    _plain(value, 160)
                    for value in (
                        preview.subject,
                        preview.sender,
                        preview.date,
                    )
                )
                lines.append(_plain(preview.text, 2400))
                lines.append(localize("Preview may be shortened.", language))
                return GmailOutput("\n".join(lines), _ACTIONS)
            case _:
                assert_never(action)
        return GmailOutput(localize(message, language), _ACTIONS)

    def _status(self, language: str) -> GmailOutput:
        if self._client is None:
            _ = _config(self._config_path)
        connected = (
            self._client is not None
            and self._client.status() is ConnectionStatus.CONNECTED
        )
        message = "Gmail is connected." if connected else "Gmail is disconnected."
        return GmailOutput(localize(message, language), _ACTIONS)

    @staticmethod
    def _list(client: GmailClient, query: str, language: str) -> GmailOutput:
        page = client.list_messages(query=query, page_size=10)
        lines = [localize("Gmail message IDs (one page):", language)]
        lines.extend(f"/gmail read {item.id}" for item in page.messages)
        if not page.messages:
            lines.append(localize("No messages found.", language))
        if page.next_page_token:
            lines.append(localize("More results exist. Refine your query.", language))
        return GmailOutput("\n".join(lines), _ACTIONS)
