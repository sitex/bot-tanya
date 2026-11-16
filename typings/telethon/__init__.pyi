from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import overload

from . import events
from .tl.custom.message import Message
from .tl.functions.bots import (
    SetBotCommandsRequest,
    SetBotInfoRequest,
    SetBotMenuButtonRequest,
)
from .tl.functions.messages import SetBotCallbackAnswerRequest
from .tl.types import KeyboardInlineButton, User

class Button:
    @staticmethod
    def inline(text: str, data: bytes) -> KeyboardInlineButton: ...

class TelegramClient:
    def __init__(
        self,
        session: str,
        api_id: int,
        api_hash: str,
        *,
        request_retries: int = ...,
        connection_retries: int = ...,
        flood_sleep_threshold: int = ...,
        raise_last_call_error: bool = ...,
        sequential_updates: bool = ...,
        catch_up: bool = ...,
        receive_updates: bool = ...,
    ) -> None: ...
    @overload
    def add_event_handler(
        self,
        callback: Callable[[events.NewMessage.Event], Awaitable[None]],
        event: events.NewMessage,
    ) -> None: ...
    @overload
    def add_event_handler(
        self,
        callback: Callable[[events.CallbackQuery.Event], Awaitable[None]],
        event: events.CallbackQuery,
    ) -> None: ...
    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def is_user_authorized(self) -> bool: ...
    async def sign_in(self, *, bot_token: str) -> User: ...
    async def get_me(self) -> User | None: ...
    async def send_message(
        self,
        entity: int,
        message: str,
        *,
        buttons: list[list[KeyboardInlineButton]] | None = ...,
        parse_mode: None = ...,
        link_preview: bool = ...,
    ) -> Message: ...
    async def edit_message(
        self,
        entity: int,
        message: int,
        *,
        buttons: list[list[KeyboardInlineButton]] | None = ...,
    ) -> Message: ...
    async def send_file(
        self, entity: int, file: Path, *, voice_note: bool = ..., parse_mode: None = ...
    ) -> Message: ...
    async def download_media(
        self, message: Message, file: type[bytes]
    ) -> bytes | None: ...
    async def __call__(
        self,
        request: SetBotCallbackAnswerRequest
        | SetBotInfoRequest
        | SetBotCommandsRequest
        | SetBotMenuButtonRequest,
    ) -> bool: ...
    @property
    def disconnected(self) -> Awaitable[None]: ...
