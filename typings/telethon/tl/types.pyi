from datetime import datetime

class KeyboardInlineButton: ...

class PeerUser:
    def __init__(self, user_id: int) -> None: ...

class User:
    id: int
    bot: bool | None
    username: str | None

class Document:
    def __init__(
        self,
        document_id: int,
        access_hash: int,
        file_reference: bytes,
        date: datetime | None,
        mime_type: str,
        size: int,
        dc_id: int,
        attributes: list[DocumentAttributeAudio],
    ) -> None: ...
    size: int

class DocumentAttributeAudio:
    def __init__(self, duration: int, *, voice: bool = ...) -> None: ...
    duration: float

class MessageMediaDocument:
    def __init__(self, *, document: Document, voice: bool = ...) -> None: ...

class BotCommand:
    def __init__(self, command: str, description: str) -> None: ...

class BotCommandScopeDefault: ...
class BotMenuButtonCommands: ...
class InputUserEmpty: ...
