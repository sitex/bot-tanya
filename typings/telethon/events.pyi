from .tl.custom.message import Message

class NewMessage:
    def __init__(self, *, incoming: bool = ...) -> None: ...
    class Event:
        def __init__(self, message: Message) -> None: ...
        id: int
        sender_id: int | None
        chat_id: int | None
        is_private: bool
        raw_text: str
        message: Message

class CallbackQuery:
    class Event:
        id: int
        sender_id: int | None
        chat_id: int | None
        is_private: bool
        message_id: int
        data: bytes | None
