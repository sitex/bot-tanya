from telethon.tl.types import Document, MessageMediaDocument, PeerUser

class Message:
    def __init__(
        self,
        message_id: int,
        peer_id: PeerUser,
        *,
        from_id: PeerUser | None = ...,
        message: str | None = ...,
        media: MessageMediaDocument | None = ...,
    ) -> None: ...
    id: int
    voice: Document | None
    document: Document | None
    audio: Document | None
    @property
    def file(self) -> File | None: ...

class File:
    duration: float | None
    size: int | None
