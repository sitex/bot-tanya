"""Bounded audio receipts and safe failures shared by provider and artifact store."""

from subprocess import DEVNULL
from typing import Annotated, ClassVar, Final, Literal, Self
from uuid import UUID

import anyio
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from tanya_bot.schemas import FrozenModel
from tanya_bot.telegram_models import VOICE_TEXT_CHARACTERS, GatewayFailure

MAX_AUDIO_BYTES: Final = 5 * 1024 * 1024
MAX_STATE_BYTES: Final = 8 * 1024 * 1024
MAX_DIRECTORY_BYTES: Final = 64 * 1024 * 1024
MAX_REQUESTS: Final = 128
MAX_AUDIO_SECONDS: Final = 120
MAX_TTS_CHARACTERS: Final = VOICE_TEXT_CHARACTERS
MAX_TRANSCRIPT_CHARACTERS: Final = 16_000
PROVIDER_SECONDS: Final = 90
MEDIA_SECONDS: Final = 15
OGG_HEADER_BYTES: Final = 27
OPUS_HEADER_BYTES: Final = 19
TTS_MODEL: Final = "gpt-4o-mini-tts"
STT_MODEL: Final = "gpt-transcribe"
VOICES: Final = frozenset(
    {
        "alloy",
        "ash",
        "ballad",
        "coral",
        "echo",
        "sage",
        "shimmer",
        "verse",
        "marin",
        "cedar",
    }
)
type Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", strict=True)]
type Transcript = Annotated[
    str, Field(min_length=1, max_length=MAX_TRANSCRIPT_CHARACTERS, strict=True)
]
type AudioKind = Literal["speech", "transcription"]


class AudioFailure(GatewayFailure):
    """Fixed safe code only; never carries provider body or submitted content."""


class TranscriptionResult(FrozenModel):
    """Exact cached transcript bound to root's stable incoming-update UUID."""

    request_id: UUID
    text: Transcript


class ProviderTranscript(BaseModel):
    """Parse only text; documented language metadata is intentionally unused."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")
    text: Transcript


class AudioReceipt(FrozenModel):
    """A missing outcome is a permanent held reservation, never permission to retry."""

    request_id: UUID
    kind: AudioKind
    digest: Digest
    artifact_id: UUID
    audio_digest: Digest | None = None
    audio_size: Annotated[int, Field(ge=1, le=MAX_AUDIO_BYTES, strict=True)] | None = (
        None
    )
    text: Transcript | None = None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        """Reject partial outcomes and outcomes attached to the wrong operation."""
        speech = self.kind == "speech"
        if (
            (self.audio_digest is None) != (self.audio_size is None)
            or (speech and self.text is not None)
            or (not speech and self.audio_digest is not None)
        ):
            message = "Invalid audio receipt"
            code = "audio_receipt"
            raise PydanticCustomError(code, message)
        return self


class AudioState(FrozenModel):
    """Append-only identities; no reset, expiry, automatic eviction or retries."""

    version: Literal[1] = 1
    receipts: Annotated[tuple[AudioReceipt, ...], Field(max_length=MAX_REQUESTS)] = ()

    @model_validator(mode="after")
    def unique(self) -> Self:
        """Reject ambiguous request identities and forged artifact collisions."""
        if len({r.request_id for r in self.receipts}) != len(self.receipts) or len(
            {r.artifact_id for r in self.receipts}
        ) != len(self.receipts):
            message = "Invalid audio state"
            code = "audio_state"
            raise PydanticCustomError(code, message)
        return self


async def telegram_opus(audio: bytes) -> bytes:
    """Decode bounded provider bytes into mono Ogg Opus using offline ffmpeg."""
    try:
        with anyio.fail_after(MEDIA_SECONDS):
            process = await anyio.run_process(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "quiet",
                    "-nostdin",
                    "-protocol_whitelist",
                    "pipe",
                    "-i",
                    "pipe:0",
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "48000",
                    "-c:a",
                    "libopus",
                    "-b:a",
                    "48k",
                    "-t",
                    "121",
                    "-fs",
                    str(MAX_AUDIO_BYTES),
                    "-f",
                    "ogg",
                    "pipe:1",
                ],
                input=audio,
                stderr=DEVNULL,
                check=False,
            )
    except (OSError, TimeoutError):
        raise AudioFailure(code="audio_conversion") from None
    data = process.stdout
    if process.returncode or not data or len(data) > MAX_AUDIO_BYTES:
        raise AudioFailure(code="audio_conversion")
    check_opus(data)
    return data


def check_opus(data: bytes) -> None:
    """Require one complete mono Ogg Opus stream within Telegram's time bound."""
    offset = 0
    granule = 0
    pre_skip = 0
    serial = data[14:18]
    ended = False
    while offset < len(data):
        header = data[offset : offset + 27]
        if len(header) != OGG_HEADER_BYTES or header[:5] != b"OggS\x00" or ended:
            raise AudioFailure(code="audio_invalid_media")
        segments = header[26]
        lacing = data[offset + 27 : offset + 27 + segments]
        end = offset + 27 + segments + sum(lacing)
        payload = data[offset + 27 + segments : end]
        if len(lacing) != segments or end > len(data) or header[14:18] != serial:
            raise AudioFailure(code="audio_invalid_media")
        if offset == 0:
            if len(payload) < OPUS_HEADER_BYTES or payload[:10] != b"OpusHead\x01\x01":
                raise AudioFailure(code="audio_invalid_media")
            pre_skip = int.from_bytes(payload[10:12], "little")
        granule = int.from_bytes(header[6:14], "little")
        ended = bool(header[5] & 4)
        offset = end
    if not ended or not pre_skip < granule <= pre_skip + MAX_AUDIO_SECONDS * 48000:
        raise AudioFailure(code="audio_invalid_media")
