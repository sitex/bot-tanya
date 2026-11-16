"""At-most-once OpenAI speech and transcription, with local-only recovery."""

import hashlib
import json
import socket
from http import HTTPStatus
from pathlib import Path
from typing import Final
from uuid import UUID, uuid4

import anyio
import httpx2
from pydantic import ValidationError

from tanya_bot import config
from tanya_bot.telegram_artifacts import ArtifactStore
from tanya_bot.telegram_audio_models import (
    MAX_AUDIO_BYTES,
    MAX_TTS_CHARACTERS,
    PROVIDER_SECONDS,
    STT_MODEL,
    TTS_MODEL,
    VOICES,
    AudioFailure,
    AudioReceipt,
    ProviderTranscript,
    TranscriptionResult,
    telegram_opus,
)
from tanya_bot.telegram_models import VoiceRequest, VoiceResult

_SPEECH_URL: Final = "https://api.openai.com/v1/audio/speech"
_TRANSCRIBE_URL: Final = "https://api.openai.com/v1/audio/transcriptions"
_TRANSCRIPT_BYTES: Final = 128 * 1024


def _speech_receipt(request: VoiceRequest) -> AudioReceipt:
    if (
        request.voice not in VOICES
        or not request.text.strip()
        or len(request.text) > MAX_TTS_CHARACTERS
    ):
        raise AudioFailure(code="audio_invalid_request")
    items = [TTS_MODEL, request.text, request.voice, "opus"]
    if request.delivery is not None:
        items.append(request.delivery.model_dump_json())
    data = json.dumps(
        items,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode()
    return AudioReceipt(
        request_id=request.request_id,
        kind="speech",
        digest=hashlib.sha256(data).hexdigest(),
        artifact_id=uuid4(),
    )


def _transcript_receipt(request_id: UUID, audio: bytes) -> AudioReceipt:
    if not 0 < len(audio) <= MAX_AUDIO_BYTES:
        raise AudioFailure(code="audio_invalid_request")
    digest = hashlib.sha256(STT_MODEL.encode() + b"\x00" + audio).hexdigest()
    return AudioReceipt(
        request_id=request_id,
        kind="transcription",
        digest=digest,
        artifact_id=uuid4(),
    )


async def _body(response: httpx2.Response, limit: int) -> bytes:
    if response.status_code != HTTPStatus.OK:
        raise AudioFailure(code="audio_provider_status")
    if response.headers.get("content-encoding", "identity") != "identity":
        raise AudioFailure(code="audio_provider_encoding")
    content = bytearray()
    async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
        if len(content) + len(chunk) > limit:
            raise AudioFailure(code="audio_provider_capacity")
        content.extend(chunk)
    if not content:
        raise AudioFailure(code="audio_provider_empty")
    return bytes(content)


class TelegramVoiceGateway:
    """Bounded adapter; the store lock spans durable reservation and paid call."""

    def __init__(
        self,
        directory: Path,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        """Select a fresh private bot directory; inject a transport for offline QA."""
        self.store: ArtifactStore = ArtifactStore(directory)
        self.transport: httpx2.AsyncBaseTransport | None = transport

    def _client(self) -> httpx2.AsyncClient:
        key = config.api_key()
        if key is None:
            raise AudioFailure(code="audio_key_missing")
        transport = self.transport or httpx2.AsyncHTTPTransport(
            http2=True,
            retries=0,
            trust_env=False,
            limits=httpx2.Limits(
                max_connections=1,
                max_keepalive_connections=1,
                keepalive_expiry=30,
            ),
            socket_options=[(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)],
        )
        return httpx2.AsyncClient(
            transport=transport,
            trust_env=False,
            follow_redirects=False,
            timeout=httpx2.Timeout(connect=5, read=30, write=10, pool=5),
            headers={
                "Authorization": f"Bearer {key.get_secret_value()}",
                "Accept-Encoding": "identity",
            },
        )

    async def generate(self, request: VoiceRequest) -> VoiceResult:
        """Generate once; any unsuccessful attempt leaves a durable held UUID."""
        candidate = _speech_receipt(request)
        with self.store.exclusive() as transaction:
            receipt = transaction.find(candidate)
            if receipt is not None:
                if receipt.audio_digest is None:
                    raise AudioFailure(code="audio_held")
                _ = transaction.audio(receipt)
                return VoiceResult(
                    request_id=request.request_id, artifact_id=str(receipt.artifact_id)
                )
            async with self._client() as client:
                transaction.reserve(candidate)
                try:
                    with anyio.fail_after(PROVIDER_SECONDS):
                        async with client.stream(
                            "POST",
                            _SPEECH_URL,
                            json={
                                "model": TTS_MODEL,
                                "input": request.text,
                                "voice": request.voice,
                                "response_format": "opus",
                                **(
                                    {
                                        "instructions": (
                                            request.delivery.tts_instructions
                                        ),
                                        "speed": request.delivery.tts_speed,
                                    }
                                    if request.delivery is not None
                                    else {}
                                ),
                            },
                        ) as response:
                            audio = await _body(response, MAX_AUDIO_BYTES)
                    normalized = await telegram_opus(audio)
                except (httpx2.HTTPError, TimeoutError):
                    raise AudioFailure(code="audio_provider_unknown") from None
                transaction.complete_audio(candidate, normalized)
            return VoiceResult(
                request_id=request.request_id, artifact_id=str(candidate.artifact_id)
            )

    async def recover(self, request: VoiceRequest) -> VoiceResult | None:
        """Read receipts and existing audio only; never create a provider client."""
        with self.store.exclusive() as transaction:
            receipt = transaction.find(_speech_receipt(request))
            if receipt is None or receipt.audio_digest is None:
                return None
            _ = transaction.audio(receipt)
            return VoiceResult(
                request_id=request.request_id, artifact_id=str(receipt.artifact_id)
            )

    def resolve(self, result: VoiceResult) -> Path:
        """Expose a private, integrity-checked UUID artifact for immediate sending."""
        return self.store.resolve(result)

    async def transcribe(self, request_id: UUID, audio: bytes) -> TranscriptionResult:
        """Root gates owner/metadata first; exact downloaded bytes bind this UUID."""
        candidate = _transcript_receipt(request_id, audio)
        with self.store.exclusive() as transaction:
            receipt = transaction.find(candidate)
            if receipt is not None:
                if receipt.text is None:
                    raise AudioFailure(code="audio_held")
                return TranscriptionResult(request_id=request_id, text=receipt.text)
            async with self._client() as client:
                transaction.reserve(candidate)
                try:
                    with anyio.fail_after(PROVIDER_SECONDS):
                        async with client.stream(
                            "POST",
                            _TRANSCRIBE_URL,
                            data={"model": STT_MODEL},
                            files={"file": ("voice.ogg", audio, "audio/ogg")},
                        ) as response:
                            data = await _body(response, _TRANSCRIPT_BYTES)
                    parsed = ProviderTranscript.model_validate_json(data)
                    if not parsed.text.strip():
                        raise AudioFailure(code="audio_provider_invalid")
                except (httpx2.HTTPError, TimeoutError):
                    raise AudioFailure(code="audio_provider_unknown") from None
                except ValidationError:
                    raise AudioFailure(code="audio_provider_invalid") from None
                transaction.save(candidate.model_copy(update={"text": parsed.text}))
            return TranscriptionResult(request_id=request_id, text=parsed.text)

    async def recover_transcript(
        self,
        request_id: UUID,
        audio: bytes,
    ) -> TranscriptionResult | None:
        """Return exact local text for these bytes, without POST or retries."""
        with self.store.exclusive() as transaction:
            receipt = transaction.find(_transcript_receipt(request_id, audio))
            if receipt is None or receipt.text is None:
                return None
            return TranscriptionResult(request_id=request_id, text=receipt.text)
