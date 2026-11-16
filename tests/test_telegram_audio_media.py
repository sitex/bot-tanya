import struct
from pathlib import Path
from typing import NoReturn
from uuid import uuid4

import anyio
import httpx2
import pytest

from tanya_bot import telegram_voice
from tanya_bot.telegram_audio_models import MAX_AUDIO_SECONDS, AudioFailure
from tanya_bot.telegram_models import VoiceRequest
from tanya_bot.telegram_voice import TelegramVoiceGateway


@pytest.mark.anyio
@pytest.mark.parametrize("seconds", [MAX_AUDIO_SECONDS, MAX_AUDIO_SECONDS + 1])
async def test_normalized_speech_duration_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seconds: int,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    samples = b"\x00\x00" * (8000 * seconds)
    audio = (
        b"RIFF"
        + struct.pack("<I", 36 + len(samples))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, 8000, 16_000, 2, 16)
        + b"data"
        + struct.pack("<I", len(samples))
        + samples
    )
    provider = TelegramVoiceGateway(
        tmp_path / "voice",
        httpx2.MockTransport(
            lambda _: httpx2.Response(200, content=audio),
        ),
    )
    request = VoiceRequest(request_id=uuid4(), text="Hello", voice="marin")
    if seconds > MAX_AUDIO_SECONDS:
        with pytest.raises(AudioFailure, match="audio_invalid_media"):
            _ = await provider.generate(request)
        assert await provider.recover(request) is None
    else:
        result = await provider.generate(request)
        assert provider.resolve(result).read_bytes().startswith(b"OggS")


@pytest.mark.anyio
async def test_total_provider_deadline_leaves_held_uuid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    monkeypatch.setattr(telegram_voice, "PROVIDER_SECONDS", 0)
    calls: list[httpx2.Request] = []

    async def stall(request: httpx2.Request) -> NoReturn:
        calls.append(request)
        await anyio.sleep_forever()
        pytest.fail("stalled provider resumed")

    provider = TelegramVoiceGateway(tmp_path / "voice", httpx2.MockTransport(stall))
    request_id = uuid4()
    with pytest.raises(AudioFailure, match="audio_provider_unknown"):
        _ = await provider.transcribe(request_id, b"voice")
    assert await provider.recover_transcript(request_id, b"voice") is None
    with pytest.raises(AudioFailure, match="audio_held"):
        _ = await provider.transcribe(request_id, b"voice")
    assert len(calls) <= 1


@pytest.mark.anyio
async def test_missing_ffmpeg_is_safe_and_does_not_repeat_post(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    calls: list[httpx2.Request] = []

    async def reply(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        return httpx2.Response(200, content=b"audio")

    async def missing(*_args: str, **_kwargs: bytes) -> NoReturn:
        raise FileNotFoundError

    monkeypatch.setattr(anyio, "run_process", missing)
    provider = TelegramVoiceGateway(tmp_path / "voice", httpx2.MockTransport(reply))
    request = VoiceRequest(request_id=uuid4(), text="Hello", voice="marin")
    with pytest.raises(AudioFailure, match="audio_conversion"):
        _ = await provider.generate(request)
    with pytest.raises(AudioFailure, match="audio_held"):
        _ = await provider.generate(request)
    assert len(calls) == 1
