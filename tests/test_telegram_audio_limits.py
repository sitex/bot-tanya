import os
from pathlib import Path
from typing import NoReturn
from uuid import uuid4

import anyio
import httpx2
import pytest

from tanya_bot.telegram_audio_models import (
    MAX_AUDIO_BYTES,
    MAX_REQUESTS,
    MAX_TTS_CHARACTERS,
    AudioFailure,
)
from tanya_bot.telegram_models import VoiceRequest
from tanya_bot.telegram_voice import TelegramVoiceGateway


@pytest.mark.anyio
@pytest.mark.parametrize("audio", [b"", b"x" * (MAX_AUDIO_BYTES + 1)])
async def test_invalid_input_bytes_never_reach_provider(
    tmp_path: Path, audio: bytes
) -> None:
    provider = TelegramVoiceGateway(tmp_path / "voice")
    with pytest.raises(AudioFailure, match="audio_invalid_request"):
        _ = await provider.transcribe(uuid4(), audio)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("text", "voice"),
    [
        ("   ", "marin"),
        ("x" * (MAX_TTS_CHARACTERS + 1), "cedar"),
        ("text", "fable"),
        ("text", "../marin"),
    ],
)
async def test_invalid_speech_never_reaches_provider(
    tmp_path: Path, text: str, voice: str
) -> None:
    provider = TelegramVoiceGateway(tmp_path / "voice")
    with pytest.raises(AudioFailure, match="audio_invalid_request"):
        _ = await provider.generate(
            VoiceRequest(request_id=uuid4(), text=text, voice=voice)
        )


@pytest.mark.anyio
async def test_capacity_never_evicts_completed_uuid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    calls: list[httpx2.Request] = []

    async def reply(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        return httpx2.Response(200, json={"text": "retained"})

    provider = TelegramVoiceGateway(tmp_path / "voice", httpx2.MockTransport(reply))
    first_id = uuid4()
    first = await provider.transcribe(first_id, b"voice")
    for _ in range(MAX_REQUESTS - 1):
        _ = await provider.transcribe(uuid4(), b"voice")
    with pytest.raises(AudioFailure, match="audio_capacity"):
        _ = await provider.transcribe(uuid4(), b"voice")
    assert await provider.recover_transcript(first_id, b"voice") == first
    assert len(calls) == MAX_REQUESTS


@pytest.mark.anyio
async def test_disk_failure_before_reservation_prevents_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    calls: list[httpx2.Request] = []

    async def reply(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        return httpx2.Response(200, json={"text": "unreachable"})

    provider = TelegramVoiceGateway(tmp_path / "voice", httpx2.MockTransport(reply))

    def disk_full(_fd: int) -> NoReturn:
        raise OSError

    monkeypatch.setattr(os, "fsync", disk_full)
    with pytest.raises(AudioFailure, match="audio_storage"):
        _ = await provider.transcribe(uuid4(), b"voice")
    assert calls == []


@pytest.mark.anyio
async def test_completion_disk_failure_preserves_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    path = tmp_path / "voice"
    calls: list[httpx2.Request] = []
    request_id = uuid4()

    def disk_full(_fd: int) -> NoReturn:
        raise OSError

    async def reply(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        monkeypatch.setattr(os, "fsync", disk_full)
        return httpx2.Response(200, json={"text": "not durable"})

    provider = TelegramVoiceGateway(path, httpx2.MockTransport(reply))
    with pytest.raises(AudioFailure, match="audio_storage"):
        _ = await provider.transcribe(request_id, b"voice")
    assert await provider.recover_transcript(request_id, b"voice") is None
    with pytest.raises(AudioFailure, match="audio_held"):
        _ = await provider.transcribe(request_id, b"voice")
    assert len(calls) == 1


@pytest.mark.anyio
async def test_parallel_provider_cannot_cross_store_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    ready = anyio.Event()
    path = tmp_path / "voice"

    async def wait(_request: httpx2.Request) -> NoReturn:
        ready.set()
        await anyio.sleep_forever()
        pytest.fail("cancelled provider resumed")

    first = TelegramVoiceGateway(path, httpx2.MockTransport(wait))
    second = TelegramVoiceGateway(path, httpx2.MockTransport(wait))
    async with anyio.create_task_group() as tasks:
        _ = tasks.start_soon(first.transcribe, uuid4(), b"voice")
        with anyio.fail_after(2):
            await ready.wait()
        with pytest.raises(AudioFailure, match="audio_busy"):
            _ = await second.transcribe(uuid4(), b"other voice")
        tasks.cancel_scope.cancel()


@pytest.mark.anyio
async def test_invalid_audio_conversion_remains_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    provider = TelegramVoiceGateway(
        tmp_path / "voice",
        httpx2.MockTransport(
            lambda _: httpx2.Response(200, content=b"not an audio file"),
        ),
    )
    request = VoiceRequest(request_id=uuid4(), text="Hello", voice="marin")
    with pytest.raises(AudioFailure, match="audio_conversion"):
        _ = await provider.generate(request)
    assert await provider.recover(request) is None
    with pytest.raises(AudioFailure, match="audio_held"):
        _ = await provider.generate(request)
