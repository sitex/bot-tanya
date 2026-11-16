from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn, assert_never, override
from uuid import UUID, uuid4

import anyio
import httpx2
import pytest

from tanya_bot.telegram_audio_models import AudioFailure
from tanya_bot.telegram_models import GatewayFailure, VoiceRequest
from tanya_bot.telegram_voice import TelegramVoiceGateway


@dataclass(frozen=True, slots=True)
class AudioWire:
    calls: list[httpx2.Request] = field(default_factory=list)
    outcomes: list[httpx2.Response | httpx2.RequestError] = field(default_factory=list)

    async def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.calls.append(request)
        outcome = self.outcomes.pop(0)
        match outcome:
            case httpx2.Response():
                return outcome
            case httpx2.RequestError():
                raise outcome
            case _:
                assert_never(outcome)


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> AudioWire:
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    return AudioWire()


def gateway(path: Path, wire: AudioWire) -> TelegramVoiceGateway:
    return TelegramVoiceGateway(path, httpx2.MockTransport(wire.handle))


@pytest.mark.anyio
async def test_transcript_is_cached_exactly_across_restart(
    tmp_path: Path, wire: AudioWire
) -> None:
    wire.outcomes.append(
        httpx2.Response(
            200,
            json={
                "text": "  Exact words.\n",
                "languages": [{"code": "en"}],
            },
        )
    )
    request_id = uuid4()
    path = tmp_path / "audio"
    first = await gateway(path, wire).transcribe(request_id, b"owner-voice")

    restarted = gateway(path, wire)
    assert await restarted.recover_transcript(request_id, b"owner-voice") == first
    assert await restarted.transcribe(request_id, b"owner-voice") == first
    assert first.text == "  Exact words.\n"
    assert first.request_id == request_id
    assert len(wire.calls) == 1
    posted = wire.calls[0]
    assert str(posted.url) == "https://api.openai.com/v1/audio/transcriptions"
    assert b"gpt-transcribe" in posted.content
    assert b"owner-voice" in posted.content


@pytest.mark.anyio
@pytest.mark.parametrize(
    "outcome",
    [
        httpx2.Response(503, text="private provider body"),
        httpx2.Response(307, headers={"location": "https://elsewhere.invalid"}),
        httpx2.Response(200, content=b"broken json"),
        httpx2.Response(200, json={"text": ""}),
        httpx2.Response(200, json={"text": "   "}),
        httpx2.Response(200, json={"text": "x" * 16_001}),
        httpx2.ReadTimeout("private network detail"),
    ],
)
async def test_failed_transcription_stays_held_after_restart(
    tmp_path: Path,
    wire: AudioWire,
    outcome: httpx2.Response | httpx2.RequestError,
) -> None:
    request_id = uuid4()
    path = tmp_path / "audio"
    wire.outcomes.append(outcome)
    with pytest.raises(GatewayFailure) as failure:
        _ = await gateway(path, wire).transcribe(request_id, b"voice")
    assert "private" not in str(failure.value)
    restarted = gateway(path, wire)
    assert await restarted.recover_transcript(request_id, b"voice") is None
    with pytest.raises(AudioFailure, match="audio_held"):
        _ = await restarted.transcribe(request_id, b"voice")
    assert len(wire.calls) == 1


@pytest.mark.anyio
async def test_same_uuid_different_bytes_or_operation_conflicts(
    tmp_path: Path, wire: AudioWire
) -> None:
    wire.outcomes.append(httpx2.Response(200, json={"text": "first"}))
    provider = gateway(tmp_path / "audio", wire)
    request_id = uuid4()
    _ = await provider.transcribe(request_id, b"first")

    with pytest.raises(AudioFailure, match="audio_conflict"):
        _ = await provider.transcribe(request_id, b"different")
    with pytest.raises(AudioFailure, match="audio_conflict"):
        _ = await provider.recover_transcript(request_id, b"different")
    with pytest.raises(AudioFailure, match="audio_conflict"):
        _ = await provider.generate(
            VoiceRequest(request_id=request_id, text="text", voice="marin")
        )
    assert len(wire.calls) == 1


@pytest.mark.anyio
async def test_cancelled_post_is_durably_held_before_wire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    path = tmp_path / "audio"
    request_id = uuid4()
    posted = anyio.Event()
    calls: list[UUID] = []

    async def cancel(_request: httpx2.Request) -> NoReturn:
        calls.append(request_id)
        data = (path / "state.json").read_bytes()
        assert str(request_id).encode() in data
        posted.set()
        await anyio.sleep_forever()
        pytest.fail("cancelled task resumed")

    provider = TelegramVoiceGateway(path, httpx2.MockTransport(cancel))
    async with anyio.create_task_group() as group:
        _ = group.start_soon(provider.transcribe, request_id, b"voice")
        with anyio.fail_after(2):
            await posted.wait()
        group.cancel_scope.cancel()

    restarted = TelegramVoiceGateway(path, httpx2.MockTransport(cancel))
    assert await restarted.recover_transcript(request_id, b"voice") is None
    with pytest.raises(AudioFailure, match="audio_held"):
        _ = await restarted.transcribe(request_id, b"voice")
    assert calls == [request_id]


@pytest.mark.anyio
async def test_recovery_absence_uses_no_credentials_or_wire(
    tmp_path: Path, wire: AudioWire, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = gateway(tmp_path / "audio", wire)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    before = (tmp_path / "audio" / "state.json").read_bytes()
    assert await provider.recover_transcript(uuid4(), b"audio") is None
    assert (
        await provider.recover(
            VoiceRequest(request_id=uuid4(), text="Hello", voice="cedar")
        )
        is None
    )
    assert (tmp_path / "audio" / "state.json").read_bytes() == before
    assert wire.calls == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    "voice",
    [
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
    ],
)
async def test_ios_voice_is_accepted_but_unknown_post_is_held(
    tmp_path: Path, wire: AudioWire, voice: str
) -> None:
    provider = gateway(tmp_path / "audio", wire)
    wire.outcomes.append(httpx2.Response(500))
    request = VoiceRequest(request_id=uuid4(), text="Hello", voice=voice)
    with pytest.raises(AudioFailure, match="audio_provider_status"):
        _ = await provider.generate(request)
    assert len(wire.calls) == 1
    assert str(wire.calls[0].url) == "https://api.openai.com/v1/audio/speech"
    assert b"gpt-4o-mini-tts" in wire.calls[0].content
    assert b'"response_format":"opus"' in wire.calls[0].content
    with pytest.raises(AudioFailure, match="audio_held"):
        _ = await provider.generate(request)


class OversizedStream(httpx2.AsyncByteStream):
    @override
    async def __aiter__(self) -> AsyncIterator[bytes]:
        for _ in range(82):
            yield b"x" * (64 * 1024)
        pytest.fail("provider consumed beyond the audio byte budget")


@pytest.mark.anyio
async def test_streaming_speech_stops_at_byte_budget(
    tmp_path: Path, wire: AudioWire
) -> None:
    wire.outcomes.append(httpx2.Response(200, stream=OversizedStream()))
    provider = gateway(tmp_path / "audio", wire)
    request = VoiceRequest(request_id=uuid4(), text="Hello", voice="marin")
    with pytest.raises(AudioFailure, match="audio_provider_capacity"):
        _ = await provider.generate(request)
    assert await provider.recover(request) is None
    assert len(wire.calls) == 1
