import os
import stat
import struct
from pathlib import Path
from typing import Literal, assert_never
from uuid import UUID, uuid4

import httpx2
import pytest

from tanya_bot.telegram_audio_models import AudioFailure
from tanya_bot.telegram_models import VoiceRequest, VoiceResult
from tanya_bot.telegram_voice import TelegramVoiceGateway


@pytest.fixture
def pcm_fixture() -> bytes:
    samples = b"\x00\x00" * 2400
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(samples))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, 24_000, 48_000, 2, 16)
        + b"data"
        + struct.pack("<I", len(samples))
        + samples
    )


@pytest.mark.anyio
async def test_speech_normalizes_then_resolves_after_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pcm_fixture: bytes,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    calls: list[httpx2.Request] = []

    async def provider(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        return httpx2.Response(200, content=pcm_fixture)

    path = tmp_path / "voice"
    request = VoiceRequest(request_id=uuid4(), text="Hello", voice="cedar")
    first = TelegramVoiceGateway(path, httpx2.MockTransport(provider))
    result = await first.generate(request)
    restarted = TelegramVoiceGateway(path, httpx2.MockTransport(provider))

    assert result.request_id == request.request_id
    assert UUID(result.artifact_id) != request.request_id
    audio_path = restarted.resolve(result)
    audio = audio_path.read_bytes()
    assert audio.startswith(b"OggS")
    assert b"OpusHead\x01\x01" in audio
    assert await restarted.recover(request) == result
    assert await restarted.generate(request) == result
    assert len(calls) == 1
    assert stat.S_IMODE(path.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in path.iterdir())


@pytest.mark.anyio
async def test_artifact_hash_mismatch_and_wrong_binding_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pcm_fixture: bytes,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    provider = TelegramVoiceGateway(
        tmp_path / "voice",
        httpx2.MockTransport(
            lambda _: httpx2.Response(200, content=pcm_fixture),
        ),
    )
    request = VoiceRequest(request_id=uuid4(), text="Hello", voice="marin")
    result = await provider.generate(request)
    path = provider.resolve(result)
    with pytest.raises(AudioFailure, match="audio_invalid_artifact"):
        _ = provider.resolve(
            VoiceResult(request_id=uuid4(), artifact_id=result.artifact_id)
        )
    _ = path.write_bytes(b"corrupted audio")
    with pytest.raises(AudioFailure, match="audio_corrupt"):
        _ = provider.resolve(result)
    with pytest.raises(AudioFailure, match="audio_corrupt"):
        _ = await provider.recover(request)


@pytest.mark.parametrize(
    "artifact",
    [
        "../state.json",
        "/etc/passwd",
        str(uuid4()).upper(),
        "https://example.invalid/file",
    ],
)
def test_untrusted_artifact_is_never_a_path(tmp_path: Path, artifact: str) -> None:
    provider = TelegramVoiceGateway(tmp_path / "voice")
    with pytest.raises(AudioFailure, match="audio_invalid_artifact"):
        _ = provider.resolve(VoiceResult(request_id=uuid4(), artifact_id=artifact))


@pytest.mark.parametrize(
    "kind", ["directory", "parent", "state", "lock", "hardlink", "mode"]
)
def test_unsafe_private_storage_fails_closed(
    tmp_path: Path,
    kind: Literal["directory", "parent", "state", "lock", "hardlink", "mode"],
) -> None:
    path = tmp_path / "voice"
    _ = TelegramVoiceGateway(path)
    target = tmp_path / "target"
    _ = target.write_bytes(b"private marker")
    target.chmod(0o600)
    match kind:
        case "directory":
            moved = tmp_path / "moved"
            _ = path.rename(moved)
            path.symlink_to(moved, target_is_directory=True)
        case "parent":
            parent = tmp_path / "linked"
            parent.symlink_to(tmp_path, target_is_directory=True)
            path = parent / "must-not-create"
        case "state" | "lock":
            victim = path / ("state.json" if kind == "state" else "state.lock")
            victim.unlink()
            victim.symlink_to(target)
        case "hardlink":
            victim = path / "state.json"
            victim.unlink()
            os.link(target, victim)
        case "mode":
            (path / "state.json").chmod(0o644)
        case _:
            assert_never(kind)

    with pytest.raises(AudioFailure):
        _ = TelegramVoiceGateway(path)
    assert target.read_bytes() == b"private marker"
    assert not (tmp_path / "must-not-create").exists()


@pytest.mark.anyio
@pytest.mark.parametrize("corruption", [b"{", b'{"version":99}', b'{"receipts":[{}]}'])
async def test_corrupt_state_never_resets_or_calls_provider(
    tmp_path: Path,
    corruption: bytes,
) -> None:
    path = tmp_path / "voice"
    provider = TelegramVoiceGateway(path)
    _ = (path / "state.json").write_bytes(corruption)
    with pytest.raises(AudioFailure, match="audio_corrupt"):
        _ = await provider.recover_transcript(uuid4(), b"voice")
    assert (path / "state.json").read_bytes() == corruption


def test_missing_ledger_after_restart_is_corruption(tmp_path: Path) -> None:
    path = tmp_path / "voice"
    _ = TelegramVoiceGateway(path)
    (path / "state.json").unlink()
    with pytest.raises(AudioFailure, match="audio_corrupt"):
        _ = TelegramVoiceGateway(path)
    assert not (path / "state.json").exists()
