import hashlib
import json
from pathlib import Path
from uuid import uuid4

import httpx2
import pytest
from tanya_persona import DeliveryPlan

from tanya_bot.telegram_audio_models import AudioFailure, AudioState
from tanya_bot.telegram_models import VoiceRequest
from tanya_bot.telegram_voice import TelegramVoiceGateway


@pytest.mark.anyio
async def test_legacy_speech_hash_is_byte_compatible_and_plan_changes_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "offline")

    def unavailable(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(503)

    directory = tmp_path / "audio"
    gateway = TelegramVoiceGateway(directory, httpx2.MockTransport(unavailable))
    request = VoiceRequest(request_id=uuid4(), text="Точный текст", voice="cedar")
    data = json.dumps(
        ["gpt-4o-mini-tts", "Точный текст", "cedar", "opus"],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode()
    with pytest.raises(AudioFailure, match="audio_provider_status"):
        _ = await gateway.generate(request)
    state = AudioState.model_validate_json((directory / "state.json").read_bytes())
    assert state.receipts[0].digest == hashlib.sha256(data).hexdigest()
    enhanced = request.model_copy(
        update={
            "request_id": uuid4(),
            "delivery": DeliveryPlan(pace="slow"),
        }
    )
    with pytest.raises(AudioFailure, match="audio_provider_status"):
        _ = await gateway.generate(enhanced)
    after = AudioState.model_validate_json((directory / "state.json").read_bytes())
    assert after.receipts[1].digest != state.receipts[0].digest


@pytest.mark.anyio
async def test_speech_plan_reaches_single_request_and_frozen_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "offline")
    calls: list[httpx2.Request] = []
    plan = DeliveryPlan(pace="slow", kindness="gentle")

    def wire(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        assert json.loads(request.content) == {
            "model": "gpt-4o-mini-tts",
            "input": "One answer",
            "voice": "cedar",
            "response_format": "opus",
            "speed": 0.9,
            "instructions": plan.tts_instructions,
        }
        return httpx2.Response(503)

    gateway = TelegramVoiceGateway(tmp_path / "audio", httpx2.MockTransport(wire))
    request = VoiceRequest(
        request_id=uuid4(),
        text="One answer",
        voice="cedar",
        delivery=plan,
    )
    with pytest.raises(AudioFailure, match="audio_provider_status"):
        _ = await gateway.generate(request)
    assert await gateway.recover(request) is None
    with pytest.raises(AudioFailure, match="audio_held"):
        _ = await gateway.generate(request)
    changed = request.model_copy(update={"delivery": DeliveryPlan(pace="brisk")})
    with pytest.raises(AudioFailure, match="audio_conflict"):
        _ = await gateway.recover(changed)
    assert len(calls) == 1
