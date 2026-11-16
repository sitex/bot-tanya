import hashlib
import json
import stat
from typing import ClassVar, Literal, assert_never

import httpx2
import pytest
from pydantic import BaseModel, ConfigDict

from tanya_bot.app import create_app
from tanya_bot.receipts import exclusive_store
from tanya_bot.schemas import (
    ChatRequest,
    ChatResponse,
    CompanionProfile,
    ErrorResponse,
    Message,
)
from tests.conftest import Harness

pytestmark = pytest.mark.anyio


class OutboundRequest(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    model: str
    store: bool
    max_output_tokens: int
    reasoning: dict[str, Literal["none"]]
    instructions: str
    input: tuple[Message, ...]


class DeliveredProfile(BaseModel):
    profile: CompanionProfile


async def test_character_preferences_reach_provider_context(
    harness: Harness, chat_request: ChatRequest
) -> None:
    profile = CompanionProfile(
        style="direct", user_name="Reader", about="Enjoys astronomy"
    )
    request = chat_request.model_copy(update={"profile": profile})
    async with harness.client() as client:
        response = await client.post("/v1/chat", content=request.model_dump_json())

    assert response.status_code == 200
    payload = OutboundRequest.model_validate_json(harness.wire.calls[0].content)
    delivered = DeliveredProfile.model_validate_json(
        payload.instructions.splitlines()[-1]
    )
    assert delivered.profile == profile


async def test_receipt_from_before_character_profiles_replays_without_provider_call(
    harness: Harness, chat_request: ChatRequest
) -> None:
    legacy_request = json.dumps(
        {
            "request_id": str(chat_request.request_id),
            "messages": [message.model_dump() for message in chat_request.messages],
            "companion_name": chat_request.companion_name,
            "language": chat_request.language,
            "memories": chat_request.memories,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    cached = ChatResponse(
        request_id=chat_request.request_id,
        reply="Legacy saved reply",
        model="legacy-model",
        input_tokens=3,
        output_tokens=2,
    )
    with exclusive_store(harness.directory) as connection:
        _ = connection.execute(
            "INSERT INTO receipts (request_id, digest, response) VALUES (?, ?, ?)",
            (
                str(chat_request.request_id),
                hashlib.sha256(legacy_request.encode()).hexdigest(),
                cached.model_dump_json(),
            ),
        )
        connection.commit()

    async with harness.client() as client:
        response = await client.post("/v1/chat", content=chat_request.model_dump_json())

    assert response.status_code == 200
    assert ChatResponse.model_validate_json(response.content) == cached
    assert harness.wire.calls == []


async def test_success_contract_and_responses_wire(
    harness: Harness, chat_request: ChatRequest
) -> None:
    async with harness.client() as client:
        response = await client.post("/v1/chat", content=chat_request.model_dump_json())

    assert response.status_code == 200
    reply = ChatResponse.model_validate_json(response.content)
    assert reply == ChatResponse(
        request_id=chat_request.request_id,
        reply="Hello.\nHow are you?",
        model="gpt-6-luna-snapshot",
        input_tokens=41,
        output_tokens=7,
    )
    assert len(harness.wire.calls) == 1
    sent = harness.wire.calls[0]
    assert sent.method == "POST"
    assert str(sent.url) == "https://api.openai.com/v1/responses"
    assert sent.headers["authorization"] == "Bearer sk-test-offline"
    payload = OutboundRequest.model_validate_json(sent.content)
    assert (payload.model, payload.store, payload.max_output_tokens) == (
        "gpt-6-luna",
        False,
        800,
    )
    assert payload.input == chat_request.messages
    assert payload.reasoning == {"effort": "none"}


async def test_success_replays_from_disk_after_app_recreation_without_api_key(
    harness: Harness, chat_request: ChatRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with harness.client() as client:
        first = await client.post("/v1/chat", content=chat_request.model_dump_json())
    monkeypatch.delenv("OPENAI_API_KEY")
    restarted = Harness(
        harness.directory,
        harness.wire,
        create_app(harness.directory, httpx2.MockTransport(harness.wire.handle)),
    )

    async with restarted.client() as client:
        replay = await client.post("/v1/chat", content=chat_request.model_dump_json())

    assert first.status_code == replay.status_code == 200
    assert replay.content == first.content
    assert len(harness.wire.calls) == 1
    assert stat.S_IMODE(harness.directory.stat().st_mode) == 0o700
    for path in harness.directory.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    database = (harness.directory / "receipts.sqlite3").read_bytes()
    for message in chat_request.messages:
        assert message.content.encode() not in database
    assert b"sk-test-offline" not in database


@pytest.mark.parametrize("change", ["message", "memory", "name", "language"])
async def test_changed_payload_with_same_id_is_conflict(
    harness: Harness,
    chat_request: ChatRequest,
    change: Literal["message", "memory", "name", "language"],
) -> None:
    async with harness.client() as client:
        initial = await client.post("/v1/chat", content=chat_request.model_dump_json())
        assert initial.status_code == 200
        match change:
            case "message":
                changed = chat_request.model_copy(
                    update={"messages": (Message(role="user", content="Different"),)}
                )
            case "memory":
                changed = chat_request.model_copy(update={"memories": ("Different",)})
            case "name":
                changed = chat_request.model_copy(update={"companion_name": "Sol"})
            case "language":
                changed = chat_request.model_copy(update={"language": "fr"})
            case _:
                assert_never(change)
        response = await client.post("/v1/chat", content=changed.model_dump_json())

    assert response.status_code == 409
    assert ErrorResponse.model_validate_json(response.content).detail.code == (
        "request_conflict"
    )
    assert len(harness.wire.calls) == 1
