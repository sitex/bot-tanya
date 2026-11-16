import json
from uuid import uuid4

import httpx2
import pytest
from tanya_memory import DerivedSummary, TurnContext
from tanya_persona import DeliveryPlan

from tanya_bot.schemas import ChatRequest
from tanya_bot.turns import TurnRequest, TurnResponse
from tests.conftest import Harness

pytestmark = pytest.mark.anyio


def envelope(chat: ChatRequest) -> TurnRequest:
    return TurnRequest(
        chat=chat,
        context=TurnContext(
            epoch=3,
            memory_revision=2,
            summary=DerivedSummary(text="Earlier topic", source_ids=(uuid4(),)),
            source_ids=(uuid4(),),
            user_entry_id=uuid4(),
            assistant_entry_id=uuid4(),
        ),
    )


def provider_answer() -> httpx2.Response:
    answer = json.dumps(
        {
            "reply": "Current reply",
            "memory_updates": [],
            "summary": "New summary",
            "delivery": DeliveryPlan().model_dump(),
        }
    )
    return httpx2.Response(
        200,
        json={
            "status": "completed",
            "model": "offline",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
            },
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": answer},
                    ],
                }
            ],
        },
    )


async def test_v2_atomic_summary_delivery_cache_and_frozen_identity(
    harness: Harness,
    chat_request: ChatRequest,
) -> None:
    turn = envelope(chat_request)
    harness.wire.outcomes.append(provider_answer())
    async with harness.client() as client:
        response = await client.post(
            "/v2/telegram/chat", content=turn.model_dump_json()
        )
        assert response.status_code == 200
        saved = TurnResponse.model_validate_json(response.content)
        replay = await client.post("/v2/telegram/chat", content=turn.model_dump_json())
        changed = turn.model_copy(
            update={"context": turn.context.model_copy(update={"epoch": 4})}
        )
        conflict = await client.post(
            "/v2/telegram/chat", content=changed.model_dump_json()
        )
    assert saved.summary == "New summary"
    assert saved.delivery == DeliveryPlan()
    assert replay.content == response.content
    assert conflict.status_code == 409
    assert len(harness.wire.calls) == 1


@pytest.mark.parametrize(
    "outcome",
    [
        httpx2.Response(503),
        httpx2.Response(200, content=b"invalid json"),
        httpx2.ReadTimeout("offline unknown"),
    ],
)
async def test_v2_unknown_is_held_without_repair_or_repeat(
    harness: Harness,
    chat_request: ChatRequest,
    outcome: httpx2.Response | httpx2.RequestError,
) -> None:
    turn = envelope(chat_request)
    harness.wire.outcomes.append(outcome)
    async with harness.client() as client:
        first = await client.post("/v2/telegram/chat", content=turn.model_dump_json())
        second = await client.post("/v2/telegram/chat", content=turn.model_dump_json())
    assert first.status_code == 502
    assert second.status_code == 409
    assert second.json()["detail"]["code"] == "request_outcome_unknown"
    assert len(harness.wire.calls) == 1


async def test_v2_summary_is_included_in_total_context_budget(
    harness: Harness,
    chat_request: ChatRequest,
) -> None:
    turn = envelope(chat_request)
    chat = chat_request.model_copy(update={"memories": ("m" * 15_000,)})
    oversized = turn.model_copy(
        update={
            "chat": chat,
            "context": turn.context.model_copy(
                update={
                    "summary": DerivedSummary(text="s" * 1800),
                }
            ),
        }
    )
    async with harness.client() as client:
        response = await client.post(
            "/v2/telegram/chat", content=oversized.model_dump_json()
        )
    assert response.status_code == 422
    assert harness.wire.calls == []
