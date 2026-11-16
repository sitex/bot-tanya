import json
from typing import Literal

import httpx2
import pytest

from tanya_bot.schemas import ChatRequest, ChatResponse, ErrorResponse, MemoryUpdate
from tests.conftest import Harness

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize("mixed", [False, True])
async def test_refusal_blocks_reply_and_memory_publication_without_resending(
    harness: Harness,
    chat_request: ChatRequest,
    *,
    mixed: bool,
) -> None:
    request = chat_request.model_copy(update={"capture_memory": True})
    parts = [{"type": "refusal", "refusal": "PRIVATE PROVIDER DETAIL"}]
    if mixed:
        parts.append(
            {
                "type": "output_text",
                "text": '{"reply":"Do not publish","memory_updates":[]}',
            }
        )
    harness.wire.outcomes.append(
        httpx2.Response(
            200,
            json={
                "status": "completed",
                "model": "memory-model",
                "output": [{"type": "message", "role": "assistant", "content": parts}],
                "usage": {"input_tokens": 8, "output_tokens": 1},
            },
        )
    )

    async with harness.client() as client:
        response = await client.post("/v1/chat", content=request.model_dump_json())
        replay = await client.post("/v1/chat", content=request.model_dump_json())

    assert response.status_code == 502
    assert (
        ErrorResponse.model_validate_json(response.content).detail.code
        == "provider_refused"
    )
    assert "PRIVATE PROVIDER DETAIL" not in response.text
    assert replay.status_code == 409
    assert len(harness.wire.calls) == 1


def structured_output(text: str) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "status": "completed",
            "model": "memory-model",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            ],
            "usage": {"input_tokens": 30, "output_tokens": 20},
        },
    )


@pytest.mark.parametrize("remembered", ["I enjoy watercolor.", " fact ", "x" * 600])
async def test_memory_and_reply_replay_together_without_another_provider_call(
    harness: Harness,
    chat_request: ChatRequest,
    remembered: str,
) -> None:
    request = chat_request.model_copy(
        update={"capture_memory": True, "memories": (remembered,)}
    )
    updates = (
        MemoryUpdate(action="remember", text="I am learning to draw."),
        MemoryUpdate(action="forget", text=remembered),
    )
    harness.wire.outcomes.append(
        structured_output(
            json.dumps(
                {
                    "reply": "We can start with a small watercolor exercise.",
                    "memory_updates": [update.model_dump() for update in updates],
                }
            )
        )
    )

    async with harness.client() as client:
        first = await client.post("/v1/chat", content=request.model_dump_json())
        replay = await client.post("/v1/chat", content=request.model_dump_json())

    assert first.status_code == replay.status_code == 200
    reply = ChatResponse.model_validate_json(replay.content)
    assert reply.reply == "We can start with a small watercolor exercise."
    assert reply.memory_updates == updates
    assert first.content == replay.content
    assert len(harness.wire.calls) == 1


@pytest.mark.parametrize(
    "bad_output",
    [
        "not JSON",
        '{"reply":"Hello","memory_updates":[{"action":"erase_all","text":"x"}]}',
        json.dumps(
            {
                "reply": "Hello",
                "memory_updates": [{"action": "remember", "text": "x" * 501}],
            }
        ),
    ],
)
async def test_malformed_memory_output_is_not_published_or_sent_again(
    harness: Harness,
    chat_request: ChatRequest,
    bad_output: str,
) -> None:
    request = chat_request.model_copy(update={"capture_memory": True})
    harness.wire.outcomes.append(structured_output(bad_output))

    async with harness.client() as client:
        failed = await client.post("/v1/chat", content=request.model_dump_json())
        repeated = await client.post("/v1/chat", content=request.model_dump_json())

    assert failed.status_code == 502
    assert repeated.status_code == 409
    assert len(harness.wire.calls) == 1


@pytest.mark.parametrize("action", ["remember", "forget"])
async def test_model_cannot_forget_a_fact_outside_supplied_memories(
    harness: Harness,
    chat_request: ChatRequest,
    action: Literal["remember", "forget"],
) -> None:
    request = chat_request.model_copy(update={"capture_memory": True})
    update = MemoryUpdate(action=action, text="An unrelated local fact")
    harness.wire.outcomes.append(
        structured_output(
            json.dumps(
                {
                    "reply": "A bounded answer",
                    "memory_updates": [update.model_dump()],
                }
            )
        )
    )

    async with harness.client() as client:
        response = await client.post("/v1/chat", content=request.model_dump_json())

    assert response.status_code == 200
    assert ChatResponse.model_validate_json(response.content).memory_updates == (
        (update,) if action == "remember" else ()
    )
