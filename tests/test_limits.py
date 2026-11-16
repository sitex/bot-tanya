from typing import Literal, assert_never
from uuid import uuid4

import anyio
import httpx2
import pytest

from tanya_bot.app import create_app
from tanya_bot.schemas import ChatRequest, CompanionProfile, ErrorResponse, Message
from tests.conftest import Harness, completed_response

pytestmark = pytest.mark.anyio
type InvalidContext = Literal[
    "turns", "characters", "memories", "role", "uuid", "profile"
]


@pytest.mark.parametrize(
    "kind", ["turns", "characters", "memories", "role", "uuid", "profile"]
)
async def test_invalid_or_oversized_context_is_rejected_before_provider(
    harness: Harness, chat_request: ChatRequest, kind: InvalidContext
) -> None:
    match kind:
        case "turns":
            changed = chat_request.model_copy(
                update={
                    "messages": tuple(
                        Message(role="user", content="x") for _ in range(33)
                    )
                }
            )
        case "characters":
            changed = chat_request.model_copy(
                update={"messages": (Message(role="user", content="x" * 16000),)}
            )
        case "memories":
            changed = chat_request.model_copy(update={"memories": ("x" * 16000,)})
        case "profile":
            changed = chat_request.model_copy(
                update={
                    "messages": (Message(role="user", content="x" * 15_000),),
                    "profile": CompanionProfile(about="x" * 2_000),
                }
            )
        case "role" | "uuid":
            changed = chat_request
        case _:
            assert_never(kind)
    content = changed.model_dump_json()
    match kind:
        case "role":
            content = content.replace('"role":"user"', '"role":"system"')
        case "uuid":
            content = content.replace(str(chat_request.request_id), "invalid-id")
        case "turns" | "characters" | "memories" | "profile":
            pass
        case _:
            assert_never(kind)
    async with harness.client() as client:
        response = await client.post("/v1/chat", content=content)

    assert response.status_code == 422
    assert ErrorResponse.model_validate_json(response.content).detail.code == (
        "invalid_request"
    )
    assert harness.wire.calls == []


async def test_maximum_valid_context_and_turn_count_are_accepted(
    harness: Harness,
) -> None:
    request = ChatRequest(
        request_id=uuid4(),
        messages=(
            *(Message(role="user", content="x") for _ in range(31)),
            Message(role="user", content="x" * (16000 - 31 - 5 - 2)),
        ),
    )
    async with harness.client() as client:
        response = await client.post("/v1/chat", content=request.model_dump_json())

    assert response.status_code == 200
    assert len(harness.wire.calls) == 1


async def test_oversized_wire_body_is_rejected(harness: Harness) -> None:
    async with harness.client() as client:
        response = await client.post("/v1/chat", content=b" " * 128001)

    assert response.status_code == 413
    assert harness.wire.calls == []


async def test_parallel_apps_allow_one_provider_call_and_hold_cancelled_send(
    harness: Harness, chat_request: ChatRequest
) -> None:
    started, release = anyio.Event(), anyio.Event()

    async def blocked_provider(request: httpx2.Request) -> httpx2.Response:
        harness.wire.calls.append(request)
        started.set()
        await release.wait()
        return completed_response()

    first_app = Harness(
        harness.directory,
        harness.wire,
        create_app(harness.directory, httpx2.MockTransport(blocked_provider)),
    )
    async with first_app.client() as first_client, harness.client() as other_client:

        async def send() -> None:
            _ = await first_client.post(
                "/v1/chat", content=chat_request.model_dump_json()
            )

        with anyio.fail_after(5):
            async with anyio.create_task_group() as group:
                _ = group.start_soon(send)
                await started.wait()
                concurrent = await other_client.post(
                    "/v1/chat", content=chat_request.model_dump_json()
                )
                assert concurrent.status_code == 503
                assert ErrorResponse.model_validate_json(
                    concurrent.content
                ).detail.code == ("server_busy")
                group.cancel_scope.cancel()
        held = await other_client.post(
            "/v1/chat", content=chat_request.model_dump_json()
        )

    assert held.status_code == 409
    assert ErrorResponse.model_validate_json(held.content).detail.code == (
        "request_outcome_unknown"
    )
    assert len(harness.wire.calls) == 1
