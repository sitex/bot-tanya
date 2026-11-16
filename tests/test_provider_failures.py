from typing import Literal, assert_never

import httpx2
import pytest

from tanya_bot.app import create_app
from tanya_bot.schemas import ChatRequest, ErrorResponse
from tests.conftest import Harness

pytestmark = pytest.mark.anyio
type Failure = Literal[
    "credit",
    "subscription_credit",
    "rate",
    "usage_limit_reached",
    "auth",
    "server",
    "timeout",
    "connection",
    "invalid",
    "incomplete",
    "empty",
    "redirect",
]


@pytest.mark.parametrize(
    "failure",
    [
        "credit",
        "subscription_credit",
        "rate",
        "usage_limit_reached",
        "auth",
        "server",
        "timeout",
        "connection",
        "invalid",
        "incomplete",
        "empty",
        "redirect",
    ],
)
async def test_provider_failure_is_safe_and_never_automatically_resent(
    harness: Harness, chat_request: ChatRequest, failure: Failure
) -> None:
    match failure:
        case "credit" | "subscription_credit":
            provider_code = (
                "credit_balance_exhausted"
                if failure == "subscription_credit"
                else "insufficient_quota"
            )
            outcome = httpx2.Response(
                429, json={"error": {"code": provider_code, "message": "SECRET"}}
            )
            status, code = 402, "credit_limit"
        case "rate" | "usage_limit_reached":
            outcome = httpx2.Response(
                429, json={"error": {"code": failure, "message": "SECRET"}}
            )
            status, code = 429, "provider_rate_limit"
        case "auth":
            outcome = httpx2.Response(401, content=b"SECRET")
            status, code = 502, "provider_auth"
        case "server":
            outcome = httpx2.Response(500, content=b"SECRET")
            status, code = 502, "provider_error"
        case "redirect":
            outcome = httpx2.Response(307, headers={"Location": "https://other.test"})
            status, code = 502, "provider_error"
        case "timeout":
            outcome = httpx2.ReadTimeout("SECRET")
            status, code = 504, "provider_timeout"
        case "connection":
            outcome = httpx2.ConnectError("SECRET")
            status, code = 502, "provider_unavailable"
        case "invalid":
            outcome = httpx2.Response(200, content=b"SECRET")
            status, code = 502, "provider_invalid_response"
        case "incomplete" | "empty":
            outcome = httpx2.Response(
                200,
                json={
                    "status": "incomplete" if failure == "incomplete" else "completed",
                    "model": "gpt-6-luna",
                    "output": [],
                    "usage": {"input_tokens": 8, "output_tokens": 0},
                },
            )
            status, code = (
                502,
                (
                    "provider_incomplete"
                    if failure == "incomplete"
                    else "provider_empty_reply"
                ),
            )
        case _:
            assert_never(failure)
    harness.wire.outcomes.append(outcome)
    async with harness.client() as client:
        first = await client.post("/v1/chat", content=chat_request.model_dump_json())
    restarted = Harness(
        harness.directory,
        harness.wire,
        create_app(harness.directory, httpx2.MockTransport(harness.wire.handle)),
    )

    async with restarted.client() as client:
        duplicate = await client.post(
            "/v1/chat", content=chat_request.model_dump_json()
        )

    assert first.status_code == status
    assert ErrorResponse.model_validate_json(first.content).detail.code == code
    assert "SECRET" not in first.text
    assert duplicate.status_code == 409
    assert ErrorResponse.model_validate_json(duplicate.content).detail.code == (
        "request_outcome_unknown"
    )
    assert len(harness.wire.calls) == 1
