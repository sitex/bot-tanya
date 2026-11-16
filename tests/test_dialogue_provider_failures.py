import json
from collections.abc import AsyncIterator
from typing import override

import httpx2
import pytest
from pydantic import SecretStr

from tanya_bot.conversation_memory import JSONValue
from tanya_bot.dialogue_provider import generate, payload
from tanya_bot.errors import ServiceError
from tests.test_dialogue_provider import (
    ProviderWire,
    answer_item,
    call_item,
    envelope,
    reasoning_item,
    root,
)

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize(
    "output",
    [
        [],
        [reasoning_item()],
        [call_item(), call_item()],
        [call_item(), answer_item()],
        [answer_item(), answer_item()],
        [{**call_item(), "name": "gmail_send"}],
        [{**call_item(), "arguments": '{"query":"x","limit":"5"}'}],
        [{**call_item(), "arguments": '{"query":"x","limit":true}'}],
        [{**call_item(), "arguments": '{"query":"x","limit":6}'}],
        [{**call_item(), "arguments": '{"query":"x","limit":5,"extra":1}'}],
        [{**call_item(), "arguments": '{"query":"x"}'}],
        [{**call_item(), "arguments": '{"query":"x","limit":5,"name":"gmail_read"}'}],
        [{**call_item(), "arguments": "not-json"}],
        [{**call_item(), "name": "gmail_read", "arguments": '{"refs":[]}'}],
        [{**call_item(), "name": "gmail_read", "arguments": '{"refs":[1]}'}],
        [{**call_item(), "status": "in_progress"}],
        [{**answer_item(), "status": "incomplete"}],
        [{**answer_item(), "role": "user"}],
        [
            {
                **answer_item(),
                "content": [{"type": "refusal", "refusal": "private-body"}],
            }
        ],
        [{**answer_item(), "content": [{"type": "unknown", "text": "private-body"}]}],
        [{**answer_item(), "content": [{"type": "output_text", "text": "bad-json"}]}],
        [{"type": "web_search_call", "id": "ws_1"}],
    ],
)
async def test_invalid_outcomes_are_terminal_without_repair(
    output: list[dict[str, JSONValue]],
) -> None:
    wire = ProviderWire(envelope(output))
    with pytest.raises(ServiceError) as caught:
        _ = await generate(
            payload(root(), "decide"),
            SecretStr("key"),
            httpx2.MockTransport(wire.handle),
        )
    assert caught.value.code == "dialogue_invalid_response"
    assert len(wire.calls) == 1
    assert "private-body" not in str(caught.value.envelope())


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "incomplete"},
        {"status": "failed"},
        {"id": ""},
        {"error": {"message": "private-body"}},
        {"usage": {"input_tokens": "1", "output_tokens": 2}},
        {"usage": {"input_tokens": -1, "output_tokens": 2}},
    ],
)
async def test_invalid_envelopes_are_sanitized(changes: dict[str, JSONValue]) -> None:
    wire = ProviderWire({**envelope([answer_item()]), **changes})
    with pytest.raises(ServiceError, match="dialogue_invalid_response"):
        _ = await generate(
            payload(root(), "decide"),
            SecretStr("key"),
            httpx2.MockTransport(wire.handle),
        )
    assert len(wire.calls) == 1


class ChunkStream(httpx2.AsyncByteStream):
    pulled: int = 0
    closed: bool = False

    @override
    async def __aiter__(self) -> AsyncIterator[bytes]:
        for _ in range(100):
            self.pulled += 1
            yield b"x" * 8192

    @override
    async def aclose(self) -> None:
        self.closed = True


async def test_oversized_stream_stops_reading_and_closes_without_retry() -> None:
    stream = ChunkStream()
    calls: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        return httpx2.Response(200, stream=stream)

    with pytest.raises(ServiceError, match="dialogue_response_limit"):
        _ = await generate(
            payload(root(), "decide"), SecretStr("key"), httpx2.MockTransport(handle)
        )
    assert len(calls) == 1
    assert stream.pulled == 33
    assert stream.closed


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (302, "provider_error"),
        (401, "provider_auth"),
        (429, "provider_rate_limit"),
        (503, "provider_error"),
    ],
)
async def test_http_failures_are_terminal_and_redirects_are_disabled(
    status: int,
    code: str,
) -> None:
    calls: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        return httpx2.Response(
            status,
            content=b"private-body",
            headers={"Location": "https://example.test/redirect"},
        )

    with pytest.raises(ServiceError) as caught:
        _ = await generate(
            payload(root(), "decide"), SecretStr("key"), httpx2.MockTransport(handle)
        )
    assert caught.value.code == code
    assert len(calls) == 1


async def test_timeout_is_unknown_without_retry() -> None:
    calls: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        message = "private-error"
        raise httpx2.ReadTimeout(message, request=request)

    with pytest.raises(ServiceError, match="provider_unknown"):
        _ = await generate(
            payload(root(), "decide"), SecretStr("key"), httpx2.MockTransport(handle)
        )
    assert len(calls) == 1


@pytest.mark.parametrize("body", [b"not-json", b"\xff", b'{"status":'])
async def test_malformed_response_body_is_sanitized_without_repair(body: bytes) -> None:
    calls: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        return httpx2.Response(200, content=body)

    with pytest.raises(ServiceError, match="dialogue_invalid_response"):
        _ = await generate(
            payload(root(), "decide"), SecretStr("key"), httpx2.MockTransport(handle)
        )
    assert len(calls) == 1


async def test_response_at_exact_256kib_limit_is_accepted() -> None:
    encoded = json.dumps(envelope([answer_item()])).encode()
    body = encoded + b" " * (262_144 - len(encoded))

    def handle(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=body)

    receipt = await generate(
        payload(root(), "decide"), SecretStr("key"), httpx2.MockTransport(handle)
    )
    assert receipt.answer is not None
    assert receipt.answer.reply == "Hello"
