"""One bounded Responses POST for an already persisted dialogue payload."""

import json
from typing import Annotated, Final, Literal, assert_never

import httpx2
from pydantic import Field, SecretStr, TypeAdapter, ValidationError

from tanya_bot.conversation_memory import JSONValue
from tanya_bot.dialogue_models import (
    MailCall,
    MailRead,
    MailSearch,
    PhaseReceipt,
)
from tanya_bot.dialogue_prompt import (
    payload,
    protect_answer,
)
from tanya_bot.errors import ServiceError
from tanya_bot.provider import ENDPOINT, ProviderModel, Usage, provider_error
from tanya_bot.turns import SCHEMA_VALUE, TurnAnswer

__all__ = ("generate", "payload")

RESPONSE_LIMIT: Final = 256 * 1024


class _Text(ProviderModel):
    type: Literal["output_text"]
    text: str


class _Message(ProviderModel):
    type: Literal["message"]
    id: Annotated[str, Field(min_length=1)]
    status: Literal["completed"]
    role: Literal["assistant"]
    content: Annotated[tuple[_Text, ...], Field(min_length=1)]


class _Summary(ProviderModel):
    type: Literal["summary_text"]
    text: str


class _Reasoning(ProviderModel):
    type: Literal["reasoning"]
    id: Annotated[str, Field(min_length=1)]
    summary: tuple[_Summary, ...]
    encrypted_content: str | None = None
    status: Literal["completed"] | None = None


class _Function(ProviderModel):
    type: Literal["function_call"]
    id: Annotated[str, Field(min_length=1)]
    call_id: Annotated[str, Field(min_length=1, max_length=256)]
    status: Literal["completed"] | None = None
    name: Literal["gmail_search_read", "gmail_read"]
    arguments: str


_ITEM: Final = TypeAdapter(
    Annotated[_Message | _Reasoning | _Function, Field(discriminator="type")]
)


class _Envelope(ProviderModel):
    id: Annotated[str, Field(min_length=1, max_length=256)]
    status: Literal["completed"]
    model: Annotated[str, Field(min_length=1, max_length=128)]
    output: Annotated[tuple[dict[str, JSONValue], ...], Field(max_length=32)]
    usage: Usage
    error: None = None
    incomplete_details: None = None


def _invalid() -> ServiceError:
    return ServiceError(502, "dialogue_invalid_response", "Dialogue result is invalid.")


def _tool(call: _Function, request: dict[str, JSONValue]) -> MailCall:
    args = SCHEMA_VALUE.validate_json(call.arguments, strict=True)
    if "name" in args or request.get("tool_choice") != "auto":
        raise _invalid()
    match call.name:
        case "gmail_search_read":
            action = MailSearch.model_validate_json(call.arguments, strict=True)
        case "gmail_read":
            action = MailRead.model_validate_json(call.arguments, strict=True)
        case _:
            assert_never(call.name)
    return MailCall(call_id=call.call_id, action=action)


def _receipt(body: bytes, request: dict[str, JSONValue]) -> PhaseReceipt:
    parsed = _Envelope.model_validate_json(body, strict=True)
    outcomes: list[TurnAnswer | MailCall] = []
    for raw in parsed.output:
        item = _ITEM.validate_json(json.dumps(raw), strict=True)
        match item:
            case _Reasoning():
                pass
            case _Message(content=content):
                answer = TurnAnswer.model_validate_json(
                    "".join(part.text for part in content), strict=True
                )
                outcomes.append(protect_answer(answer, request))
            case _Function() as call:
                outcomes.append(_tool(call, request))
            case _:
                assert_never(item)
    if len(outcomes) != 1:
        raise _invalid()
    answer: TurnAnswer | None = None
    tool: MailCall | None = None
    outcome = outcomes[0]
    match outcome:
        case TurnAnswer() as answer:
            pass
        case MailCall() as tool:
            pass
        case _:
            assert_never(outcome)
    return PhaseReceipt(
        provider_id=parsed.id,
        model=parsed.model,
        input_tokens=parsed.usage.input_tokens,
        output_tokens=parsed.usage.output_tokens,
        output=parsed.output,
        answer=answer,
        tool=tool,
    )


async def generate(
    payload: dict[str, JSONValue],
    key: SecretStr,
    transport: httpx2.AsyncBaseTransport,
) -> PhaseReceipt:
    """Send once without redirects, retries, fallback, or response repair."""
    try:
        async with (
            httpx2.AsyncClient(
                transport=transport,
                trust_env=False,
                follow_redirects=False,
                timeout=httpx2.Timeout(connect=5, read=45, write=10, pool=5),
            ) as client,
            client.stream(
                "POST",
                ENDPOINT,
                json=payload,
                headers={"Authorization": "Bearer " + key.get_secret_value()},
            ) as response,
        ):
            body = bytearray()
            async for chunk in response.aiter_bytes(chunk_size=8192):
                if len(body) + len(chunk) > RESPONSE_LIMIT:
                    raise ServiceError(
                        502,
                        "dialogue_response_limit",
                        "Dialogue response is too large.",
                    )
                body.extend(chunk)
            if not response.is_success:
                raise provider_error(
                    httpx2.Response(response.status_code, content=bytes(body))
                )
    except httpx2.RequestError:
        raise ServiceError(
            502, "provider_unknown", "Dialogue outcome is unknown."
        ) from None
    try:
        return _receipt(bytes(body), payload)
    except (ValidationError, RecursionError):
        raise _invalid() from None
