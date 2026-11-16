import json

import httpx2
import pytest
from pydantic import SecretStr

from tanya_bot.conversation_memory import JSONValue
from tanya_bot.dialogue_models import (
    DialoguePhase,
    DialogueRoot,
    MailResult,
    MailSelection,
    MailSource,
)
from tanya_bot.dialogue_provider import generate, payload
from tanya_bot.errors import ServiceError
from tanya_bot.schemas import Message
from tanya_bot.turns import SCHEMA_VALUE
from tests.test_dialogue_provider import (
    ProviderWire,
    call_item,
    envelope,
    reasoning_item,
    root,
)

pytestmark = pytest.mark.anyio


def mail_result() -> MailResult:
    return MailResult(
        sources=tuple(
            MailSource(
                ref=f"M{index:012x}",
                message_id=f"private-message-{index}",
                subject="Mail subject",
                sender="External sender",
                date="Monday",
                text="Untrusted mail text. " * 190,
                truncated=False,
            )
            for index in range(5)
        ),
        query="newer_than:7d",
        retrieved_at=1,
    )


async def final_root() -> DialogueRoot:
    current = root()
    wire = ProviderWire(envelope([reasoning_item(), call_item()]))
    receipt = await generate(
        payload(current, "decide"), SecretStr("key"), httpx2.MockTransport(wire.handle)
    )
    return current.model_copy(
        update={
            "tool_result": mail_result(),
            "phases": (DialoguePhase(kind="decide", sent=True, receipt=receipt),),
        }
    )


def input_items(request: dict[str, JSONValue]) -> list[dict[str, JSONValue]]:
    items = request["input"]
    assert isinstance(items, list)
    return [SCHEMA_VALUE.validate_python(item) for item in items]


async def test_final_replays_full_output_and_matching_sanitized_tool_result() -> None:
    current = await final_root()
    request = payload(current, "final")
    items = input_items(request)
    assert items[-3:-1] == [reasoning_item(), call_item()]
    assert items[-1]["type"] == "function_call_output"
    assert items[-1]["call_id"] == "call_1"
    result = items[-1]["output"]
    assert isinstance(result, str)
    projected = SCHEMA_VALUE.validate_json(result)
    assert projected["query"] == "newer_than:7d"
    assert projected["untrusted"] is True
    assert "private-message-" not in json.dumps(request)
    assert "123456789" not in json.dumps(request)
    assert "private-update" not in json.dumps(request)
    assert len(result) <= 16_000
    assert request["tool_choice"] == "none"
    assert "tools" not in request
    assert request["model"] == "configured-luna"
    assert request["store"] is False
    assert request["max_output_tokens"] == 2200
    assert request["reasoning"] == {"effort": "none"}


async def test_final_rejects_any_additional_tool_call() -> None:
    request = payload(await final_root(), "final")
    wire = ProviderWire(envelope([call_item()]))
    with pytest.raises(ServiceError, match="dialogue_invalid_response"):
        _ = await generate(request, SecretStr("key"), httpx2.MockTransport(wire.handle))
    assert len(wire.calls) == 1


async def test_selection_prefix_replays_with_combined_source_budget() -> None:
    current = await final_root()
    selection = MailSelection(
        owner_id=current.owner_id,
        conversation_id=current.conversation_id,
        root_id=current.turn.chat.request_id,
        created_at=1,
        accessed_at=1,
        result=mail_result(),
    )
    selected = current.model_copy(update={"selection": selection})
    decide = payload(selected.model_copy(update={"tool_result": None}), "decide")
    final = payload(selected, "final")
    previous = input_items(decide)
    continued = input_items(final)
    assert continued[: len(previous)] == previous
    source_text = [previous[0]["content"], continued[-1]["output"]]
    assert all(isinstance(value, str) for value in source_text)
    assert sum(len(str(value)) for value in source_text) <= 16_000
    assert "private-message-" not in json.dumps(final)


def test_payload_is_deterministic_and_strict_tools_match_arguments() -> None:
    current = root()
    original = current.model_dump_json()
    request = payload(current, "decide")
    assert request == payload(current, "decide")
    assert current.model_dump_json() == original
    assert request["tool_choice"] == "auto"
    assert request["parallel_tool_calls"] is False
    tools = request["tools"]
    assert isinstance(tools, list)
    for raw in tools:
        tool = SCHEMA_VALUE.validate_python(raw)
        schema = SCHEMA_VALUE.validate_python(tool["parameters"])
        properties = SCHEMA_VALUE.validate_python(schema["properties"])
        assert tool["strict"] is True
        assert schema["additionalProperties"] is False
        assert schema["required"] == list(properties)
        assert "name" not in properties


def test_missing_predecessor_cannot_create_a_final_payload() -> None:
    with pytest.raises(ServiceError, match="dialogue_invalid_phase"):
        _ = payload(root(), "final")


async def test_whole_prompt_budget_rejects_excessive_context() -> None:
    current = await final_root()
    chat = current.turn.chat.model_copy(
        update={
            "messages": (Message(role="user", content="q" * 15_900),),
        }
    )
    current = current.model_copy(
        update={
            "turn": current.turn.model_copy(update={"chat": chat}),
        }
    )
    with pytest.raises(ServiceError, match="dialogue_prompt_limit"):
        _ = payload(current, "final")
