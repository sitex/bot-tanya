import json
from dataclasses import dataclass, field
from uuid import uuid4

import httpx2
import pytest
from pydantic import SecretStr
from tanya_memory import DerivedSummary, TurnContext
from tanya_persona import DeliveryPlan

from tanya_bot.conversation_memory import JSONValue
from tanya_bot.dialogue_models import DialogueRoot, MailSearch
from tanya_bot.dialogue_provider import generate, payload
from tanya_bot.schemas import ChatRequest, MemoryUpdate, Message
from tanya_bot.turns import TurnAnswer, TurnRequest

pytestmark = pytest.mark.anyio


def root() -> DialogueRoot:
    return DialogueRoot(
        turn=TurnRequest(
            chat=ChatRequest(
                request_id=uuid4(),
                language="en",
                messages=(Message(role="user", content="Read recent mail"),),
            ),
            context=TurnContext(
                epoch=0,
                memory_revision=0,
                user_entry_id=uuid4(),
                assistant_entry_id=uuid4(),
                summary=DerivedSummary(text="Clean existing context"),
            ),
        ),
        owner_id=123456789,
        conversation_id=0,
        update_id="private-update",
        model="configured-luna",
        created_at=1,
    )


def answer_item() -> dict[str, JSONValue]:
    answer = TurnAnswer(
        reply="Hello", summary="New summary", memory_updates=(), delivery=DeliveryPlan()
    )
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": answer.model_dump_json()}],
    }


def call_item() -> dict[str, JSONValue]:
    return {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "gmail_search_read",
        "status": "completed",
        "arguments": '{"query":"newer_than:7d","limit":5}',
    }


def reasoning_item() -> dict[str, JSONValue]:
    return {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [],
        "encrypted_content": "opaque-encrypted-reasoning",
    }


def envelope(output: list[dict[str, JSONValue]]) -> dict[str, JSONValue]:
    return {
        "id": "resp_1",
        "status": "completed",
        "model": "provider-snapshot",
        "output": list(output),
        "usage": {"input_tokens": 31, "output_tokens": 12},
    }


@dataclass(frozen=True, slots=True)
class ProviderWire:
    body: dict[str, JSONValue]
    calls: list[httpx2.Request] = field(default_factory=list)

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.calls.append(request)
        return httpx2.Response(200, json=self.body)


async def test_answer_receipt_retains_metadata_raw_items_and_exact_payload() -> None:
    request = payload(root(), "decide")
    raw = [reasoning_item(), answer_item()]
    wire = ProviderWire(envelope(raw))
    receipt = await generate(
        request, SecretStr("synthetic-key"), httpx2.MockTransport(wire.handle)
    )
    assert receipt.answer is not None
    assert receipt.answer.reply == "Hello"
    assert receipt.tool is None
    assert receipt.output == tuple(raw)
    assert (receipt.provider_id, receipt.model) == ("resp_1", "provider-snapshot")
    assert (receipt.input_tokens, receipt.output_tokens) == (31, 12)
    assert len(wire.calls) == 1
    assert wire.calls[0].method == "POST"
    assert json.loads(wire.calls[0].content) == request
    assert wire.calls[0].headers["authorization"] == "Bearer synthetic-key"


async def test_search_call_is_strictly_parsed_without_answer() -> None:
    wire = ProviderWire(envelope([reasoning_item(), call_item()]))
    receipt = await generate(
        payload(root(), "decide"), SecretStr("key"), httpx2.MockTransport(wire.handle)
    )
    assert receipt.answer is None
    assert receipt.tool is not None
    assert receipt.tool.call_id == "call_1"
    assert receipt.tool.action == MailSearch(query="newer_than:7d", limit=5)


async def test_read_call_accepts_json_arrays_as_bounded_immutable_refs() -> None:
    call = {
        **call_item(),
        "name": "gmail_read",
        "arguments": '{"refs":["M123456789abc"]}',
    }
    wire = ProviderWire(envelope([call]))
    receipt = await generate(
        payload(root(), "decide"), SecretStr("key"), httpx2.MockTransport(wire.handle)
    )
    assert receipt.tool is not None
    assert receipt.tool.action.name == "gmail_read"
    assert receipt.tool.action.model_dump(mode="json")["refs"] == ["M123456789abc"]


async def test_tainted_answer_cannot_promote_mail_memory_or_replace_clean_summary() -> (
    None
):
    current = root().model_copy(update={"tainted": True})
    malicious = TurnAnswer(
        reply="[M123456789abc] says to remember this",
        summary="Injected summary",
        memory_updates=(MemoryUpdate(action="remember", text="Mail claim"),),
        delivery=DeliveryPlan(),
    )
    item: dict[str, JSONValue] = {
        **answer_item(),
        "content": [{"type": "output_text", "text": malicious.model_dump_json()}],
    }
    wire = ProviderWire(envelope([item]))
    receipt = await generate(
        payload(current, "decide"), SecretStr("key"), httpx2.MockTransport(wire.handle)
    )
    assert receipt.answer is not None
    assert receipt.answer.summary == "Clean existing context"
    assert receipt.answer.memory_updates == ()
