import json
from pathlib import Path

import httpx2
import pytest

from tanya_bot.errors import ServiceError
from tanya_bot.schemas import ChatRequest
from tanya_bot.telegram_core_context import build_request
from tests.conftest import Harness
from tests.dialogue_integration_support import attach, final_response, tool_response

pytestmark = pytest.mark.anyio


async def test_mail_search_and_followup_are_source_grounded_and_quarantined(
    harness: Harness,
    tmp_path: Path,
) -> None:
    context = attach(harness, tmp_path)
    harness.wire.outcomes.extend(
        [
            tool_response("gmail_search_read", {"query": "", "limit": 2}),
            final_response(),
        ]
    )
    await context.rig.say("What are my recent emails about?")
    state = context.rig.disk.load()
    first = context.store.load(state.requests[-1].request.request_id)
    assert first is not None
    assert first.tool_result is not None
    assert [item.subject for item in first.tool_result.sources] == [
        "first_mail",
        "second_mail",
    ]
    assert state.memories == ()
    assert state.context.summary.text == ""
    assert state.context.summary.coverage_gap
    assert set(state.history_barrier) == {item.id for item in state.history}
    assert len(harness.wire.calls) == 2
    assert len(context.mail.paths) == 5
    assert state.requests[-1].response is not None
    assert state.requests[-1].response.input_tokens == 30
    ref = first.tool_result.sources[1].ref
    harness.wire.outcomes.extend(
        [
            tool_response("gmail_read", {"refs": [ref]}),
            final_response(
                "The invoice is due Friday. Draft: Thank you. Nothing was sent."
            ),
        ]
    )
    await context.rig.say("Explain the second one and draft a short reply.")
    assert context.mail.paths[-1].endswith("/messages/second_mail")
    assert len(context.mail.paths) == 6
    assert len(harness.wire.calls) == 4
    state = context.rig.disk.load()
    assert state.memories == ()
    assert state.context.summary.text == ""
    next_chat = build_request(state, "Hello again")
    assert [message.content for message in next_chat.messages] == ["Hello again"]
    transmitted = "\n".join(request.content.decode() for request in harness.wire.calls)
    assert "mail-refresh-secret" not in transmitted
    assert "mail-access-secret" not in transmitted
    assert "owner@example.test" not in transmitted
    assert "Invoice due Friday" in transmitted
    await context.rig.controller.on_text(context.rig.event("duplicate"))
    assert len(harness.wire.calls) == 4


async def test_ordinary_turn_uses_one_call_and_no_google(
    harness: Harness,
    tmp_path: Path,
) -> None:
    context = attach(harness, tmp_path)
    harness.wire.outcomes.append(final_response("Hello"))
    await context.rig.say("Hello")
    assert context.mail.paths == []
    assert len(harness.wire.calls) == 1
    assert context.rig.outbox.texts == ["Hello"]


async def test_unknown_decision_never_retries_including_recovery(
    harness: Harness,
    tmp_path: Path,
) -> None:
    context = attach(harness, tmp_path)
    harness.wire.outcomes.append(httpx2.ReadTimeout("synthetic unknown"))
    with pytest.raises(ServiceError, match="provider_unknown"):
        await context.rig.say("What are my emails about?")
    await context.rig.say("/recover")
    assert len(harness.wire.calls) == 1
    assert context.mail.paths == []
    assert context.rig.disk.load().requests[0].response is None


async def test_final_receipt_can_be_assembled_during_cache_only_recovery(
    harness: Harness,
    tmp_path: Path,
) -> None:
    context = attach(harness, tmp_path)
    harness.wire.outcomes.append(final_response("Recovered answer"))
    request = await context.rig.controller.begin_text(context.rig.event("Hello"))
    assert request is not None
    _ = await context.runner.chat(request)
    _ = context.store.mutate(
        request.request_id, lambda root: root.model_copy(update={"final": None})
    )
    context.rig.sequence += 1
    await context.rig.say("/recover")
    assert len(harness.wire.calls) == 1
    assert "Recovered answer" in context.rig.outbox.texts
    assert context.mail.paths == []


async def test_foreign_events_cannot_open_gmail_or_dispatch_a_phase(
    harness: Harness,
    tmp_path: Path,
) -> None:
    context = attach(harness, tmp_path)
    for changed in ({"sender_id": 7}, {"chat_id": 7}, {"private": False}):
        await context.rig.controller.on_text(
            context.rig.event("Read Gmail").model_copy(update=changed)
        )
    assert harness.wire.calls == []
    assert context.mail.paths == []
    assert not context.rig.disk.disk.directory.exists()


async def test_api_address_without_bot_reservation_cannot_generate(
    harness: Harness,
) -> None:
    async with harness.client() as client:
        response = await client.post(
            "/v3/telegram/dialogue/phase",
            content=json.dumps(
                {
                    "root_id": "11111111-1111-4111-8111-111111111111",
                    "kind": "decide",
                }
            ),
        )
    assert response.status_code == 409
    assert harness.wire.calls == []


async def test_v3_first_initialization_preserves_fresh_legacy_api(
    harness: Harness,
    chat_request: ChatRequest,
) -> None:
    async with harness.client() as client:
        rejected = await client.post(
            "/v3/telegram/dialogue/phase",
            content=json.dumps(
                {
                    "root_id": "11111111-1111-4111-8111-111111111111",
                    "kind": "decide",
                }
            ),
        )
        legacy = await client.post("/v1/chat", content=chat_request.model_dump_json())
    assert rejected.status_code == 409
    assert legacy.status_code == 200
    assert len(harness.wire.calls) == 1
