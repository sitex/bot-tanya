from datetime import date
from pathlib import Path
from typing import Literal, assert_never

import pytest
from tanya_memory import DerivedSummary
from tanya_persona import DeliveryPlan

from tanya_bot.memory_runtime import MemoryCoordinator
from tanya_bot.schemas import ChatRequest, MemoryUpdate
from tanya_bot.telegram_models import Entry
from tanya_bot.turns import TurnResponse
from tests.test_telegram_core_support import Rig

pytestmark = pytest.mark.anyio


def enhanced(rig: Rig, path: Path) -> None:
    rig.controller.memory = MemoryCoordinator(path / "vectors")


def answer(request: ChatRequest) -> TurnResponse:
    return TurnResponse(
        request_id=request.request_id,
        reply="Saved.",
        summary="Discussing a trip",
        delivery=DeliveryPlan(),
        model="offline",
        input_tokens=2,
        output_tokens=3,
        memory_updates=(),
    )


async def test_enhanced_restart_preserves_summary_sources_date_and_voice_plan(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state", voice=True)
    enhanced(rig, tmp_path)
    await rig.say("/settings voice_replies on")
    rig.sequence += 1
    request = await rig.controller.begin_text(rig.event("Plan a trip"))
    assert request is not None
    response = answer(request)
    await rig.controller.complete_text(request.request_id, response)
    await rig.controller.maintain_memory(date(2026, 11, 16))

    resumed = Rig(tmp_path / "state", voice=True)
    state = resumed.disk.load()
    pending = state.requests[-1]
    assert pending.context is not None
    assert state.context.summary.text == response.summary
    assert state.context.summary.source_ids == (
        pending.context.user_entry_id,
        pending.context.assistant_entry_id,
    )
    assert (
        tuple(entry.id for entry in state.history) == state.context.summary.source_ids
    )
    assert state.context.maintenance_day == date(2026, 11, 16)
    assert pending.response == response
    assert rig.voice.calls[0].delivery == response.delivery
    assert rig.outbox.texts[-1] == "Saved."
    resumed.sequence = 20
    await resumed.say("/new")
    menu = resumed.disk.load().replies[-1]
    await resumed.controller.on_callback(resumed.callback(menu, "new_confirm"))
    new = resumed.disk.load()
    assert new.conversation_id == 1
    assert new.context.summary.text == ""
    assert new.history == ()
    assert new.context.maintenance_day == date(2026, 11, 16)


@pytest.mark.parametrize("barrier", ["edit", "cancel", "archive"])
async def test_late_enhanced_results_never_restore_derived_state(
    tmp_path: Path,
    barrier: Literal["edit", "cancel", "archive"],
) -> None:
    rig = Rig(tmp_path / "state", voice=True)
    enhanced(rig, tmp_path)
    request = await rig.controller.begin_text(rig.event("Old question"))
    assert request is not None
    response = answer(request).model_copy(
        update={
            "reply": "Late answer",
            "memory_updates": (MemoryUpdate(action="remember", text="Late fact"),),
        }
    )
    match barrier:
        case "edit":
            await rig.say("/memory add Manual fact")
        case "cancel":
            await rig.controller.cancel_pending()
        case "archive":
            await rig.say("/new")
            menu = rig.disk.load().replies[-1]
            await rig.controller.on_callback(rig.callback(menu, "new_confirm"))
        case _:
            assert_never(barrier)
    await rig.controller.complete_text(request.request_id, response)
    rig.gateway.cached = response
    await rig.say("/recover")
    state = rig.disk.load()
    assert state.context.summary.text == ""
    assert "Late fact" not in [entry.text for entry in state.memories]
    assert state.history == ()
    assert all(reply.delivery is None for reply in state.replies)
    assert rig.voice.calls == []
    assert state.requests[0].response == response
    assert rig.outbox.texts.count(response.reply) == (1 if barrier == "cancel" else 0)


async def test_forget_barrier_excludes_old_summary_history_and_cached_sources(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state")
    enhanced(rig, tmp_path)
    await rig.say("/memory add Private old fact")
    rig.sequence += 1
    first = await rig.controller.begin_text(rig.event("Private old fact"))
    assert first is not None
    await rig.controller.complete_text(first.request_id, answer(first))
    identifier = rig.disk.load().memories[0].id
    await rig.say(f"/memory delete {identifier}")
    rig.sequence += 1
    next_request = await rig.controller.begin_text(rig.event("Next question"))
    assert next_request is not None
    assert next_request.memories == ()
    assert [message.content for message in next_request.messages] == ["Next question"]
    state = rig.disk.load()
    assert state.requests[-1].context is not None
    assert state.requests[-1].context.summary.text == ""
    assert state.requests[0].response is not None


async def test_current_input_survives_summary_budget_pressure(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "state")
    enhanced(rig, tmp_path)
    with rig.disk.exclusive():
        state = rig.disk.load()
        original = DerivedSummary(text="s" * 1800)
        rig.disk.save(
            state.model_copy(
                update={
                    "context": state.context.model_copy(update={"summary": original}),
                }
            )
        )
    fixed = len(state.settings.name) + len(state.settings.language)
    fixed += state.settings.profile.context_size()
    text = "q" * (16_000 - fixed - 10)
    request = await rig.controller.begin_text(rig.event(text))
    assert request is not None
    context = rig.disk.load().requests[-1].context
    assert context is not None
    assert request.messages[-1].content == text
    assert len(context.summary.text) == 10
    assert fixed + len(text) + len(context.summary.text) == 16_000
    await rig.controller.complete_text(request.request_id, answer(request))
    saved = rig.disk.load().context.summary
    assert saved.text == original.text
    assert saved.coverage_gap


async def test_retrieved_memories_cannot_crowd_out_recent_history(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state")
    enhanced(rig, tmp_path)
    with rig.disk.exclusive():
        state = rig.disk.load()
        recent = (Entry(text="u" * 3000), Entry(text="a" * 3000))
        rig.disk.save(
            state.model_copy(
                update={
                    "history": recent,
                    "history_roles": ("user", "assistant"),
                    "memories": (Entry(text="m" * 9000),),
                }
            )
        )
    request = await rig.controller.begin_text(rig.event("q" * 1000))
    assert request is not None
    assert [message.content for message in request.messages] == [
        recent[0].text,
        recent[1].text,
        "q" * 1000,
    ]
    assert request.memories == ()
    context = rig.disk.load().requests[-1].context
    assert context is not None
    assert context.source_ids == tuple(entry.id for entry in recent)
