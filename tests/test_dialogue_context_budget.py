import json
from pathlib import Path

import pytest

from tanya_bot.dialogue_models import MailSelection
from tanya_bot.dialogue_provider import payload
from tanya_bot.memory_runtime import MemoryCoordinator
from tanya_bot.telegram_core_context import build_request
from tanya_bot.telegram_models import Entry, TelegramState
from tanya_bot.turns import TurnRequest
from tests.test_dialogue_provider_payload import final_root, input_items, mail_result

pytestmark = pytest.mark.anyio


async def test_runtime_context_reserves_room_for_mail_and_exact_continuation(
    tmp_path: Path,
) -> None:
    history = tuple(Entry(text=f"History {index}: " + "h" * 1900) for index in range(8))
    state = TelegramState(
        owner_id=42,
        history=history,
        history_roles=("user", "assistant") * 4,
    )
    coordinator = MemoryCoordinator(tmp_path / "index")
    coordinator.context_budget = 8000
    prepared, context = await coordinator.prepare(
        state, build_request(state, "Read my recent mail")
    )
    assert sum(len(message.content) for message in prepared.messages) <= 8000
    assert len(context.source_ids) == len(prepared.messages) - 1
    assert context.coverage_gap
    current = await final_root()
    selected = current.model_copy(
        update={
            "turn": TurnRequest(chat=prepared, context=context),
            "tainted": True,
            "selection": MailSelection(
                owner_id=42,
                conversation_id=0,
                root_id=prepared.request_id,
                created_at=1,
                accessed_at=1,
                result=mail_result(),
                answer="Draft " * 333,
            ),
        }
    )
    decision = payload(selected.model_copy(update={"tool_result": None}), "decide")
    final = payload(selected, "final")
    assert len(json.dumps(final, ensure_ascii=False)) <= 32000
    prefix = input_items(decision)
    assert input_items(final)[: len(prefix)] == prefix


async def test_latest_owner_input_yields_no_bytes_to_dialogue_history_budget(
    tmp_path: Path,
) -> None:
    coordinator = MemoryCoordinator(tmp_path / "index")
    coordinator.context_budget = 8000
    state = TelegramState(owner_id=42)
    text = "q" * 15900
    prepared, _ = await coordinator.prepare(state, build_request(state, text))
    assert prepared.messages[-1].content == text
