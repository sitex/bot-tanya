from tanya_bot.schemas import CONTEXT_CHARACTERS, MemoryUpdate
from tanya_bot.telegram_core_context import apply_response, build_request
from tanya_bot.telegram_models import Entry, PendingChat, TelegramState
from tests.test_enhanced_turns import answer


def remaining(state: TelegramState, text: str) -> int:
    return (
        CONTEXT_CHARACTERS
        - len(state.settings.name)
        - len(state.settings.language)
        - state.settings.profile.context_size()
        - len(state.context.summary.text)
        - len(text)
    )


def test_enhanced_memory_reservation_keeps_short_fact_and_recent_history() -> None:
    text = "Latest question"
    state = TelegramState(owner_id=42)
    available = remaining(state, text)
    recent = Entry(text="recent" * 100)
    state = state.model_copy(
        update={
            "history": (Entry(text="o" * (available - len(recent.text))), recent),
            "history_roles": ("user", "assistant"),
        }
    )
    fact = "Exact relevant fact"
    request = build_request(state, text, (fact,))
    assert request.memories == (fact,)
    assert [item.content for item in request.messages] == [recent.text, text]
    assert sum(len(item.content) for item in request.messages) + sum(
        map(len, request.memories)
    ) <= remaining(state, "")


def test_enhanced_packing_skips_oversized_fact_and_preserves_forget_bytes() -> None:
    text = "Question"
    fact = "  Keep Exact Case and whitespace  "
    state = TelegramState(owner_id=42)
    oversized = "x" * (remaining(state, text) + 1)
    state = state.model_copy(
        update={"memories": (Entry(text=oversized), Entry(text=fact))}
    )
    request = build_request(state, text, (oversized, fact))
    assert request.memories == (fact,)
    assert state.memories[1].text == fact
    pending = PendingChat(update_id="1", request=request)
    wrong = answer(request).model_copy(
        update={"memory_updates": (MemoryUpdate(action="forget", text=fact.strip()),)}
    )
    assert apply_response(state, pending, wrong).memories == state.memories
    exact = wrong.model_copy(
        update={"memory_updates": (MemoryUpdate(action="forget", text=fact),)}
    )
    assert apply_response(state, pending, exact).memories == (state.memories[0],)


def test_enhanced_spare_space_keeps_large_fact_and_rank_order() -> None:
    state = TelegramState(
        owner_id=42,
        history=(Entry(text="Recent exchange"),),
        history_roles=("assistant",),
    )
    facts = ("L" * 9000, "short fact")
    request = build_request(state, "Question", facts)
    assert request.memories == facts
    assert [item.content for item in request.messages] == [
        "Recent exchange",
        "Question",
    ]
    assert sum(len(item.content) for item in request.messages) + sum(
        map(len, request.memories)
    ) <= remaining(state, "")


def test_legacy_memory_prefix_policy_is_unchanged() -> None:
    state = TelegramState(owner_id=42)
    state = state.model_copy(
        update={
            "memories": (
                Entry(text="short fact"),
                Entry(text="x" * (remaining(state, "Question") + 1)),
            )
        }
    )
    assert build_request(state, "Question").memories == ()
