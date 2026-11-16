"""Bounded chat context and exact same-response memory application."""

from typing import Final, Literal, assert_never
from uuid import uuid4

from tanya_memory import invalidate, update_summary

from tanya_bot.schemas import (
    CONTEXT_CHARACTERS,
    ChatRequest,
    ChatResponse,
    Message,
)
from tanya_bot.telegram_models import (
    ENTRY_CAPACITY,
    Entry,
    PendingChat,
    TelegramState,
)
from tanya_bot.telegram_store import CoreError
from tanya_bot.turns import TurnResponse

HISTORY_CONTEXT: Final = 32
MIN_TERM_LENGTH: Final = 2
MEMORY_RESERVATION: Final = 2000


def context_history(
    state: TelegramState,
) -> tuple[tuple[Entry, Literal["user", "assistant"]], ...]:
    """A forget barrier excludes old dialogue without rewriting source receipts."""
    excluded = set(state.history_barrier)
    return tuple(
        (entry, role)
        for entry, role in zip(state.history, state.history_roles, strict=True)
        if entry.id not in excluded
    )


def build_request(
    state: TelegramState,
    text: str,
    retrieved: tuple[str, ...] | None = None,
    *,
    budget: int = CONTEXT_CHARACTERS,
) -> ChatRequest:
    """Keep latest input intact, rank memories like iOS and trim old history."""
    settings = state.settings
    fixed = (
        len(settings.name) + len(settings.language) + settings.profile.context_size()
    )
    fixed += len(state.context.summary.text) if retrieved is not None else 0
    available = min(budget, CONTEXT_CHARACTERS) - fixed - len(text)
    if available < 0 or not text.strip():
        raise CoreError(code="input_too_long_or_empty")
    terms = {term.casefold() for term in text.split() if len(term) > MIN_TERM_LENGTH}
    ranked = sorted(
        enumerate(state.memories),
        key=lambda pair: (
            sum(term in pair[1].text.casefold() for term in terms),
            pair[0],
        ),
        reverse=True,
    )
    if retrieved is None:
        memories = _fit_memories(
            tuple(entry.text for _, entry in ranked[:8]), available
        )
        selected = _recent_messages(state, available - sum(map(len, memories)))
    else:
        reserved = _pack_indices(retrieved, min(MEMORY_RESERVATION, available))
        selected = _recent_messages(
            state, available - sum(len(retrieved[index]) for index in reserved)
        )
        remaining = available - sum(len(message.content) for message in selected)
        packed = _pack_indices(retrieved, remaining, reserved)
        memories = tuple(retrieved[index] for index in packed)
    return ChatRequest(
        request_id=uuid4(),
        messages=(*selected, Message(role="user", content=text)),
        companion_name=settings.name,
        language=settings.language,
        profile=settings.profile,
        memories=memories,
        capture_memory=True,
    )


def _fit_memories(memories: tuple[str, ...], budget: int) -> tuple[str, ...]:
    selected = list(memories)
    while sum(map(len, selected)) > budget:
        _ = selected.pop()
    return tuple(selected)


def _pack_indices(
    memories: tuple[str, ...], budget: int, reserved: tuple[int, ...] = ()
) -> tuple[int, ...]:
    selected = set(reserved)
    remaining = budget - sum(len(memories[index]) for index in reserved)
    for index, text in enumerate(memories):
        if index not in selected and len(text) <= remaining:
            selected.add(index)
            remaining -= len(text)
    return tuple(sorted(selected))


def _recent_messages(state: TelegramState, budget: int) -> tuple[Message, ...]:
    selected: list[Message] = []
    for entry, role in reversed(context_history(state)):
        if len(selected) >= HISTORY_CONTEXT - 1 or len(entry.text) > budget:
            break
        selected.append(Message(role=role, content=entry.text))
        budget -= len(entry.text)
    return tuple(reversed(selected))


def _updated_memories(
    state: TelegramState,
    pending: PendingChat,
    response: ChatResponse,
) -> tuple[Entry, ...]:
    memories = list(state.memories)
    for update in response.memory_updates or ():
        match update.action:
            case "forget":
                if update.text in pending.request.memories:
                    memories = [
                        entry for entry in memories if entry.text != update.text
                    ]
            case "remember":
                text = update.text.strip()
                if (
                    text
                    and len(memories) < ENTRY_CAPACITY
                    and all(
                        entry.text.casefold() != text.casefold() for entry in memories
                    )
                ):
                    memories.append(Entry(text=text))
            case _:
                assert_never(update.action)
    return tuple(memories)


def apply_response(
    state: TelegramState,
    pending: PendingChat,
    response: ChatResponse,
) -> TelegramState:
    """Forget only exact supplied memory; new facts are bounded and deduplicated."""
    if (
        response.request_id != pending.request.request_id
        or len(response.reply) > CONTEXT_CHARACTERS
    ):
        raise CoreError(code="invalid_response")
    memories = (
        state.memories
        if pending.mail_tainted
        else _updated_memories(state, pending, response)
    )
    user = Entry(text=pending.request.messages[-1].content)
    assistant = Entry(text=response.reply)
    context = state.context
    if pending.context is not None:
        user = user.model_copy(update={"id": pending.context.user_entry_id})
        assistant = assistant.model_copy(
            update={"id": pending.context.assistant_entry_id}
        )
        match response:
            case TurnResponse():
                context = update_summary(context, pending.context, response.summary)
                if pending.context.summary != state.context.summary:
                    context = context.model_copy(
                        update={
                            "summary": context.summary.model_copy(
                                update={
                                    "coverage_gap": True,
                                }
                            ),
                        }
                    )
            case ChatResponse():
                pass
            case _:
                assert_never(response)
    removed = any(
        entry.id not in {item.id for item in memories} for entry in state.memories
    )
    if pending.mail_tainted:
        context = state.context.model_copy(
            update={
                "summary": state.context.summary.model_copy(
                    update={"coverage_gap": True}
                ),
            }
        )
    if tuple(memories) != state.memories:
        changed = invalidate(context)
        context = (
            changed
            if removed
            else changed.model_copy(update={"summary": context.summary})
        )
    history = (*state.history, user, assistant)[-64:]
    return state.model_copy(
        update={
            "memories": tuple(memories),
            "history": history,
            "history_roles": (*state.history_roles, "user", "assistant")[-64:],
            "context": context,
            "history_barrier": tuple(item.id for item in history)
            if removed
            else tuple(
                item.id
                for item in history
                if item.id in state.history_barrier
                or (pending.mail_tainted and item.id in {user.id, assistant.id})
            ),
        }
    )
