from pathlib import Path
from uuid import uuid4

import pytest
from tanya_memory import TurnContext

from tanya_bot.conversation_memory import JSONValue
from tanya_bot.dialogue_models import DialoguePhase, DialogueRoot, PhaseKind
from tanya_bot.dialogue_store import DialogueStore
from tanya_bot.errors import ServiceError
from tanya_bot.schemas import ChatRequest
from tanya_bot.turns import TurnRequest


@pytest.fixture
def root(chat_request: ChatRequest) -> DialogueRoot:
    return DialogueRoot(
        turn=TurnRequest(
            chat=chat_request,
            context=TurnContext(
                epoch=1,
                memory_revision=0,
                user_entry_id=uuid4(),
                assistant_entry_id=uuid4(),
            ),
        ),
        owner_id=17,
        conversation_id=3,
        update_id="telegram:123",
        model="offline-model",
        created_at=100.0,
        phases=(DialoguePhase(kind="decide"), DialoguePhase(kind="final")),
    )


@pytest.mark.parametrize("kind", ["decide", "final"])
def test_phase_claim_survives_restart_and_cannot_be_replayed(
    tmp_path: Path, root: DialogueRoot, kind: PhaseKind
) -> None:
    store = DialogueStore(tmp_path / "dialogue")
    store.create(root)
    root_id = root.turn.chat.request_id
    payload: dict[str, JSONValue] = {"input": ["frozen turn"], "model": root.model}

    claimed = store.claim(root_id, kind, payload)
    restarted = DialogueStore(store.directory)
    with pytest.raises(ServiceError, match=r"^dialogue_held$"):
        _ = restarted.claim(root_id, kind, {"input": "replacement"})

    assert restarted.load(root_id) == claimed
    phase = next(item for item in claimed.phases if item.kind == kind)
    assert phase.sent
    assert phase.payload == payload


@pytest.mark.parametrize("changed", [False, True])
def test_duplicate_root_never_replaces_unsent_authority(
    tmp_path: Path, root: DialogueRoot, *, changed: bool
) -> None:
    store = DialogueStore(tmp_path / "dialogue")
    store.create(root)
    duplicate = root.model_copy(update={"model": "other-model"}) if changed else root

    with pytest.raises(ServiceError, match=r"^dialogue_held$"):
        DialogueStore(store.directory).create(duplicate)

    assert store.load(root.turn.chat.request_id) == root


@pytest.mark.parametrize("kind", ["decide", "final"])
def test_cancelled_root_blocks_even_an_unsent_authorized_phase(
    tmp_path: Path, root: DialogueRoot, kind: PhaseKind
) -> None:
    root = root.model_copy(update={"cancelled": True})
    store = DialogueStore(tmp_path / "dialogue")
    store.create(root)

    with pytest.raises(ServiceError, match=r"^dialogue_cancelled$"):
        _ = store.claim(root.turn.chat.request_id, kind, {"input": "mail"})

    assert store.load(root.turn.chat.request_id) == root


def test_missing_phase_is_not_reauthorized_on_restart(
    tmp_path: Path, root: DialogueRoot
) -> None:
    root = root.model_copy(update={"phases": (DialoguePhase(kind="decide"),)})
    store = DialogueStore(tmp_path / "dialogue")
    store.create(root)
    restarted = DialogueStore(store.directory)

    with pytest.raises(ServiceError, match=r"^phase_missing$"):
        _ = restarted.claim(root.turn.chat.request_id, "final", {})

    assert restarted.load(root.turn.chat.request_id) == root


def test_missing_database_keeps_existing_root_held(
    tmp_path: Path, root: DialogueRoot
) -> None:
    store = DialogueStore(tmp_path / "dialogue")
    store.create(root)
    database = store.directory / "receipts.sqlite3"
    database.unlink()
    restarted = DialogueStore(store.directory)

    with pytest.raises(ServiceError, match=r"^storage_unavailable$"):
        restarted.create(root)
    with pytest.raises(ServiceError, match=r"^storage_unavailable$"):
        _ = restarted.claim(root.turn.chat.request_id, "decide", {})

    assert not database.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("owner_id", 18),
        ("conversation_id", 4),
        ("update_id", "telegram:124"),
        ("model", "replacement-model"),
        ("created_at", 101.0),
    ],
)
def test_root_identity_fields_cannot_change(
    tmp_path: Path, root: DialogueRoot, field: str, value: JSONValue
) -> None:
    store = DialogueStore(tmp_path / "dialogue")
    store.create(root)

    with pytest.raises(ServiceError, match=r"^dialogue_conflict$"):
        _ = store.mutate(
            root.turn.chat.request_id,
            lambda saved: saved.model_copy(update={field: value}),
        )

    assert store.load(root.turn.chat.request_id) == root


def test_exact_turn_context_cannot_be_replaced(
    tmp_path: Path, root: DialogueRoot
) -> None:
    store = DialogueStore(tmp_path / "dialogue")
    store.create(root)
    changed = root.turn.model_copy(
        update={"context": root.turn.context.model_copy(update={"epoch": 2})}
    )

    with pytest.raises(ServiceError, match=r"^dialogue_conflict$"):
        _ = store.mutate(
            root.turn.chat.request_id,
            lambda saved: saved.model_copy(update={"turn": changed}),
        )

    assert store.load(root.turn.chat.request_id) == root
