from pathlib import Path
from uuid import uuid4

import anyio
import pytest
from pydantic import ValidationError

from tanya_bot.schemas import MemoryUpdate
from tanya_bot.telegram_models import GatewayFailure, IncomingText
from tanya_bot.telegram_store import CoreError
from tests.test_telegram_core_support import Rig

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize(
    ("sender", "chat", "private"), [(7, 42, True), (42, 7, True), (42, 42, False)]
)
async def test_foreign_or_group_events_do_not_read_state_or_generate(
    tmp_path: Path,
    sender: int,
    chat: int,
    *,
    private: bool,
) -> None:
    rig = Rig(tmp_path / "state")
    await rig.controller.on_text(
        IncomingText(
            update_id="1",
            sender_id=sender,
            chat_id=chat,
            private=private,
            text="secret",
        )
    )
    assert not rig.disk.disk.directory.exists()
    assert rig.gateway.calls == []
    assert rig.outbox.texts == []


async def test_profile_history_same_response_memory_and_exact_forget(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state")
    await rig.say("/settings name Mira")
    await rig.say("/settings language en")
    await rig.say("/settings style direct")
    await rig.say("/settings about Synthetic profile")
    await rig.say("/memory add Keep exact case")
    await rig.say("/memory add Other fact")
    rig.gateway.updates = (
        MemoryUpdate(action="forget", text="keep exact case"),
        MemoryUpdate(action="forget", text="Other fact"),
        MemoryUpdate(action="remember", text=" New fact "),
        MemoryUpdate(action="remember", text="new fact"),
    )
    await rig.say("Other fact correction")
    state = rig.disk.load()
    assert [entry.text for entry in state.memories] == ["Keep exact case", "New fact"]
    request = rig.gateway.calls[0]
    assert request.capture_memory is True
    assert request.companion_name == "Mira"
    assert request.language == "en"
    assert request.profile is not None
    assert request.profile.style == "direct"
    assert request.profile.about == "Synthetic profile"
    assert state.requests[0].response is not None
    assert [entry.text for entry in state.history] == [
        "Other fact correction",
        rig.gateway.reply,
    ]
    rig.gateway.updates = ()
    await rig.say("Next question")
    assert [message.content for message in rig.gateway.calls[1].messages] == [
        "Next question",
    ]
    assert len(rig.gateway.calls) == 2


async def test_unknown_duplicate_restart_and_explicit_cached_recovery(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state")
    rig.gateway.fail = True
    with pytest.raises(GatewayFailure, match="synthetic_unknown"):
        await rig.say("Question")
    pending = rig.disk.load().requests[0]
    await rig.controller.on_text(rig.event("Question"))
    await rig.say("Different question")
    await rig.say("/cancel")
    assert len(rig.gateway.calls) == 1
    resumed = Rig(tmp_path / "state")
    resumed.sequence = 20
    await resumed.say("/recover")
    assert resumed.gateway.calls == []
    assert resumed.gateway.recoveries == [pending.request.request_id]
    assert resumed.disk.load().requests[0].response is None
    resumed.gateway.cached = resumed.gateway.response(pending.request)
    await resumed.say("/recover")
    assert resumed.gateway.calls == []
    assert resumed.outbox.texts.count(resumed.gateway.reply) == 1
    await resumed.say("/recover")
    assert resumed.outbox.texts.count(resumed.gateway.reply) == 1


async def test_task_cancellation_keeps_reservation_and_blocks_duplicate(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state")
    rig.gateway.block = True
    async with anyio.create_task_group() as group:
        _ = group.start_soon(rig.say, "Question")
        await rig.gateway.entered.wait()
        group.cancel_scope.cancel()
    assert rig.disk.load().requests[0].response is None
    await rig.say("Question again")
    assert len(rig.gateway.calls) == 1


async def test_disk_failure_never_publishes_unsaved_completion(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "state")
    rig.disk.fail_completion = True
    with pytest.raises(OSError, match="Synthetic disk failure"):
        await rig.say("Question")
    assert rig.outbox.texts == []
    assert rig.disk.load().requests[0].response is None
    rig.disk.fail_completion = False
    await rig.say("Question again")
    assert len(rig.gateway.calls) == 1


async def test_transport_timeout_keeps_saved_text_without_resending(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state")
    rig.outbox.fail_send = True
    with pytest.raises(TimeoutError):
        await rig.say("Question")
    state = rig.disk.load()
    assert state.requests[0].response is not None
    assert state.replies[0].delivery_attempted
    rig.outbox.fail_send = False
    await rig.say("/recover")
    assert rig.gateway.reply not in rig.outbox.texts
    assert len(rig.gateway.calls) == 1


async def test_external_answer_requires_reservation_and_matching_uuid(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state")
    request = await rig.controller.begin_text(rig.event("Transcript"))
    assert request is not None
    assert rig.gateway.calls == []
    response = rig.gateway.response(request)
    with pytest.raises(CoreError, match="missing_reservation"):
        await rig.controller.complete_text(uuid4(), response)
    with pytest.raises(CoreError, match="invalid_response"):
        await rig.controller.complete_text(
            request.request_id, response.model_copy(update={"request_id": uuid4()})
        )
    await rig.controller.complete_text(request.request_id, response)
    await rig.controller.complete_text(request.request_id, response)
    assert rig.outbox.texts.count(response.reply) == 1


async def test_context_bounds_reject_input_and_trim_old_history(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "state")
    with pytest.raises(ValidationError):
        _ = rig.event("a" * 16_001)
    await rig.say("a" * 16_000)
    assert not rig.gateway.calls
    for _ in range(6):
        await rig.say("x" * 3000)
    assert rig.gateway.calls[-1].messages[-1].content == "x" * 3000
    assert len(rig.gateway.calls[-1].messages) < 11
    assert all(len(text.encode("utf-16-le")) // 2 <= 4096 for text in rig.outbox.texts)


async def test_cancel_tombstone_suppresses_late_answer_until_explicit_recovery(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state")
    request = await rig.controller.begin_text(rig.event("Transcript"))
    assert request is not None
    await rig.controller.cancel_pending()
    await rig.say("/cancel")
    await rig.controller.complete_text(
        request.request_id, rig.gateway.response(request)
    )
    assert rig.disk.load().requests[0].cancelled
    assert rig.disk.load().requests[0].response is None
    assert rig.gateway.reply not in rig.outbox.texts
    rig.gateway.cached = rig.gateway.response(request)
    await rig.say("/recover")
    assert rig.outbox.texts.count(rig.gateway.reply) == 1
    assert rig.gateway.calls == []
