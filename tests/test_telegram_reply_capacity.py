from pathlib import Path

import anyio
import pytest

from tanya_bot.telegram_completion import ReplyCompletion
from tanya_bot.telegram_core_session import CoreSession
from tanya_bot.telegram_models import EVENT_CAPACITY, TELEGRAM_CHUNK, Entry, StoredReply
from tanya_bot.telegram_store import CoreError
from tests.test_memory_runtime import PausedMemory
from tests.test_telegram_core_support import Rig

pytestmark = pytest.mark.anyio


def fill_replies(rig: Rig, count: int) -> None:
    with rig.disk.exclusive():
        state = rig.disk.load()
        rig.disk.save(
            state.model_copy(
                update={
                    "replies": tuple(StoredReply(text="Saved") for _ in range(count))
                }
            )
        )


@pytest.mark.parametrize("text", ["Question", "/memory add Rejected fact"])
async def test_full_replies_reject_before_generation_or_local_effects(
    tmp_path: Path, text: str
) -> None:
    rig = Rig(tmp_path / "state")
    fill_replies(rig, EVENT_CAPACITY)
    before = rig.disk.load()
    with pytest.raises(CoreError, match="reply_capacity"):
        await rig.say(text)
    state = rig.disk.load()
    assert rig.gateway.calls == []
    assert state.requests == before.requests
    assert state.memories == before.memories
    assert state.replies == before.replies
    assert rig.outbox.texts == []


@pytest.mark.parametrize(
    ("cancelled", "archived"), [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("text", ["/memory add Rejected fact", "New question"])
async def test_unresolved_slot_survives_local_actions_and_restart(
    tmp_path: Path, text: str, *, cancelled: bool, archived: bool
) -> None:
    rig = Rig(tmp_path / "state")
    request = await rig.controller.begin_text(rig.event("Reserved question"))
    assert request is not None
    if cancelled:
        await rig.controller.cancel_pending()
    with rig.disk.exclusive():
        state = rig.disk.load()
        rig.disk.save(state.model_copy(update={"conversation_id": int(archived)}))
    fill_replies(rig, EVENT_CAPACITY - 1)
    resumed = Rig(tmp_path / "state")
    resumed.sequence = 10
    with pytest.raises(CoreError, match="reply_capacity"):
        await resumed.say(text)
    assert resumed.disk.load().memories == ()
    with resumed.disk.exclusive():
        session = CoreSession(resumed.disk, resumed.outbox, voice=False)
        await ReplyCompletion(session, None).complete(
            request.request_id, resumed.gateway.response(request), recover=True
        )
    state = resumed.disk.load()
    assert len(state.replies) == EVENT_CAPACITY
    assert state.requests[0].response is not None
    assert resumed.gateway.calls == []
    assert len(resumed.outbox.texts) == (0 if archived else 1)


async def test_last_slot_is_reserved_then_exchanged_once(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "state")
    fill_replies(rig, EVENT_CAPACITY - 1)
    request = await rig.controller.begin_text(rig.event("Last admitted question"))
    assert request is not None
    with pytest.raises(CoreError, match="reply_capacity"):
        await rig.say("/help")
    response = rig.gateway.response(request)
    await rig.controller.complete_text(request.request_id, response)
    await rig.controller.complete_text(request.request_id, response)
    assert len(rig.disk.load().replies) == EVENT_CAPACITY
    assert rig.outbox.texts == [response.reply]


async def test_full_replies_reject_callback_before_settings_mutation(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state")
    await rig.say("/settings")
    before = rig.disk.load()
    menu = before.replies[-1]
    with rig.disk.exclusive():
        rig.disk.save(
            before.model_copy(
                update={
                    "replies": (
                        menu,
                        *(StoredReply(text="Saved") for _ in range(EVENT_CAPACITY - 1)),
                    )
                }
            )
        )
    full = rig.disk.load()
    with pytest.raises(CoreError, match="reply_capacity"):
        await rig.controller.on_callback(rig.callback(menu, "set_name"))
    assert rig.disk.load() == full
    assert rig.gateway.calls == []


@pytest.mark.parametrize("plain", [False, True])
async def test_multichunk_local_output_is_admitted_as_a_whole(
    tmp_path: Path, *, plain: bool
) -> None:
    rig = Rig(tmp_path / "state")
    fill_replies(rig, EVENT_CAPACITY - 1)
    with rig.disk.exclusive():
        state = rig.disk.load()
        rig.disk.save(
            state.model_copy(
                update={"memories": (Entry(text="x" * (TELEGRAM_CHUNK + 1)),)}
            )
        )
    if plain:
        with rig.disk.exclusive():
            session = CoreSession(rig.disk, rig.outbox, voice=False)
            with pytest.raises(CoreError, match="reply_capacity"):
                await session.say_plain("x" * (TELEGRAM_CHUNK + 1))
    else:
        with pytest.raises(CoreError, match="reply_capacity"):
            await rig.say("/memory")
    assert len(rig.disk.load().replies) == EVENT_CAPACITY - 1
    assert rig.outbox.texts == []


async def test_capacity_is_rechecked_after_unlocked_memory_preparation(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state")
    fill_replies(rig, EVENT_CAPACITY - 1)
    memory = PausedMemory(tmp_path / "vectors")
    rig.controller.memory = memory

    async def submit() -> None:
        with pytest.raises(CoreError, match="reply_capacity"):
            await rig.say("Question during preparation")

    with anyio.fail_after(3):
        async with anyio.create_task_group() as tasks:
            _ = tasks.start_soon(submit)
            await memory.prepared.wait()
            await rig.say("/help")
            memory.release.set()
    assert rig.disk.load().requests == ()
    assert len(rig.disk.load().replies) == EVENT_CAPACITY
    assert rig.gateway.calls == []


async def test_append_preserves_the_unresolved_reply_slot(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "state")
    fill_replies(rig, EVENT_CAPACITY - 1)
    request = await rig.controller.begin_text(rig.event("Reserved question"))
    assert request is not None
    with rig.disk.exclusive():
        session = CoreSession(rig.disk, rig.outbox, voice=False)
        with pytest.raises(CoreError, match="reply_capacity"):
            session.append(StoredReply(text="Rejected menu", kind="menu"))
    assert len(rig.disk.load().replies) == EVENT_CAPACITY - 1
    assert rig.outbox.texts == []


async def test_legacy_oversubscription_loads_but_cannot_allocate(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state")
    request = await rig.controller.begin_text(rig.event("Old pending"))
    assert request is not None
    fill_replies(rig, EVENT_CAPACITY)
    resumed = Rig(tmp_path / "state")
    assert resumed.disk.load().requests[0].request == request
    resumed.sequence = 10
    with pytest.raises(CoreError, match="reply_capacity"):
        await resumed.say("/help")
    assert resumed.gateway.calls == []
