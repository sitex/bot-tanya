from pathlib import Path
from uuid import UUID, uuid4

import pytest

from tanya_bot.telegram_models import (
    EVENT_CAPACITY,
    IncomingText,
    IncomingVoice,
    VoiceInput,
)
from tanya_bot.telegram_store import CoreError
from tests.test_telegram_core_support import Rig

pytestmark = pytest.mark.anyio


def incoming() -> IncomingVoice:
    return IncomingVoice(update_id="42:100", sender_id=42, chat_id=42, private=True)


async def test_foreign_media_cannot_reserve_or_open_state(tmp_path: Path) -> None:
    rig = Rig(tmp_path / "core")
    event = incoming().model_copy(update={"sender_id": 7})
    assert await rig.controller.reserve_voice_input(event, uuid4()) is None
    assert not rig.disk.disk.directory.exists()


async def test_voice_reservation_replays_read_only_after_restart(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "core")
    request_id = uuid4()
    first = await rig.controller.reserve_voice_input(incoming(), request_id)
    assert first is not None
    reserved, fresh = first
    assert fresh
    resumed = Rig(tmp_path / "core")
    duplicate = await resumed.controller.reserve_voice_input(incoming(), request_id)
    assert duplicate == (reserved, False)
    assert resumed.gateway.calls == []
    with pytest.raises(CoreError, match="voice_input_conflict"):
        _ = await resumed.controller.reserve_voice_input(incoming(), uuid4())


async def test_cancelled_input_never_becomes_a_late_message_after_restart(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "core")
    request_id = uuid4()
    first = await rig.controller.reserve_voice_input(incoming(), request_id)
    assert first is not None
    await rig.say("/cancel")
    resumed = Rig(tmp_path / "core")
    assert await resumed.controller.reserve_voice_input(incoming(), request_id) is None
    await resumed.controller.on_text(
        IncomingText(
            update_id=incoming().update_id,
            conversation_id=first[0].conversation_id,
            sender_id=42,
            chat_id=42,
            private=True,
            text="Late cached transcript",
        )
    )
    assert resumed.gateway.calls == []
    assert resumed.disk.load().history == ()


async def test_old_input_is_rejected_after_new_chat_but_fresh_audio_can_reserve(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "core")
    request_id = uuid4()
    old = await rig.controller.reserve_voice_input(incoming(), request_id)
    assert old is not None
    await rig.say("/new")
    menu = rig.disk.load().replies[-1]
    await rig.controller.on_callback(rig.callback(menu, "new_confirm"))
    assert await rig.controller.reserve_voice_input(incoming(), request_id) is None
    await rig.controller.on_text(
        IncomingText(
            update_id=incoming().update_id,
            conversation_id=0,
            sender_id=42,
            chat_id=42,
            private=True,
            text="Old transcript",
        )
    )
    event = incoming().model_copy(update={"update_id": "42:101"})
    fresh = await rig.controller.reserve_voice_input(event, uuid4())
    assert fresh is not None
    assert fresh[0].conversation_id == 1
    assert fresh[1]
    assert rig.gateway.calls == []


async def test_completed_transcript_duplicate_does_not_generate_again(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "core")
    request_id = uuid4()
    reserved = await rig.controller.reserve_voice_input(incoming(), request_id)
    assert reserved is not None
    await rig.controller.on_text(
        IncomingText(
            update_id=incoming().update_id,
            conversation_id=reserved[0].conversation_id,
            sender_id=42,
            chat_id=42,
            private=True,
            text="Current transcript",
        )
    )
    assert await rig.controller.reserve_voice_input(incoming(), request_id) is None
    assert len(rig.gateway.calls) == 1


async def test_voice_input_capacity_keeps_all_existing_identities(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "core")
    saved = tuple(
        VoiceInput(
            update_id=str(index), request_id=UUID(int=index + 1), conversation_id=0
        )
        for index in range(EVENT_CAPACITY)
    )
    with rig.disk.exclusive():
        rig.disk.save(rig.disk.load().model_copy(update={"voice_inputs": saved}))
    with pytest.raises(CoreError, match="voice_input_capacity"):
        _ = await rig.controller.reserve_voice_input(incoming(), uuid4())
    assert rig.disk.load().voice_inputs == saved
    assert rig.gateway.calls == []
