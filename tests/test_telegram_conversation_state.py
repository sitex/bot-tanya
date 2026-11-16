from pathlib import Path

import pytest
from pydantic import ValidationError

from tanya_bot.telegram_models import EVENT_CAPACITY, StoredReply, TelegramState
from tanya_bot.telegram_store import CoreError
from tests.test_telegram_core_support import Rig

pytestmark = pytest.mark.anyio


async def test_legacy_snapshot_loads_as_conversation_zero_without_losing_receipts(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state")
    await rig.say("Existing legacy conversation")
    before = rig.disk.load()
    legacy = before.model_dump_json(
        exclude={
            "conversation_id": True,
            "new_confirmation_token": True,
            "settings_menu_token": True,
            "voice_inputs": True,
            "requests": {"__all__": {"conversation_id"}},
            "replies": {"__all__": {"conversation_id", "setting_field"}},
        }
    )
    _ = (rig.disk.disk.directory / "state.json").write_text(legacy)
    resumed = Rig(tmp_path / "state")
    restored = resumed.disk.load()
    assert restored == before
    assert restored.conversation_id == 0
    assert (
        restored.requests[0].request.request_id == before.requests[0].request.request_id
    )
    assert restored.replies[0].token == before.replies[0].token


async def test_full_event_ledger_fails_closed_instead_of_resetting_for_new(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state")
    state = TelegramState(
        owner_id=42, seen_updates=tuple(str(index) for index in range(EVENT_CAPACITY))
    )
    with rig.disk.exclusive():
        rig.disk.save(state)
    with pytest.raises(CoreError, match="update_capacity"):
        await rig.say("/new")
    assert rig.disk.load() == state
    assert rig.gateway.calls == []


async def test_failed_confirmation_save_keeps_original_context_and_token(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path / "state")
    await rig.say("Existing conversation")
    await rig.say("/new")
    before = rig.disk.load()
    menu = before.replies[-1]
    rig.disk.fail_all = True
    with pytest.raises(OSError, match="Synthetic disk failure"):
        await rig.controller.on_callback(rig.callback(menu, "new_confirm"))
    assert rig.disk.load() == before
    rig.disk.fail_all = False
    await rig.controller.on_callback(rig.callback(menu, "new_confirm"))
    assert rig.disk.load().conversation_id == 1
    assert len(rig.gateway.calls) == 1


@pytest.mark.parametrize("field", ["new_confirmation_token", "settings_menu_token"])
async def test_orphaned_confirmation_or_settings_token_is_corrupt(field: str) -> None:
    state = TelegramState(owner_id=42).model_copy(update={field: "missing"})
    with pytest.raises(ValidationError, match="invalid_state"):
        _ = TelegramState.model_validate_json(state.model_dump_json())


async def test_future_conversation_reply_is_corrupt() -> None:
    with pytest.raises(ValidationError, match="invalid_state"):
        _ = TelegramState(
            owner_id=42, replies=(StoredReply(text="Future", conversation_id=1),)
        )
