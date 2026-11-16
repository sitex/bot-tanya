from pathlib import Path

import pytest
from tanya_gmail import MessagePreview

from tanya_bot.dialogue_models import PhaseAddress, PhaseReceipt
from tanya_bot.errors import ServiceError
from tests.conftest import Harness
from tests.dialogue_integration_support import attach, final_response, tool_response

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize("during_read", [False, True])
async def test_reference_deadline_is_rechecked_after_decision_and_each_read(
    harness: Harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    during_read: bool,
) -> None:
    context = attach(harness, tmp_path)
    harness.wire.outcomes.extend(
        [
            tool_response("gmail_search_read", {"query": "", "limit": 2}),
            final_response(),
        ]
    )
    await context.rig.say("Read my recent mail")
    root = context.store.roots()[0]
    assert root.selection is not None
    deadline = root.selection.created_at + 7201
    refs = [item.ref for item in root.selection.result.sources]
    original_phase = context.runner.services.gateway.phase
    original_read = context.runner.services.account.read

    async def phase(address: PhaseAddress) -> PhaseReceipt:
        result = await original_phase(address)
        if not during_read:
            monkeypatch.setattr("tanya_bot.dialogue_tools.time.time", lambda: deadline)
        return result

    async def read(message_id: str, limit: int) -> MessagePreview:
        result = await original_read(message_id, limit)
        if during_read:
            monkeypatch.setattr("tanya_bot.dialogue_tools.time.time", lambda: deadline)
        return result

    monkeypatch.setattr(context.runner.services.gateway, "phase", phase)
    monkeypatch.setattr(context.runner.services.account, "read", read)
    harness.wire.outcomes.append(tool_response("gmail_read", {"refs": refs}))
    with pytest.raises(ServiceError, match="mail_reference_expired"):
        await context.rig.say("Explain the selected messages")
    assert len(harness.wire.calls) == 3
    assert len(context.mail.paths) == (6 if during_read else 5)
    await context.rig.say("/recover")
    assert len(harness.wire.calls) == 3
    assert len(context.mail.paths) == (6 if during_read else 5)
