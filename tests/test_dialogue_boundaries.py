import fcntl
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import anyio
import httpx2
import pytest

from tanya_bot.dialogue_models import DialoguePhase, DialogueRoot, PhaseAddress
from tanya_bot.dialogue_sources import prune_sources, recent_selection
from tanya_bot.errors import ServiceError
from tanya_bot.gmail_commands import GmailCommands
from tanya_bot.telegram_core_commands import cancelled_state
from tanya_bot.telegram_gmail import GmailFlow
from tanya_bot.turns import TurnRequest
from tests.conftest import Harness
from tests.dialogue_integration_support import attach, final_response, tool_response

pytestmark = pytest.mark.anyio


async def test_cancel_during_first_preview_stops_remaining_reads_and_final(
    harness: Harness,
    tmp_path: Path,
) -> None:
    context = attach(harness, tmp_path)
    entered = anyio.Event()
    release = threading.Event()
    failures: list[str] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/messages/first_mail"):
            anyio.from_thread.run_sync(entered.set)
            assert release.wait(timeout=5)
        return context.mail.handle(request)

    context.runner.services.account.transport = httpx2.MockTransport(handle)
    harness.wire.outcomes.append(
        tool_response("gmail_search_read", {"query": "", "limit": 2})
    )

    async def invoke() -> None:
        try:
            await context.rig.say("Read my recent mail")
        except ServiceError as error:
            failures.append(error.code)

    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            _ = group.start_soon(invoke)
            await entered.wait()
            await context.rig.controller.cancel_pending()
            release.set()
    assert failures == ["dialogue_cancelled"]
    assert len(harness.wire.calls) == 1
    assert len(context.mail.paths) == 4
    assert context.rig.outbox.texts == []
    assert context.rig.disk.load().memories == ()
    await context.rig.say("/recover")
    assert len(harness.wire.calls) == 1
    assert len(context.mail.paths) == 4


async def test_disconnect_removes_current_selection_without_any_new_mail_reads(
    harness: Harness,
    tmp_path: Path,
) -> None:
    context = attach(harness, tmp_path)
    harness.wire.outcomes.extend(
        [
            tool_response("gmail_search_read", {"query": "", "limit": 2}),
            final_response(),
        ]
    )
    await context.rig.say("Read my recent mail")
    commands = GmailCommands(tmp_path / "gmail.json", tmp_path / "tokens")
    commands.account = context.runner.services.account
    context.rig.controller.gmail = GmailFlow(
        context.rig.disk,
        context.rig.outbox,
        context.rig.controller.lock,
        commands,
        cancel=context.rig.controller.cancel_pending,
    )
    await context.rig.say("/gmail disconnect")
    roots = context.store.roots()
    assert all(root.cancelled for root in roots)
    assert recent_selection(context.store, 42, 0, roots[0].created_at + 1) is None
    assert len(context.mail.paths) == 5
    assert len(harness.wire.calls) == 2
    assert not (tmp_path / "tokens" / "refresh.json").exists()


async def test_source_expiry_keeps_final_and_replay_identity(
    harness: Harness,
    tmp_path: Path,
) -> None:
    context = attach(harness, tmp_path)
    harness.wire.outcomes.extend(
        [
            tool_response("gmail_search_read", {"query": "", "limit": 2}),
            final_response(),
        ]
    )
    await context.rig.say("Read my recent mail")
    request = context.rig.disk.load().requests[-1].request
    root = context.store.load(request.request_id)
    assert root is not None
    prune_sources(context.store, root.created_at + 86401)
    expired = context.store.load(request.request_id)
    assert expired is not None
    assert expired.final == root.final
    assert expired.tool_result is None
    assert expired.selection is None
    assert expired.operations == ()
    assert all(
        phase.payload is None and phase.receipt is None for phase in expired.phases
    )
    assert "remember my password" not in expired.model_dump_json()
    with pytest.raises(ServiceError, match="dialogue_held"):
        context.store.create(root)
    assert context.runner.recover(request) == root.final
    assert len(harness.wire.calls) == 2
    assert len(context.mail.paths) == 5


async def test_invented_reference_never_reads_mail_or_starts_final(
    harness: Harness,
    tmp_path: Path,
) -> None:
    context = attach(harness, tmp_path)
    harness.wire.outcomes.append(
        tool_response("gmail_read", {"refs": ["Maaaaaaaaaaaa"]})
    )
    with pytest.raises(ServiceError, match="mail_reference_invalid"):
        await context.rig.say("Explain the message")
    await context.rig.say("/recover")
    assert len(harness.wire.calls) == 1
    assert context.mail.paths == []


async def test_core_cancel_survives_crash_before_sidecar_fence(
    harness: Harness,
    tmp_path: Path,
) -> None:
    context = attach(harness, tmp_path)
    request = await context.rig.controller.begin_text(
        context.rig.event("Read recent mail")
    )
    assert request is not None
    pending = context.rig.disk.load().requests[-1]
    assert pending.context is not None
    context.store.create(
        DialogueRoot(
            turn=TurnRequest(chat=request, context=pending.context),
            owner_id=42,
            conversation_id=pending.conversation_id,
            update_id=pending.update_id,
            model="offline-model",
            created_at=time.time(),
            phases=(DialoguePhase(kind="decide"),),
        )
    )
    with context.rig.disk.exclusive():
        context.rig.disk.save(cancelled_state(context.rig.disk.load()))
    harness.wire.outcomes.append(final_response())
    async with harness.client() as client:
        response = await client.post(
            "/v3/telegram/dialogue/phase",
            content=PhaseAddress(
                root_id=request.request_id, kind="decide"
            ).model_dump_json(),
        )
    assert response.status_code == 409
    assert harness.wire.calls == []
    assert context.mail.paths == []


async def test_bot_never_acquires_core_while_holding_sidecar(
    harness: Harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = attach(harness, tmp_path)
    original = context.rig.disk.exclusive

    @contextmanager
    def exclusive() -> Generator[None]:
        path = context.store.directory / "request.lock"
        if path.exists():
            with path.open("rb") as probe:
                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
        with original():
            yield

    monkeypatch.setattr(context.rig.disk, "exclusive", exclusive)
    harness.wire.outcomes.extend(
        [
            tool_response("gmail_search_read", {"query": "", "limit": 2}),
            final_response(),
        ]
    )
    await context.rig.say("Read my recent mail")
    assert len(harness.wire.calls) == 2
    assert len(context.mail.paths) == 5
