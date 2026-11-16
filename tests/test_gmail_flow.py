from pathlib import Path

import anyio
import pytest

from tanya_bot.gmail_commands import GmailCommands, GmailOutput
from tanya_bot.telegram_gmail import GmailFlow
from tests.test_telegram_core_support import Rig

pytestmark = pytest.mark.anyio


def attach(rig: Rig, path: Path) -> GmailCommands:
    commands = GmailCommands(path / "missing.json", path / "tokens")
    rig.controller.gmail = GmailFlow(
        rig.disk,
        rig.outbox,
        rig.controller.lock,
        commands,
    )
    return commands


async def test_gmail_owner_gate_duplicate_claim_and_plain_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = Rig(tmp_path / "state")
    commands = attach(rig, tmp_path)
    calls: list[str] = []

    async def handle(text: str, language: str) -> GmailOutput:
        calls.append(text)
        assert language == "ru"
        assert not rig.controller.lock.locked()
        with rig.disk.exclusive():
            assert f"text:{rig.sequence}" in rig.disk.load().seen_updates
        return GmailOutput("Saved.", ("gmail_status", "gmail_connect"))

    monkeypatch.setattr(commands, "handle", handle)
    for changed in ({"sender_id": 7}, {"chat_id": 7}, {"private": False}):
        await rig.controller.on_text(
            rig.event("/gmail list Saved.").model_copy(
                update=changed,
            )
        )
    assert not rig.disk.disk.directory.exists()
    await rig.say("/settings language ru")
    await rig.say("/gmail list Saved.")
    await rig.controller.on_text(rig.event("/gmail list Saved."))
    assert calls == ["/gmail list Saved."]
    assert rig.outbox.texts[-1] == "Saved."
    reply = rig.disk.load().replies[-1]
    assert reply.message_id is not None
    assert [
        button.text for row in rig.outbox.keyboards[reply.message_id] for button in row
    ] == ["Состояние Gmail", "Подключить Gmail"]
    assert rig.gateway.calls == []


@pytest.mark.parametrize(
    ("action", "command"),
    [
        ("gmail_status", "/gmail status"),
        ("gmail_connect", "/gmail connect"),
        ("gmail_list", "/gmail list"),
        ("gmail_disconnect", "/gmail disconnect"),
    ],
)
async def test_gmail_callbacks_require_saved_message_and_single_update_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    command: str,
) -> None:
    rig = Rig(tmp_path / "state")
    commands = attach(rig, tmp_path)
    calls: list[str] = []

    async def handle(text: str, language: str) -> GmailOutput:
        calls.append(text)
        assert language == "en"
        return GmailOutput(
            "Mail menu",
            (
                "gmail_status",
                "gmail_connect",
                "gmail_list",
                "gmail_disconnect",
            ),
        )

    monkeypatch.setattr(commands, "handle", handle)
    await rig.say("/gmail")
    menu = rig.disk.load().replies[-1]
    bad = rig.callback(menu, action).model_copy(update={"message_id": 999})
    await rig.controller.on_callback(bad)
    forged = rig.callback(menu, "gmail_read_privateID")
    await rig.controller.on_callback(forged)
    callback = rig.callback(menu, action)
    await rig.controller.on_callback(callback)
    await rig.controller.on_callback(callback)
    assert calls == ["/gmail", command]
    await rig.say("/new")
    confirmation = rig.disk.load().replies[-1]
    await rig.controller.on_callback(rig.callback(confirmation, "new_confirm"))
    await rig.controller.on_callback(rig.callback(menu, action))
    assert calls == ["/gmail", command]


async def test_new_conversation_can_finish_during_gmail_and_drops_late_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = Rig(tmp_path / "state")
    commands = attach(rig, tmp_path)
    entered, release = anyio.Event(), anyio.Event()

    async def handle(text: str, language: str) -> GmailOutput:
        assert text == "/gmail read syntheticID"
        assert language == "en"
        entered.set()
        await release.wait()
        return GmailOutput("Untrusted old mail")

    monkeypatch.setattr(commands, "handle", handle)
    with anyio.fail_after(2):
        async with anyio.create_task_group() as tasks:
            _ = tasks.start_soon(rig.say, "/gmail read syntheticID")
            await entered.wait()
            await rig.say("/new")
            confirmation = rig.disk.load().replies[-1]
            await rig.controller.on_callback(rig.callback(confirmation, "new_confirm"))
            release.set()
    assert "Untrusted old mail" not in rig.outbox.texts
    assert rig.disk.load().conversation_id == 1
    assert rig.disk.load().history == ()
    assert rig.disk.load().memories == ()
