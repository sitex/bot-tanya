from pathlib import Path
from threading import Event

import anyio
import httpx2
import pytest
from tanya_gmail import ConnectionStatus, ErrorCode, FileTokenStore, GmailError

from tanya_bot.gmail_account import GmailAccount
from tanya_bot.gmail_commands import GmailCommands
from tests.gmail_account_fakes import AccountWire, account_at
from tests.gmail_wire import GmailWire

pytestmark = pytest.mark.anyio


def test_account_construction_never_opens_config_or_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(_path: Path) -> None:
        pytest.fail("Construction performed filesystem I/O")

    monkeypatch.setattr("tanya_bot.gmail_account.load_config", forbidden)
    monkeypatch.setattr(FileTokenStore, "load", forbidden)
    wire = AccountWire()

    account = GmailAccount(
        tmp_path / "absent", tmp_path / "tokens", httpx2.MockTransport(wire.handle)
    )

    assert account.cached is None
    assert wire.paths == []
    assert tuple(tmp_path.iterdir()) == ()


async def test_restart_search_restores_saved_token_and_verifies_owner(
    tmp_path: Path,
) -> None:
    wire = AccountWire()
    original = account_at(tmp_path, wire)
    _ = await original.search("older", 2)
    restarted = GmailAccount(original.config, original.tokens, original.transport)
    wire.paths.clear()
    wire.queries.clear()

    page = await restarted.search("newer_than:1d subject:invoice", 2)

    assert tuple(message.id for message in page.messages) == ("mail_1",)
    expected_continuation = "unfollowed"
    assert page.next_page_token == expected_continuation
    assert wire.queries == ["newer_than:1d subject:invoice"]
    assert wire.paths == [
        "/token",
        "/gmail/v1/users/me/profile",
        "/gmail/v1/users/me/messages",
    ]


@pytest.mark.parametrize("email", ["other@example.test", "Owner@example.test"])
async def test_search_rejects_nonexact_owner_before_mail_access(
    tmp_path: Path, email: str
) -> None:
    wire = AccountWire()
    wire.email = email
    account = account_at(tmp_path, wire)

    with pytest.raises(GmailError) as raised:
        _ = await account.search("private", 2)

    assert raised.value.code is ErrorCode.ACCOUNT_MISMATCH
    assert wire.paths == ["/token", "/gmail/v1/users/me/profile"]
    assert account.client().status() is ConnectionStatus.DISCONNECTED


@pytest.mark.parametrize("operation", ["search", "read"])
@pytest.mark.parametrize("fail_with_timeout", [False, True])
async def test_failed_mail_operation_is_not_implicitly_repeated(
    tmp_path: Path, operation: str, *, fail_with_timeout: bool
) -> None:
    wire = AccountWire()
    wire.timeout = fail_with_timeout
    wire.mail_failure = 503
    account = account_at(tmp_path, wire)
    execute = {"search": account.search, "read": account.read}[operation]
    argument = "private" if operation == "search" else "mail_1"

    with pytest.raises(GmailError) as raised:
        _ = await execute(argument, 2)

    expected = (
        ErrorCode.NETWORK_FAILURE if fail_with_timeout else ErrorCode.HTTP_FAILURE
    )
    assert raised.value.code is expected
    suffix = "" if operation == "search" else "/mail_1"
    assert wire.paths == [
        "/token",
        "/gmail/v1/users/me/profile",
        f"/gmail/v1/users/me/messages{suffix}",
    ]


@pytest.mark.parametrize("command_first", [False, True])
async def test_commands_and_account_share_lock_across_mailbox_io(
    tmp_path: Path, *, command_first: bool
) -> None:
    wire = GmailWire()
    account = account_at(tmp_path, wire)
    commands = GmailCommands(
        account.config, account.tokens, transport=account.transport
    )
    account = commands.account
    wire.started = anyio.Event()
    wire.release = Event()
    finished: list[str] = []

    async def search() -> None:
        page = await account.search("from:test subject:hello", 10)
        assert tuple(message.id for message in page.messages) == ("a" * 128,)
        finished.append("search")

    async def command() -> None:
        text = "/gmail connect" if command_first else "/gmail disconnect"
        _ = await commands.handle(text, "en")
        finished.append("command")

    with anyio.fail_after(10):
        async with anyio.create_task_group() as tasks:
            _ = tasks.start_soon(command if command_first else search)
            await wire.started.wait()
            _ = tasks.start_soon(search if command_first else command)
            try:
                await anyio.wait_all_tasks_blocked()
                assert finished == []
                assert wire.paths == ["/token"]
            finally:
                wire.release.set()

    expected_order = ["command", "search"] if command_first else ["search", "command"]
    assert finished == expected_order
    assert (FileTokenStore(account.tokens).load() is not None) == command_first
    assert wire.paths == [
        "/token",
        "/gmail/v1/users/me/profile",
        "/gmail/v1/users/me/messages",
    ]
