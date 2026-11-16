from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, assert_never

import anyio
import httpx2
import pytest
from pydantic import SecretStr
from tanya_gmail import FileTokenStore, GmailConfig, RefreshCredential

from tanya_bot.gmail_commands import GmailCommands
from tests.gmail_wire import GmailWire

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class GmailHarness:
    commands: GmailCommands
    config_path: Path
    token_path: Path
    wire: GmailWire


@pytest.fixture
def gmail(tmp_path: Path) -> GmailHarness:
    config = GmailConfig(expected_email="owner@example.test", client_id="desktop")
    config_path = tmp_path / "gmail.json"
    _ = config_path.write_text(config.model_dump_json())
    config_path.chmod(0o600)
    token_path = tmp_path / "tokens"
    FileTokenStore(token_path).save(
        RefreshCredential(
            client_id=config.client_id,
            expected_email=config.expected_email,
            refresh_token=SecretStr("fake-refresh"),
        )
    )
    wire = GmailWire()
    return GmailHarness(
        GmailCommands(
            config_path,
            token_path,
            transport=httpx2.MockTransport(wire.handle),
        ),
        config_path,
        token_path,
        wire,
    )


@pytest.mark.anyio
async def test_unconfigured_constructor_and_status_are_local(tmp_path: Path) -> None:
    commands = GmailCommands(tmp_path / "absent", tmp_path / "tokens")

    output = await commands.handle("/gmail", "en")

    assert "not configured" in output.text
    assert str(tmp_path) not in output.text
    assert not (tmp_path / "tokens").exists()
    assert output.actions == (
        "gmail_status",
        "gmail_connect",
        "gmail_list",
        "gmail_disconnect",
    )


@pytest.mark.anyio
async def test_status_does_not_trust_saved_credentials(gmail: GmailHarness) -> None:
    output = await gmail.commands.handle("/gmail status", "en")

    assert "disconnected" in output.text
    assert gmail.wire.paths == []


@pytest.mark.anyio
async def test_connect_verifies_profile_before_reporting_connected(
    gmail: GmailHarness,
) -> None:
    connected = await gmail.commands.handle("/gmail connect", "en")
    status = await gmail.commands.handle("/gmail", "en")

    assert "is connected" in connected.text
    assert status.text == connected.text
    assert gmail.wire.paths == ["/token", "/gmail/v1/users/me/profile"]
    assert set(gmail.wire.thread_ids) != {threading.get_ident()}


@pytest.mark.anyio
async def test_account_mismatch_stays_disconnected(gmail: GmailHarness) -> None:
    gmail.wire.email = "another@example.test"

    output = await gmail.commands.handle("/gmail connect", "en")
    status = await gmail.commands.handle("/gmail status", "en")

    assert output.text.endswith("account_mismatch")
    assert "disconnected" in status.text
    assert "another@" not in output.text


@pytest.mark.anyio
async def test_missing_credentials_require_explicit_host_browser(
    gmail: GmailHarness,
) -> None:
    FileTokenStore(gmail.token_path).delete()

    output = await gmail.commands.handle("/gmail connect", "en")

    assert "host browser" in output.text
    assert gmail.wire.paths == []


@pytest.mark.anyio
async def test_list_is_one_page_without_reading_contents(gmail: GmailHarness) -> None:
    _ = await gmail.commands.handle("/gmail connect", "en")

    output = await gmail.commands.handle("/gmail list from:test subject:hello", "en")

    assert f"/gmail read {'a' * 128}" in output.text
    assert gmail.wire.paths == [
        "/token",
        "/gmail/v1/users/me/profile",
        "/gmail/v1/users/me/messages",
    ]
    assert all(len(action.encode()) <= 20 for action in output.actions)


@pytest.mark.anyio
async def test_read_bounds_plain_untrusted_content_without_executing_it(
    gmail: GmailHarness,
) -> None:
    _ = await gmail.commands.handle("/gmail connect", "en")

    output = await gmail.commands.handle(f"/gmail read {'a' * 128}", "en")

    assert "Hello <script> /gmail disconnect" in output.text
    assert "\x00" not in output.text
    assert "\u202e" not in output.text
    assert len(output.text.encode("utf-16-le")) // 2 < 4096
    assert FileTokenStore(gmail.token_path).load() is not None


@pytest.mark.anyio
async def test_disconnect_deletes_credentials_without_network(
    gmail: GmailHarness,
) -> None:
    output = await gmail.commands.handle("/gmail disconnect", "en")

    assert "removed" in output.text
    assert FileTokenStore(gmail.token_path).load() is None
    assert gmail.wire.paths == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    "unsafe", ["mode", "symlink", "parent", "hardlink", "large", "fifo"]
)
async def test_config_refuses_unsafe_files(
    gmail: GmailHarness,
    unsafe: Literal["mode", "symlink", "parent", "hardlink", "large", "fifo"],
) -> None:
    path = gmail.config_path
    match unsafe:
        case "mode":
            path.chmod(0o644)
        case "symlink":
            _ = path.rename(path.with_suffix(".real"))
            path.symlink_to(path.with_suffix(".real"))
        case "parent":
            alias = path.parent / "alias"
            alias.symlink_to(path.parent, target_is_directory=True)
            gmail = GmailHarness(
                GmailCommands(alias / path.name, gmail.token_path),
                path,
                gmail.token_path,
                gmail.wire,
            )
        case "hardlink":
            os.link(path, path.with_suffix(".link"))
        case "large":
            _ = path.write_bytes(b" " * 16385)
        case "fifo":
            path.unlink()
            os.mkfifo(path, mode=0o600)
        case _:
            assert_never(unsafe)

    output = await gmail.commands.handle("/gmail connect", "en")

    assert output.text.endswith("unsafe_storage")
    assert gmail.wire.paths == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    "command", ["/gmail send hi", "/gmail read", "/gmail status extra", "/gmailish"]
)
async def test_invalid_commands_do_not_touch_google(
    gmail: GmailHarness, command: str
) -> None:
    output = await gmail.commands.handle(command, "en")

    assert "Gmail commands:" in output.text
    assert gmail.wire.paths == []


@pytest.mark.anyio
async def test_error_output_does_not_disclose_provider_content(
    gmail: GmailHarness,
) -> None:
    gmail.wire.failure = 503

    output = await gmail.commands.handle("/gmail connect", "en")

    assert output.text.endswith("http_failure")
    assert "private-provider-secret" not in output.text


@pytest.mark.anyio
async def test_account_lock_keeps_disconnect_after_connect(gmail: GmailHarness) -> None:
    gmail.wire.started = anyio.Event()
    gmail.wire.release = threading.Event()

    async def invoke(text: str) -> None:
        _ = await gmail.commands.handle(text, "en")

    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            _ = group.start_soon(invoke, "/gmail connect")
            await gmail.wire.started.wait()
            _ = group.start_soon(invoke, "/gmail disconnect")
            await anyio.wait_all_tasks_blocked()
            gmail.wire.release.set()

    assert FileTokenStore(gmail.token_path).load() is None
