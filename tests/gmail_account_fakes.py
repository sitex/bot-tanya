from pathlib import Path
from typing import override

import httpx2
import pytest
from pydantic import SecretStr
from tanya_gmail import FileTokenStore, GmailConfig, RefreshCredential

from tanya_bot.gmail_account import GmailAccount
from tests.gmail_wire import GmailWire


def account_at(path: Path, wire: GmailWire) -> GmailAccount:
    config = GmailConfig(expected_email="owner@example.test", client_id="desktop")
    config_path = path / "gmail.json"
    _ = config_path.write_text(config.model_dump_json())
    config_path.chmod(0o600)
    tokens = path / "tokens"
    FileTokenStore(tokens).save(
        RefreshCredential(
            client_id=config.client_id,
            expected_email=config.expected_email,
            refresh_token=SecretStr("fake-refresh"),
        )
    )
    return GmailAccount(config_path, tokens, httpx2.MockTransport(wire.handle))


class AccountWire(GmailWire):
    def __init__(self) -> None:
        super().__init__()
        self.mail_failure: int = 200
        self.timeout: bool = False
        self.queries: list[str] = []

    @override
    def handle(self, request: httpx2.Request) -> httpx2.Response:
        if "/messages" in request.url.path:
            self.paths.append(request.url.path)
            assert request.headers["Authorization"] == "Bearer fake-access"
            if self.timeout:
                message = "offline unknown"
                raise httpx2.ReadTimeout(message, request=request)
            if self.mail_failure != 200:
                return httpx2.Response(self.mail_failure)
            if request.url.path.endswith("/messages"):
                self.queries.append(request.url.params["q"])
                assert request.url.params["maxResults"] == "2"
                assert "pageToken" not in request.url.params
                return httpx2.Response(
                    200,
                    json={
                        "messages": [{"id": "mail_1", "threadId": "thread_1"}],
                        "nextPageToken": "unfollowed",
                    },
                )
            pytest.fail("Unexpected synthetic mail read")
        return super().handle(request)
