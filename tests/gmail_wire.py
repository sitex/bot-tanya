import base64
import threading
from typing import Final, Literal, assert_never

import anyio
import httpx2
import pytest

type Route = Literal["token", "profile", "messages", "preview"]
ROUTES: Final[dict[str, Route]] = {
    "token": "token",
    "profile": "profile",
    "messages": "messages",
    "a" * 128: "preview",
}


class GmailWire:
    """Mutable synthetic outcomes and synchronization for offline adapter tests."""

    def __init__(self) -> None:
        self.paths: list[str] = []
        self.thread_ids: list[int] = []
        self.email: str = "owner@example.test"
        self.failure: int = 200
        self.started: anyio.Event | None = None
        self.release: threading.Event | None = None

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.paths.append(request.url.path)
        self.thread_ids.append(threading.get_ident())
        if self.failure != 200:
            return httpx2.Response(self.failure, content=b"private-provider-secret")
        route = ROUTES.get(request.url.path.rsplit("/", maxsplit=1)[-1])
        if route is None:
            pytest.fail(f"Unexpected synthetic request: {request.url.path}")
        match route:
            case "token":
                if self.started is not None and self.release is not None:
                    anyio.from_thread.run_sync(self.started.set)
                    assert self.release.wait(timeout=5)
                return httpx2.Response(
                    200,
                    json={
                        "access_token": "fake-access",
                        "expires_in": 3600,
                        "token_type": "Bearer",
                    },
                )
            case "profile":
                return httpx2.Response(200, json={"emailAddress": self.email})
            case "messages":
                assert request.url.params["maxResults"] == "10"
                assert request.url.params["q"] == "from:test subject:hello"
                assert "pageToken" not in request.url.params
                return httpx2.Response(
                    200,
                    json={
                        "messages": [{"id": "a" * 128, "threadId": "thread"}],
                        "nextPageToken": "do-not-follow",
                    },
                )
            case "preview":
                return httpx2.Response(
                    200,
                    json={
                        "id": "a" * 128,
                        "threadId": "thread",
                        "payload": {
                            "mimeType": "text/plain",
                            "headers": [
                                {
                                    "name": "Subject",
                                    "value": "<b>mail</b>\u202e" * 200,
                                }
                            ],
                            "body": {
                                "data": base64.urlsafe_b64encode(
                                    (
                                        "Hello <script> /gmail disconnect 😀\x00" * 300
                                    ).encode(),
                                ).decode()
                            },
                        },
                    },
                )
            case _:
                assert_never(route)
