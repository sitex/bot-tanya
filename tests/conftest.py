import socket
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn, assert_never
from uuid import uuid4

import httpx2
import pytest
from fastapi import FastAPI

from tanya_bot.app import create_app
from tanya_bot.schemas import ChatRequest, Message
from tanya_bot.telegram_store import TelegramStore


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(_socket: socket.socket, _address: str | tuple[str, int]) -> NoReturn:
        pytest.fail("Real network access is forbidden in the offline suite")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def completed_response() -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "status": "completed",
            "model": "gpt-6-luna-snapshot",
            "output": [
                {"type": "reasoning", "summary": []},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Hello."},
                        {"type": "output_text", "text": "How are you?"},
                    ],
                },
            ],
            "usage": {"input_tokens": 41, "output_tokens": 7},
        },
    )


@dataclass(frozen=True, slots=True)
class Wire:
    calls: list[httpx2.Request] = field(default_factory=list)
    outcomes: list[httpx2.Response | httpx2.RequestError] = field(default_factory=list)

    async def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.calls.append(request)
        outcome = self.outcomes.pop(0) if self.outcomes else completed_response()
        match outcome:
            case httpx2.Response():
                return outcome
            case httpx2.RequestError():
                raise outcome
            case _:
                assert_never(outcome)


@dataclass(frozen=True, slots=True)
class Harness:
    directory: Path
    wire: Wire
    app: FastAPI

    @asynccontextmanager
    async def client(self) -> AsyncGenerator[httpx2.AsyncClient]:
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=self.app),
            base_url="https://tolan.test",
            timeout=5.0,
            headers={"Authorization": "Bearer test-client-token"},
            trust_env=False,
        ) as client:
            yield client


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    for variable in (
        "TOLAN_MODEL",
        "TOLAN_CLIENT_TOKEN",
        "TOLAN_CLIENT_TOKEN_FILE",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("TANYA_API_KEY_FILE", str(tmp_path / "api-key"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-offline")
    monkeypatch.setenv("TOLAN_CLIENT_TOKEN", "test-client-token")
    monkeypatch.setenv("TOLAN_CLIENT_TOKEN_FILE", str(tmp_path / "client-token"))
    wire = Wire()
    directory = tmp_path / "state"
    app = create_app(
        directory,
        httpx2.MockTransport(wire.handle),
        dialogue_core=TelegramStore(tmp_path / "core", 42),
    )
    return Harness(directory, wire, app)


@pytest.fixture
def chat_request() -> ChatRequest:
    return ChatRequest(
        request_id=uuid4(),
        messages=(
            Message(role="assistant", content="What is on your mind?"),
            Message(role="user", content="I am learning to draw."),
        ),
        companion_name="Mira",
        language="en",
        memories=("I enjoy watercolor.",),
    )
