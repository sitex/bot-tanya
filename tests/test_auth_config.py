from pathlib import Path
from typing import Literal, assert_never

import pytest

from tanya_bot.schemas import ChatRequest, ErrorResponse, HealthResponse
from tests.conftest import Harness

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize(
    "authorization", ["", "Bearer wrong", "Basic test-client-token", "Bearer"]
)
async def test_authentication_precedes_body_parsing(
    harness: Harness, authorization: str
) -> None:
    async with harness.client() as client:
        response = await client.post(
            "/v1/chat", content=b"not-json", headers={"Authorization": authorization}
        )

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert ErrorResponse.model_validate_json(response.content).detail.code == "auth"
    assert harness.wire.calls == []


async def test_missing_key_then_live_private_file_provisioning(
    harness: Harness, chat_request: ChatRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    async with harness.client() as client:
        before = await client.get("/health")
        missing = await client.post("/v1/chat", content=chat_request.model_dump_json())
        assert HealthResponse.model_validate_json(before.content) == HealthResponse(
            ready=False, model="gpt-6-luna", api_key_configured=False
        )
        assert missing.status_code == 503
        assert ErrorResponse.model_validate_json(missing.content).detail.code == (
            "api_key_missing"
        )
        assert harness.wire.calls == []
        key_path = harness.directory.parent / "api-key"
        _ = key_path.write_text("sk-provisioned-offline\n")
        key_path.chmod(0o600)

        ready = await client.get("/health")
        success = await client.post("/v1/chat", content=chat_request.model_dump_json())

    assert HealthResponse.model_validate_json(ready.content).ready is True
    assert success.status_code == 200
    assert len(harness.wire.calls) == 1
    assert harness.wire.calls[0].headers["authorization"] == (
        "Bearer sk-provisioned-offline"
    )
    assert "sk-provisioned-offline" not in ready.text + success.text


@pytest.mark.parametrize("kind", ["permissions", "symlink"])
async def test_api_key_file_requires_private_regular_file(
    harness: Harness,
    chat_request: ChatRequest,
    monkeypatch: pytest.MonkeyPatch,
    kind: Literal["permissions", "symlink"],
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    path = harness.directory.parent / "api-key"
    match kind:
        case "permissions":
            _ = path.write_text("sk-private-offline")
            path.chmod(0o644)
        case "symlink":
            target = harness.directory.parent / "target-key"
            _ = target.write_text("sk-private-offline")
            target.chmod(0o600)
            path.symlink_to(target)
        case _:
            assert_never(kind)
    async with harness.client() as client:
        health = await client.get("/health")
        chat = await client.post("/v1/chat", content=chat_request.model_dump_json())

    assert HealthResponse.model_validate_json(health.content).ready is False
    assert chat.status_code == 503
    assert harness.wire.calls == []


async def test_client_file_rotation_and_model_override_are_live(
    harness: Harness,
    chat_request: ChatRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("TOLAN_CLIENT_TOKEN")
    token_file = tmp_path / "client-token"
    _ = token_file.write_text("test-client-token\n")
    token_file.chmod(0o600)
    async with harness.client() as client:
        initial = await client.get("/health")
        assert HealthResponse.model_validate_json(initial.content).ready is True
        _ = token_file.write_text("rotated-token\n")
        monkeypatch.setenv("TOLAN_MODEL", "explicit-model")

        stale = await client.post("/v1/chat", content=chat_request.model_dump_json())
        current = await client.post(
            "/v1/chat",
            content=chat_request.model_dump_json(),
            headers={"Authorization": "Bearer rotated-token"},
        )
        health = await client.get("/health")

    assert stale.status_code == 401
    assert current.status_code == 200
    assert len(harness.wire.calls) == 1
    assert HealthResponse.model_validate_json(health.content).model == "explicit-model"
