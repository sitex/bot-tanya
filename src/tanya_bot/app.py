"""ASGI entry point: uv run uvicorn tanya_bot.app:app --workers 1."""

import sqlite3
from http import HTTPStatus
from pathlib import Path
from typing import Final

import httpx2
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from tanya_bot.authentication import authenticate
from tanya_bot.config import (
    STATE_DIRECTORY,
    Settings,
    api_key,
    client_token,
)
from tanya_bot.dialogue_api import dialogue_router
from tanya_bot.errors import ServiceError
from tanya_bot.media import enforce_plain_body_limit
from tanya_bot.media_models import MEDIA_BODY_BYTES
from tanya_bot.provider import generate, http_transport
from tanya_bot.receipts import exclusive_store, lookup, mark_sent, save_success
from tanya_bot.schemas import ChatRequest, ChatResponse, HealthResponse
from tanya_bot.telegram_store import StateStore
from tanya_bot.turn_api import turn_router


async def service_error_handler(_request: Request, error: ServiceError) -> JSONResponse:
    """Expose only sanitized errors, with the Bearer authentication challenge."""
    headers = (
        {"WWW-Authenticate": "Bearer"}
        if error.status == HTTPStatus.UNAUTHORIZED
        else None
    )
    return JSONResponse(
        status_code=error.status,
        content=error.envelope().model_dump(),
        headers=headers,
    )


async def health() -> HealthResponse:
    """Report local configuration readiness without contacting OpenAI."""
    configured = api_key() is not None
    try:
        settings = Settings()
    except ValidationError:
        return HealthResponse(
            ready=False, model="invalid_configuration", api_key_configured=configured
        )
    return HealthResponse(
        ready=configured and client_token(settings) is not None,
        model=settings.model,
        api_key_configured=configured,
    )


def create_app(
    state_directory: Path = STATE_DIRECTORY,
    transport: httpx2.AsyncBaseTransport | None = None,
    *,
    dialogue_core: StateStore | None = None,
) -> FastAPI:
    """Build an isolated ASGI app, optionally using a local HTTP transport."""
    application = FastAPI(title="Tanya companion", docs_url=None, redoc_url=None)

    _ = application.exception_handler(ServiceError)(service_error_handler)
    application.add_api_route("/health", health, methods=["GET"])
    application.include_router(turn_router(state_directory, transport))
    application.include_router(
        dialogue_router(state_directory, transport, core=dialogue_core)
    )

    @application.post("/v1/chat")
    async def chat(request: Request) -> ChatResponse:
        try:
            settings = Settings()
        except ValidationError:
            raise ServiceError(
                503, "configuration", "Check the server model and token configuration."
            ) from None
        authenticate(request, settings)
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > MEDIA_BODY_BYTES:
                raise ServiceError(413, "request_too_large", "Request body too large.")
            body.extend(chunk)
        try:
            parsed = ChatRequest.model_validate_json(body)
        except ValidationError:
            enforce_plain_body_limit(body, None)
            raise ServiceError(
                422,
                "invalid_request",
                "Expected UUID, 1-32 turns, context at most 16000 characters.",
            ) from None
        enforce_plain_body_limit(body, parsed)
        try:
            with exclusive_store(state_directory) as connection:
                cached = lookup(connection, parsed)
                if cached is not None:
                    return cached
                key = api_key()
                if key is None:
                    raise ServiceError(
                        503,
                        "api_key_missing",
                        "Set OPENAI_API_KEY or TANYA_API_KEY_FILE (owned 0600 file).",
                    )
                mark_sent(connection, parsed)
                response = await generate(
                    parsed, (settings.model, key), transport or http_transport()
                )
                save_success(connection, response)
                return response
        except (OSError, sqlite3.Error, ValidationError):
            raise ServiceError(
                503, "storage_unavailable", "Private request receipts are unavailable."
            ) from None

    return application


app: Final = create_app()
