"""Telegram v2 route lives inside the existing companion API process."""

from pathlib import Path
from typing import Final

import httpx2
from fastapi import APIRouter, Request
from pydantic import ValidationError

from tanya_bot.authentication import authenticate
from tanya_bot.config import Settings, api_key
from tanya_bot.errors import ServiceError
from tanya_bot.provider import http_transport
from tanya_bot.receipts import exclusive_store
from tanya_bot.turn_provider import generate
from tanya_bot.turn_receipts import complete, lookup, reserve
from tanya_bot.turns import TurnRequest, TurnResponse

TURN_BODY_BYTES: Final = 128_000
TURN_CONTEXT_CHARACTERS: Final = 16_000


def turn_router(
    directory: Path,
    transport: httpx2.AsyncBaseTransport | None = None,
) -> APIRouter:
    """Legacy /v1/chat remains independent of enhanced envelopes and hashes."""
    router = APIRouter()

    @router.post("/v2/telegram/chat")
    async def turn(request: Request) -> TurnResponse:
        settings = Settings()
        authenticate(request, settings)
        data = bytearray()
        async for part in request.stream():
            if len(data) + len(part) > TURN_BODY_BYTES:
                raise ServiceError(413, "request_too_large", "Turn body is too large.")
            data.extend(part)
        try:
            parsed = TurnRequest.model_validate_json(data)
        except ValidationError:
            raise ServiceError(
                422, "invalid_turn", "Expected a bounded v2 turn."
            ) from None
        chat = parsed.chat
        size = (
            len(parsed.context.summary.text)
            + len(chat.companion_name)
            + len(chat.language)
        )
        size += sum(len(item.content) for item in chat.messages)
        size += sum(map(len, chat.memories))
        size += chat.profile.context_size() if chat.profile else 0
        if size > TURN_CONTEXT_CHARACTERS or chat.attachments:
            raise ServiceError(422, "context_limit", "Telegram context is too large.")
        with exclusive_store(directory / "turns-v2") as connection:
            cached = lookup(connection, parsed)
            if cached is not None:
                return cached
            key = api_key()
            if key is None:
                raise ServiceError(
                    503, "api_key_missing", "Configure the provider key."
                )
            reserve(connection, parsed)
            response = await generate(
                parsed, (settings.model, key), transport or http_transport()
            )
            complete(connection, response)
            return response

    return router
