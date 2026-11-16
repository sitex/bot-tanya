"""An authenticated phase endpoint cannot create owner mailbox authority."""

from pathlib import Path
from typing import Final

import httpx2
from fastapi import APIRouter, Request
from pydantic import ValidationError

from tanya_bot.authentication import authenticate
from tanya_bot.config import Settings, api_key
from tanya_bot.dialogue_authority import authority, configured_core
from tanya_bot.dialogue_models import PhaseAddress, PhaseReceipt
from tanya_bot.dialogue_provider import generate, payload
from tanya_bot.dialogue_store import DialogueStore, phase_for, replace_phase
from tanya_bot.errors import ServiceError
from tanya_bot.provider import http_transport
from tanya_bot.receipts import exclusive_store
from tanya_bot.telegram_store import CoreError, StateStore

PHASE_BODY_BYTES: Final = 4096


def dialogue_router(
    directory: Path,
    transport: httpx2.AsyncBaseTransport | None = None,
    *,
    core: StateStore | None = None,
) -> APIRouter:
    """Only the bot's previously committed phase may dispatch one model call."""
    router = APIRouter()
    store = DialogueStore(directory / "dialogue-v3")

    @router.post("/v3/telegram/dialogue/phase")
    async def phase(request: Request) -> PhaseReceipt:
        authenticate(request, Settings())
        data = bytearray()
        async for part in request.stream():
            if len(data) + len(part) > PHASE_BODY_BYTES:
                raise ServiceError(413, "request_too_large", "Phase body is too large.")
            data.extend(part)
        try:
            address = PhaseAddress.model_validate_json(data)
        except ValidationError:
            raise ServiceError(422, "invalid_phase", "Invalid phase address.") from None
        with exclusive_store(directory):
            pass
        root = store.load(address.root_id)
        if root is None:
            raise ServiceError(409, "dialogue_missing", "No owner reservation exists.")
        saved = phase_for(root, address.kind)
        if saved.receipt is not None:
            return saved.receipt
        key = api_key()
        if key is None:
            raise ServiceError(503, "api_key_missing", "Configure the provider key.")
        body = payload(root, address.kind)
        try:
            with authority(core or configured_core(), root):
                _ = store.claim(address.root_id, address.kind, body)
        except CoreError:
            raise ServiceError(
                503, "dialogue_authority_unavailable", "Core authority is unavailable."
            ) from None
        try:
            result = await generate(body, key, transport or http_transport())
        except ServiceError as error:
            code = error.code
            _ = store.mutate(
                address.root_id,
                lambda current: replace_phase(
                    current,
                    phase_for(current, address.kind).model_copy(
                        update={"failure": code}
                    ),
                ),
            )
            raise
        _ = store.mutate(
            address.root_id,
            lambda current: replace_phase(
                current,
                phase_for(current, address.kind).model_copy(update={"receipt": result}),
            ),
        )
        return result

    return router
