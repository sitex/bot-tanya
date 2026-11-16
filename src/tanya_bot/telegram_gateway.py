"""Bridge the existing companion API with strictly local receipt recovery."""

from http import HTTPStatus
from pathlib import Path
from typing import Final

import httpx2

from tanya_bot import turn_receipts
from tanya_bot.config import Settings, client_token
from tanya_bot.errors import ServiceError
from tanya_bot.provider import http_transport
from tanya_bot.receipts import exclusive_store, lookup
from tanya_bot.schemas import ChatRequest, ChatResponse, ErrorResponse
from tanya_bot.telegram_store import StateStore
from tanya_bot.turns import TurnRequest, TurnResponse

CHAT_URL: Final = "http://127.0.0.1:8758/v1/chat"


class LocalCompanionGateway:
    """Reuse the same profile, memory and paid request ledger as the iOS API."""

    def __init__(
        self,
        directory: Path,
        transport: httpx2.AsyncBaseTransport | None = None,
        *,
        core: StateStore | None = None,
    ) -> None:
        """Allow a bounded synthetic HTTP boundary for offline tests."""
        self.directory: Path = directory
        self.transport: httpx2.AsyncBaseTransport | None = transport
        self.core: StateStore | None = core

    def _turn(self, request: ChatRequest) -> TurnRequest | None:
        if self.core is None:
            return None
        state = self.core.load()
        pending = next(
            (
                item
                for item in state.requests
                if item.request.request_id == request.request_id
            ),
            None,
        )
        if pending is None or pending.request != request:
            raise ServiceError(
                409, "missing_reservation", "The frozen request is absent."
            )
        return (
            TurnRequest(chat=request, context=pending.context)
            if pending.context
            else None
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send exactly once; an unresolved core reservation survives any error."""
        token = client_token(Settings())
        if token is None:
            raise ServiceError(503, "client_token_missing", "Configure the local API.")
        turn = self._turn(request)
        async with httpx2.AsyncClient(
            transport=self.transport or http_transport(),
            timeout=httpx2.Timeout(100, connect=5, write=10, pool=5),
            trust_env=False,
            follow_redirects=False,
        ) as client:
            response = await client.post(
                CHAT_URL.replace("/v1/chat", "/v2/telegram/chat") if turn else CHAT_URL,
                headers={"Authorization": f"Bearer {token.get_secret_value()}"},
                content=turn.model_dump_json() if turn else request.model_dump_json(),
            )
        if response.status_code != HTTPStatus.OK:
            parsed = ErrorResponse.model_validate_json(response.content)
            raise ServiceError(response.status_code, parsed.detail.code, "Chat held.")
        return (
            TurnResponse.model_validate_json(response.content)
            if turn
            else ChatResponse.model_validate_json(response.content)
        )

    async def recover(self, request: ChatRequest) -> ChatResponse | None:
        """Read the receipt only; this method cannot issue an HTTP request."""
        turn = self._turn(request)
        directory = self.directory / "turns-v2" if turn else self.directory
        with exclusive_store(directory) as connection:
            try:
                return (
                    turn_receipts.lookup(connection, turn)
                    if turn
                    else lookup(connection, request)
                )
            except ServiceError as error:
                if error.code == "request_outcome_unknown":
                    return None
                raise
