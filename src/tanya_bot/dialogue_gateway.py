"""Local phase transport submits each reservation once, without fallback."""

from http import HTTPStatus

import httpx2

from tanya_bot.config import Settings, client_token
from tanya_bot.dialogue_models import PhaseAddress, PhaseReceipt
from tanya_bot.errors import ServiceError
from tanya_bot.provider import http_transport
from tanya_bot.schemas import ErrorResponse
from tanya_bot.telegram_gateway import CHAT_URL


class LocalPhaseGateway:
    """An injected synthetic boundary supports offline full-loop tests."""

    def __init__(self, transport: httpx2.AsyncBaseTransport | None = None) -> None:
        """Use the authenticated existing loopback API, never a second provider."""
        self.transport: httpx2.AsyncBaseTransport | None = transport

    async def phase(self, address: PhaseAddress) -> PhaseReceipt:
        """Send one address; recovery reads the shared ledger instead."""
        token = client_token(Settings())
        if token is None:
            raise ServiceError(503, "client_token_missing", "Configure the local API.")
        async with httpx2.AsyncClient(
            transport=self.transport or http_transport(),
            timeout=httpx2.Timeout(100, connect=5, write=10, pool=5),
            trust_env=False,
            follow_redirects=False,
        ) as client:
            response = await client.post(
                CHAT_URL.replace("/v1/chat", "/v3/telegram/dialogue/phase"),
                headers={"Authorization": f"Bearer {token.get_secret_value()}"},
                content=address.model_dump_json(),
            )
        if response.status_code != HTTPStatus.OK:
            error = ErrorResponse.model_validate_json(response.content)
            raise ServiceError(
                response.status_code, error.detail.code, "Dialogue held."
            )
        return PhaseReceipt.model_validate_json(response.content)
