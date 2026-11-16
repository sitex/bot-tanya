"""Shared client Bearer authentication before reading request bodies."""

import hmac

from fastapi import Request

from tanya_bot.config import Settings, client_token
from tanya_bot.errors import ServiceError


def authenticate(request: Request, settings: Settings) -> None:
    """Verify client auth before consuming or validating the dialogue."""
    token = client_token(settings)
    if token is None:
        raise ServiceError(
            503,
            "client_token_missing",
            "Configure the client token or its private file.",
        )
    authorization = request.headers.get("authorization", "")
    scheme, _, supplied = authorization.partition(" ")
    if not (
        scheme.lower() == "bearer"
        and hmac.compare_digest(supplied.encode(), token.get_secret_value().encode())
    ):
        raise ServiceError(401, "auth", "A valid client Bearer token is required.")
