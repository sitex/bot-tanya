"""Load only the new bot's private identity and SDK credentials."""

import os
from pathlib import Path
from typing import Annotated, ClassVar, Final, Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from tanya_bot.config import clean_secret, private_secret
from tanya_bot.receipts import private_file
from tanya_bot.schemas import FrozenModel
from tanya_bot.telegram_store import CoreError

BOT_DIRECTORY: Final = Path(
    os.environ.get("TANYA_DATA_DIR", str(Path.home() / ".local/share/tanya/bot"))
)
IDENTITY_BYTES: Final = 8192


class SDKCredentials(BaseSettings):
    """Read the two required SDK fields, without importing another bot's code."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        frozen=True, extra="ignore"
    )
    telegram_api_id: Annotated[int, Field(gt=0)]
    telegram_api_hash: SecretStr


class BotIdentity(FrozenModel):
    """The completed creation receipt fixes both bot and personal owner."""

    stage: Literal["created"]
    owner_id: Annotated[int, Field(gt=0)]
    bot_id: Annotated[int, Field(gt=0)]
    username: Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{4,31}$")]


def identity(directory: Path = BOT_DIRECTORY) -> BotIdentity:
    """Refuse a missing or unsafe receipt instead of starting bot creation."""
    path = directory / "creation.json"
    if not path.is_file():
        raise CoreError(code="bot_creation_required")
    with os.fdopen(private_file(path), "rb") as source:
        data = source.read(IDENTITY_BYTES + 1)
    if len(data) > IDENTITY_BYTES:
        raise CoreError(code="bot_identity_capacity")
    return BotIdentity.model_validate_json(data)


def bot_token(directory: Path = BOT_DIRECTORY) -> SecretStr:
    """Read a pre-provisioned owned 0600 credential; never include it in logs."""
    raw = os.environ.get("TANYA_BOT_TOKEN")
    token = (
        clean_secret(raw)
        if raw is not None
        else private_secret(directory / "bot-token")
    )
    if token is None:
        raise CoreError(code="bot_token_required")
    return token
