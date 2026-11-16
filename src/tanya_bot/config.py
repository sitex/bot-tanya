"""Live configuration and private secret-file access."""

import os
import stat
from pathlib import Path
from typing import Annotated, ClassVar, Final

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

STATE_DIRECTORY: Final = Path(
    os.environ.get("TANYA_API_STATE_DIR", str(Path.home() / ".local/share/tanya/api"))
)
CLIENT_TOKEN_FILE: Final = Path.home() / ".config/tanya/client-token"
SECRET_MODE: Final = 0o600
SECRET_LIMIT: Final = 8192


class Settings(BaseSettings):
    """Recreate per request so provisioning and rotation require no restart."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        frozen=True, env_prefix="TANYA_"
    )

    model: Annotated[
        str,
        Field(
            min_length=1,
            max_length=128,
            validation_alias=AliasChoices("TANYA_MODEL", "TOLAN_MODEL"),
        ),
    ] = "gpt-6-luna"
    client_token: Annotated[
        SecretStr | None,
        Field(
            validation_alias=AliasChoices("TANYA_CLIENT_TOKEN", "TOLAN_CLIENT_TOKEN")
        ),
    ] = None
    client_token_file: Annotated[
        Path,
        Field(
            validation_alias=AliasChoices(
                "TANYA_CLIENT_TOKEN_FILE", "TOLAN_CLIENT_TOKEN_FILE"
            )
        ),
    ] = CLIENT_TOKEN_FILE


def private_secret(path: Path) -> SecretStr | None:
    """Read only an owned regular 0600 file, without following a symlink."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if (
                metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != SECRET_MODE
                or not stat.S_ISREG(metadata.st_mode)
            ):
                return None
            value = stream.read(SECRET_LIMIT + 1)
        if len(value) > SECRET_LIMIT:
            return None
        return clean_secret(value.decode("utf-8"))
    except (OSError, UnicodeError):
        return None


def clean_secret(value: str) -> SecretStr | None:
    """Allow a trailing file newline, while rejecting multiline credentials."""
    stripped = value.strip()
    if stripped and stripped.isascii() and not any(c.isspace() for c in stripped):
        return SecretStr(stripped)
    return None


def api_key() -> SecretStr | None:
    """Resolve an explicitly configured credential outside the source tree."""
    value = os.environ.get("OPENAI_API_KEY")
    if value is not None:
        return clean_secret(value)
    return private_secret(
        Path(
            os.environ.get(
                "TANYA_API_KEY_FILE", str(Path.home() / ".config/tanya/openai-key")
            )
        )
    )


def client_token(settings: Settings) -> SecretStr | None:
    """Resolve the current client credential independently from the API key."""
    if settings.client_token is not None:
        return clean_secret(settings.client_token.get_secret_value())
    return private_secret(settings.client_token_file)
