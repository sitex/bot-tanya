"""Load an external Gmail configuration through owned private descriptors."""

import os
import stat
from contextlib import ExitStack
from pathlib import Path
from typing import Final

from pydantic import ValidationError
from tanya_gmail import ErrorCode, GmailConfig, GmailError

MAX_CONFIG: Final = 16384
PRIVATE_MODE: Final = 0o600


def load_config(path: Path) -> GmailConfig:
    """Reject unsafe parent links, special files and non-private credentials."""
    if not path.is_absolute() or ".." in path.parts or path == Path("/"):
        raise GmailError(ErrorCode.UNSAFE_STORAGE)
    try:
        with ExitStack() as stack:
            directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
            _ = stack.callback(os.close, directory)
            for part in path.parts[1:-1]:
                directory = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory,
                )
                _ = stack.callback(os.close, directory)
            descriptor = os.open(
                path.name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory,
            )
            stream = stack.enter_context(os.fdopen(descriptor, "rb"))
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != PRIVATE_MODE
                or info.st_nlink != 1
                or info.st_size > MAX_CONFIG
            ):
                raise GmailError(ErrorCode.UNSAFE_STORAGE)
            data = stream.read(MAX_CONFIG + 1)
            if len(data) > MAX_CONFIG:
                raise GmailError(ErrorCode.UNSAFE_STORAGE)
            return GmailConfig.model_validate_json(data)
    except FileNotFoundError:
        raise
    except OSError:
        raise GmailError(ErrorCode.UNSAFE_STORAGE) from None
    except ValidationError:
        raise GmailError(ErrorCode.INVALID_INPUT) from None
