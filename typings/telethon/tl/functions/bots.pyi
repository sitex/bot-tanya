from telethon.tl.types import (
    BotCommand,
    BotCommandScopeDefault,
    BotMenuButtonCommands,
    InputUserEmpty,
)

class SetBotCommandsRequest:
    def __init__(
        self, scope: BotCommandScopeDefault, lang_code: str, commands: list[BotCommand]
    ) -> None: ...

class SetBotMenuButtonRequest:
    def __init__(
        self, user_id: InputUserEmpty, button: BotMenuButtonCommands
    ) -> None: ...

class SetBotInfoRequest:
    def __init__(self, lang_code: str, *, name: str, about: str) -> None: ...
