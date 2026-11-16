"""Structured conversational facts returned with the same paid chat response."""

from typing import Annotated, ClassVar, Final, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from tanya_bot.schemas import MemoryUpdate


class RememberedReply(BaseModel):
    """Parse the whole provider output before publishing a reply or memory."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    reply: Annotated[str, Field(min_length=1)]
    memory_updates: Annotated[tuple[MemoryUpdate, ...], Field(max_length=8)]


type JSONValue = str | int | bool | list[JSONValue] | dict[str, JSONValue] | None


class StructuredFormat(TypedDict):
    """The exact Responses API structured-output envelope."""

    type: Literal["json_schema"]
    name: str
    strict: bool
    schema: dict[str, JSONValue]


MEMORY_INSTRUCTIONS: Final = """
Return JSON matching the supplied reply/memory_updates schema. reply is the natural
conversational reply, without describing JSON. memory_updates may be empty. Remember
only durable facts or preferences the user explicitly stated, in their language;
never infer diagnoses, emotions, identity, relationships, dates or events. Do not
store secrets, passwords, payment details or incidental sensitive facts. Never use
assistant inventions as evidence. Add only new information relevant to future chats.
For an explicit user request to forget/correct a supplied memory, use action=forget
with the EXACT existing memory text. A correction may also remember the new fact.
Do not forget memories without the user's request. At most 8 updates. New facts
must be <=500 characters; forget targets must exactly reproduce existing text,
even when it is longer. Do not add facts already present in supplied memories.
"""


def output_format() -> StructuredFormat:
    """Declare a small strict schema, independently bounded again on receipt."""
    return {
        "type": "json_schema",
        "name": "companion_reply",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reply": {"type": "string", "minLength": 1},
                "memory_updates": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["remember", "forget"],
                            },
                            "text": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 16000,
                            },
                        },
                        "required": ["action", "text"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["reply", "memory_updates"],
            "additionalProperties": False,
        },
    }
