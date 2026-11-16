"""Version-two Telegram envelopes leave legacy chat request bytes unchanged."""

from typing import Annotated, Final, assert_never

from pydantic import Field, TypeAdapter
from tanya_memory import TurnContext
from tanya_persona import DeliveryPlan

from tanya_bot.conversation_memory import JSONValue
from tanya_bot.schemas import ChatRequest, ChatResponse, FrozenModel, MemoryUpdate

SCHEMA_VALUE: Final = TypeAdapter(dict[str, JSONValue])


class TurnRequest(FrozenModel):
    """The exact frozen context is included in the new receipt fingerprint."""

    chat: ChatRequest
    context: TurnContext


class TurnAnswer(FrozenModel):
    """One provider result contains reply, explicit facts and derived guidance."""

    reply: Annotated[str, Field(min_length=1, max_length=16_000)]
    memory_updates: Annotated[tuple[MemoryUpdate, ...], Field(max_length=8)]
    summary: Annotated[str, Field(max_length=1800)]
    delivery: DeliveryPlan


class TurnResponse(ChatResponse):
    """An enhanced response is persisted as a distinct union variant."""

    summary: Annotated[str, Field(max_length=1800)]
    delivery: DeliveryPlan


def strict_schema(value: JSONValue) -> JSONValue:
    """OpenAI strict objects require every property, even model default fields."""
    match value:
        case dict():
            result = {
                key: strict_schema(item)
                for key, item in value.items()
                if key != "default"
            }
            properties = result.get("properties")
            match properties:
                case dict():
                    result["required"] = list(properties)
                    result["additionalProperties"] = False
                case list() | str() | int() | None:
                    pass
                case _:
                    assert_never(properties)
            return result
        case list():
            return [strict_schema(item) for item in value]
        case str() | int() | None:
            return value
        case _:
            assert_never(value)


def turn_schema() -> dict[str, JSONValue]:
    """Generate the bounded schema from the actual component models."""
    schema = SCHEMA_VALUE.validate_python(TurnAnswer.model_json_schema())
    return SCHEMA_VALUE.validate_python(strict_schema(schema))
