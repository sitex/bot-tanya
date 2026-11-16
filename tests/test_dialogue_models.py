import pytest
from pydantic import ValidationError
from tanya_persona import DeliveryPlan

from tanya_bot.dialogue_models import MailCall, MailSearch, PhaseReceipt
from tanya_bot.turns import TurnAnswer


@pytest.mark.parametrize("limit", ['"2"', "true", "2.0"])
def test_tool_boundary_rejects_coercible_wrong_types(limit: str) -> None:
    with pytest.raises(ValidationError):
        _ = MailSearch.model_validate_json('{"query":"", "limit":' + limit + "}")


def test_receipt_rejects_missing_outcome() -> None:
    with pytest.raises(ValidationError, match="phase_outcome"):
        _ = PhaseReceipt(
            provider_id="resp_test",
            model="offline",
            input_tokens=0,
            output_tokens=0,
            output=(),
        )


def test_receipt_rejects_mixed_tool_and_answer() -> None:
    with pytest.raises(ValidationError, match="phase_outcome"):
        _ = PhaseReceipt(
            provider_id="resp_test",
            model="offline",
            input_tokens=0,
            output_tokens=0,
            output=(),
            answer=TurnAnswer(
                reply="Hello", memory_updates=(), summary="", delivery=DeliveryPlan()
            ),
            tool=MailCall(call_id="call_test", action=MailSearch(query="", limit=1)),
        )
