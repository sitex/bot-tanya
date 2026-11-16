"""Assemble only completed phase receipts; assembly performs no external work."""

import time

from tanya_bot.dialogue_models import DialogueRoot, MailSelection, PhaseReceipt
from tanya_bot.dialogue_store import phase_for
from tanya_bot.errors import ServiceError
from tanya_bot.turns import TurnResponse


def assembled(root: DialogueRoot, receipt: PhaseReceipt) -> DialogueRoot:
    """Mail-derived turns retain the clean summary and cannot promote memories."""
    answer = receipt.answer
    if answer is None or receipt.tool is not None:
        raise ServiceError(502, "dialogue_invalid_final", "No final answer exists.")
    usage = tuple(phase.receipt for phase in root.phases if phase.receipt is not None)
    response = TurnResponse(
        request_id=root.turn.chat.request_id,
        reply=answer.reply,
        model=receipt.model,
        input_tokens=sum(item.input_tokens for item in usage),
        output_tokens=sum(item.output_tokens for item in usage),
        memory_updates=()
        if root.tainted
        else tuple(
            item
            for item in answer.memory_updates
            if item.action == "remember" or item.text in root.turn.chat.memories
        ),
        summary=root.turn.context.summary.text if root.tainted else answer.summary,
        delivery=answer.delivery,
    )
    selection = root.selection
    if root.tool_result is not None:
        call = phase_for(root, "decide").receipt
        origin = (
            selection.created_at
            if selection
            and call
            and call.tool
            and call.tool.action.name == "gmail_read"
            else root.tool_result.retrieved_at
        )
        selection = MailSelection(
            owner_id=root.owner_id,
            conversation_id=root.conversation_id,
            root_id=root.turn.chat.request_id,
            created_at=origin,
            accessed_at=time.time(),
            result=root.tool_result,
        )
    if selection is not None:
        selection = selection.model_copy(
            update={
                "answer": response.reply,
                "accessed_at": time.time(),
            }
        )
    return root.model_copy(update={"final": response, "selection": selection})
