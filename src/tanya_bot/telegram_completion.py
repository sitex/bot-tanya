"""Keep archived completions out of the active conversation and its memory."""

from typing import assert_never
from uuid import UUID

from tanya_memory import same_revision

from tanya_bot.schemas import CONTEXT_CHARACTERS, ChatResponse
from tanya_bot.telegram_core_context import apply_response
from tanya_bot.telegram_core_session import CoreSession
from tanya_bot.telegram_core_voice import voice_reply
from tanya_bot.telegram_models import (
    EVENT_CAPACITY,
    PendingChat,
    StoredReply,
    VoiceGateway,
)
from tanya_bot.telegram_store import CoreError
from tanya_bot.turns import TurnResponse


class ReplyCompletion:
    """Complete a reserved request under the controller's existing state lock."""

    def __init__(self, session: CoreSession, voice: VoiceGateway | None) -> None:
        """Use the same saved snapshot for output, memory and delivery decisions."""
        self.session: CoreSession = session
        self.voice: VoiceGateway | None = voice

    async def complete(
        self, request_id: UUID, response: ChatResponse, *, recover: bool = False
    ) -> None:
        """Save old results as receipts; explicit recovery never generates audio."""
        session = self.session
        pending = self._pending(request_id, response)
        if pending.response is not None:
            return
        archived = pending.conversation_id != session.state.conversation_id
        derived_stale = pending.context is not None and (
            pending.cancelled
            or not same_revision(session.state.context, pending.context)
        )
        if pending.cancelled and not recover and not archived:
            return
        updated = (
            session.state
            if archived or derived_stale
            else apply_response(session.state, pending, response)
        )
        reply = self._reply(pending, response, stale=archived or derived_stale)
        reply = reply.model_copy(update={"receipt_only": derived_stale and not recover})
        if not archived and not reply.receipt_only:
            reply = session.available(reply)
        completed = pending.model_copy(
            update={"response": response, "reply_token": reply.token}
        )
        if len(updated.replies) >= EVENT_CAPACITY:
            raise CoreError(code="reply_capacity")
        session.save(
            updated.model_copy(
                update={
                    "requests": tuple(
                        completed if item.request.request_id == request_id else item
                        for item in updated.requests
                    ),
                    "replies": (*updated.replies, reply),
                }
            ),
            completion=True,
        )
        if archived or reply.receipt_only:
            return
        await session.deliver(reply)
        if (
            not recover
            and session.state.settings.voice_replies
            and self.voice is not None
        ):
            bound = next(
                item for item in session.state.replies if item.token == reply.token
            )
            await voice_reply(session, bound, self.voice)

    def _pending(self, request_id: UUID, response: ChatResponse) -> PendingChat:
        pending = next(
            (
                item
                for item in self.session.state.requests
                if item.request.request_id == request_id
            ),
            None,
        )
        if pending is None:
            raise CoreError(code="missing_reservation")
        if (
            response.request_id != request_id
            or len(response.reply) > CONTEXT_CHARACTERS
        ):
            raise CoreError(code="invalid_response")
        if pending.response is not None and pending.response != response:
            raise CoreError(code="response_conflict")
        return pending

    @staticmethod
    def _reply(
        pending: PendingChat, response: ChatResponse, *, stale: bool
    ) -> StoredReply:
        reply = StoredReply(
            text=response.reply,
            conversation_id=pending.conversation_id,
        )
        match response:
            case TurnResponse():
                if not stale:
                    reply = reply.model_copy(update={"delivery": response.delivery})
            case ChatResponse():
                pass
            case _:
                assert_never(response)
        return reply
