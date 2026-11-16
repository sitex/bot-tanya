"""At-most-once voice generation and durable artifact delivery hooks."""

from uuid import uuid4

import anyio

from tanya_bot.telegram_conversations import voice_pending
from tanya_bot.telegram_core_session import CoreSession
from tanya_bot.telegram_keyboards import keyboard
from tanya_bot.telegram_models import (
    VOICE_TEXT_CHARACTERS,
    StoredReply,
    VoiceGateway,
    VoiceRequest,
)
from tanya_bot.telegram_store import CoreError


async def voice_reply(
    session: CoreSession,
    reply: StoredReply,
    gateway: VoiceGateway,
    *,
    recover: bool = False,
) -> None:
    """Claim before generation; recovery can only retrieve an existing artifact."""
    request = reply.voice_request
    if (
        reply.receipt_only
        or reply.conversation_id != session.state.conversation_id
        or (not recover and (request is not None or reply.voice_cancelled))
    ):
        return
    if request is None and len(reply.text) > VOICE_TEXT_CHARACTERS:
        return
    if request is None:
        if recover or voice_pending(session.state):
            return
        request = VoiceRequest(
            request_id=uuid4(),
            text=reply.text,
            voice=session.state.settings.profile.voice,
            delivery=reply.delivery,
        )
        reply = session.available(reply.model_copy(update={"voice_request": request}))
        session.replace(reply)
    result = reply.voice_result
    if result is None:
        result = (
            await gateway.recover(request)
            if recover
            else await gateway.generate(request)
        )
        await anyio.lowlevel.checkpoint()
        if result is None:
            return
        if result.request_id != request.request_id:
            raise CoreError(code="invalid_voice_response")
        reply = reply.model_copy(update={"voice_result": result})
        session.replace(reply)
    if not reply.voice_delivery_attempted:
        session.replace(reply.model_copy(update={"voice_delivery_attempted": True}))
        _ = await session.transport.send_voice(session.state.owner_id, result)
    if reply.message_id is not None:
        await session.transport.edit_keyboard(
            session.state.owner_id,
            reply.message_id,
            keyboard(reply, session.state.settings.language),
        )
