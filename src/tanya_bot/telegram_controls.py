"""Cache recovery and message-bound callbacks under the controller transaction."""

from tanya_bot.i18n import localize
from tanya_bot.telegram_completion import ReplyCompletion
from tanya_bot.telegram_core_actions import perform_action
from tanya_bot.telegram_core_session import CoreSession
from tanya_bot.telegram_core_voice import voice_reply
from tanya_bot.telegram_models import (
    CALLBACK_BYTES,
    CompanionGateway,
    IncomingCallback,
    StoredReply,
    VoiceGateway,
)


async def recover_results(
    session: CoreSession, gateway: CompanionGateway, voice: VoiceGateway | None
) -> None:
    """Read saved outcomes only, including explicit legacy cancelled recovery."""
    for pending in session.state.requests:
        if pending.response is None:
            cached = await gateway.recover(pending.request)
            if cached is not None:
                await ReplyCompletion(session, voice).complete(
                    pending.request.request_id, cached, recover=True
                )
    for reply in session.state.replies:
        if reply.conversation_id != session.state.conversation_id:
            continue
        if not reply.delivery_attempted:
            await session.deliver(reply)
        if voice is not None and reply.voice_request is not None:
            await voice_reply(session, reply, voice, recover=True)
    unresolved = sum(
        item.response is None and item.conversation_id == session.state.conversation_id
        for item in session.state.requests
    )
    archived = sum(
        item.response is None and item.conversation_id != session.state.conversation_id
        for item in session.state.requests
    )
    unresolved += sum(
        item.conversation_id == session.state.conversation_id
        and item.voice_request is not None
        and item.voice_result is None
        for item in session.state.replies
    )
    archived += sum(
        item.conversation_id != session.state.conversation_id
        and item.voice_request is not None
        and item.voice_result is None
        for item in session.state.replies
    )
    language = session.state.settings.language
    status = (
        "The result is still unknown; no repeat was sent."
        if unresolved
        else "Saved results check complete."
    )
    archive_note = f"Unknown requests archive: {archived}."
    await session.say(
        f"{localize(status, language)}\n{localize(archive_note, language)}"
    )


async def handle_callback(
    session: CoreSession, event: IncomingCallback, voice: VoiceGateway | None
) -> None:
    """Validate saved token, message, conversation and menu before any effect."""
    reply = await claim_callback(session, event)
    if reply is not None:
        await perform_action(session, reply, event.data.partition(":")[2], voice)


async def claim_callback(
    session: CoreSession, event: IncomingCallback
) -> StoredReply | None:
    """Claim a callback durably only against its saved message and conversation."""
    language = session.state.settings.language
    if len(event.data.encode()) > CALLBACK_BYTES or not session.claim(
        f"callback:{event.update_id}"
    ):
        await session.transport.ack(
            event.callback_id, localize("Already processed or unavailable.", language)
        )
        return None
    token, _, action = event.data.partition(":")
    reply = next(
        (
            item
            for item in session.state.replies
            if item.token == token
            and item.message_id == event.message_id
            and item.conversation_id == session.state.conversation_id
        ),
        None,
    )
    if reply is None or action not in reply.actions:
        await session.transport.ack(
            event.callback_id, localize("Button expired.", language)
        )
        return None
    if (
        action in {"new_confirm", "new_stay"}
        and reply.token != session.state.new_confirmation_token
    ):
        await session.transport.ack(
            event.callback_id, localize("Button expired.", language)
        )
        return None
    if reply.setting_field is not None and (
        reply.token != session.state.settings_menu_token
        or reply.setting_field != session.state.editing
    ):
        await session.transport.ack(
            event.callback_id, localize("Menu expired.", language)
        )
        return None
    await session.transport.ack(event.callback_id, localize("Accepted.", language))
    return reply
