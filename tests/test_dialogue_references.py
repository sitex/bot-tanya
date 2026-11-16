import time
from pathlib import Path

import pytest

from tanya_bot.dialogue_references import bound_refs
from tanya_bot.dialogue_sources import recent_selection
from tests.conftest import Harness
from tests.dialogue_integration_support import attach, final_response, tool_response
from tests.test_dialogue_provider_payload import mail_result


@pytest.mark.parametrize(
    "question",
    [
        "Explain the second one and draft a reply.",
        "Расскажи подробнее \u043e втором из выбранных писем.",
        "Прочитай второе письмо.",
    ],
)
@pytest.mark.anyio
async def test_owner_ordinal_overrides_wrong_model_ref_and_keeps_page(
    harness: Harness,
    tmp_path: Path,
    question: str,
) -> None:
    context = attach(harness, tmp_path)
    harness.wire.outcomes.extend(
        [
            tool_response("gmail_search_read", {"query": "", "limit": 2}),
            final_response(),
        ]
    )
    await context.rig.say("Read my recent mail")
    first = context.store.roots()[0]
    assert first.selection is not None
    wrong = first.selection.result.sources[0].ref
    harness.wire.outcomes.extend(
        [
            tool_response("gmail_read", {"refs": [wrong]}),
            final_response("The second message"),
        ]
    )
    await context.rig.say(question)
    assert context.mail.paths[-1].endswith("/messages/second_mail")
    latest = max(context.store.roots(), key=lambda root: root.created_at)
    assert latest.tool_result is not None
    assert latest.tool_result.sources[0].position == 2
    assert latest.selection is not None
    assert latest.selection.root_id == first.selection.root_id
    assert [source.message_id for source in latest.selection.result.sources] == [
        "first_mail",
        "second_mail",
    ]
    assert len(harness.wire.calls) == 4

    legacy = latest.selection.model_copy(
        update={"root_id": latest.turn.chat.request_id, "result": latest.tool_result}
    )
    _ = context.store.mutate(
        latest.turn.chat.request_id,
        lambda root: root.model_copy(update={"selection": legacy}),
    )
    recovered = recent_selection(context.store, 42, 0, time.time())
    assert recovered is not None
    assert [source.message_id for source in recovered.result.sources] == [
        "first_mail",
        "second_mail",
    ]


@pytest.mark.parametrize(
    "question",
    [
        "The message about Friday",
        "Письмо про пятницу",
        "Compare the first and third",
        "The second and M000000000003",
        "Read email from my second boss",
        "Прочитай письмо моей второй жены",
        "Read the second email from Google",
        "Прочитай второе письмо от Google",
    ],
)
def test_ambiguous_nonordinal_or_explicit_reference_keeps_model_binding(
    question: str,
) -> None:
    sources = mail_result().sources
    chosen = (sources[2].ref,)
    assert bound_refs(question, chosen, sources) == chosen
