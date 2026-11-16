"""Bind unambiguous owner ordinals to the displayed page, never similar subjects."""

import re
from typing import Final

from tanya_bot.dialogue_models import MailSource
from tanya_bot.errors import ServiceError

ORDINALS: Final = (
    r"first|\u043f\u0435\u0440\u0432(?:\u044b\u0439|\u0430\u044f|\u043e\u0435|\u0443\u044e|\u043e\u0433\u043e|\u043e\u0439|\u043e\u043c|\u043e\u043c\u0443)",
    r"second|\u0432\u0442\u043e\u0440(?:\u043e\u0439|\u0430\u044f|\u043e\u0435|\u0443\u044e|\u043e\u0433\u043e|\u043e\u043c|\u043e\u043c\u0443)",
    r"third|\u0442\u0440\u0435\u0442(?:\u0438\u0439|\u044c\u044f|\u044c\u0435|\u044c\u044e|\u044c\u0435\u0433\u043e|\u044c\u0435\u043c|\u044c\u0435\u043c\u0443|\u044c\u0435\u0439)",
    r"fourth|\u0447\u0435\u0442\u0432[\u0435\u0451]\u0440\u0442(?:\u044b\u0439|\u0430\u044f|\u043e\u0435|\u0443\u044e|\u043e\u0433\u043e|\u043e\u0439|\u043e\u043c|\u043e\u043c\u0443)",
    r"fifth|\u043f\u044f\u0442(?:\u044b\u0439|\u0430\u044f|\u043e\u0435|\u0443\u044e|\u043e\u0433\u043e|\u043e\u0439|\u043e\u043c|\u043e\u043c\u0443)",
)


def bound_refs(
    text: str, proposed: tuple[str, ...], sources: tuple[MailSource, ...]
) -> tuple[str, ...]:
    """Normalize only an already-proposed read of one clear ordinal reference."""
    if re.search(r"\bM[a-f0-9]{12}\b", text):
        return proposed
    chosen = tuple(
        position
        for position, pattern in enumerate(ORDINALS, start=1)
        if re.search(r"\b(?:" + pattern + r")\b", text, flags=re.IGNORECASE)
    )
    if len(chosen) != 1:
        return proposed
    found = re.search(
        r"\b(?:" + ORDINALS[chosen[0] - 1] + r")\b", text, flags=re.IGNORECASE
    )
    if found is None:
        return proposed
    suffix = text[found.end() :].lstrip().casefold()
    page_hint = (
        re.search(
            r"\b(?:selected|listed|displayed|list|\u0432\u044b\u0431\u0440\u0430\u043d\w*|\u0441\u043f\u0438\u0441\u043a\w*)\b",
            text,
            flags=re.IGNORECASE,
        )
        is not None
    )
    noun_hint = (
        re.match(
            r"(?:one|email|mail|message|letter)\b|(?:\u043f\u0438\u0441\u044c\u043c|\u0441\u043e\u043e\u0431\u0449\u0435\u043d)\w*",
            suffix,
        )
        is not None
    )
    if not (
        noun_hint
        or not suffix
        or suffix[0] in ".,!?;:"
        or (page_hint and suffix.startswith("\u0438\u0437"))
    ):
        return proposed
    if not page_hint and re.search(r"\b(?:from|\u043e\u0442)\b", suffix):
        return proposed
    matched = next(
        (
            source
            for index, source in enumerate(sources, start=1)
            if (source.position or index) == chosen[0]
        ),
        None,
    )
    if matched is None:
        raise ServiceError(422, "mail_reference_invalid", "Select a displayed message.")
    return (matched.ref,)
