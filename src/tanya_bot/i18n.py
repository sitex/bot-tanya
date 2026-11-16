"""Translate only local interface text; conversation and mail content stay verbatim."""

import re
from importlib.resources import files
from string import Formatter
from typing import ClassVar, Final, Literal

from pydantic import BaseModel, ConfigDict


class Template(BaseModel):
    """A packaged message template, never code supplied by a user or model."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    en: str
    ru: str


class Catalog(BaseModel):
    """Load the static catalog at one typed package-resource boundary."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    strings: dict[str, str]
    templates: tuple[Template, ...]


CATALOG: Final = Catalog.model_validate_json(
    files("tanya_bot").joinpath("locales.json").read_text(encoding="utf-8")
)


def ui_language(language: str) -> Literal["en", "ru"]:
    """Other conversation languages keep the documented English interface."""
    return "ru" if language.casefold().startswith("ru") else "en"


def _pattern(template: Template) -> re.Pattern[str]:
    parts: list[str] = []
    for literal, name, _, _ in Formatter().parse(template.en):
        parts.append(re.escape(literal))
        if name is not None:
            parts.append("(?P<" + name + ">.*?)")
    return re.compile("".join(parts), re.DOTALL)


PATTERNS: Final = tuple((_pattern(item), item.ru) for item in CATALOG.templates)


def localize(text: str, language: str) -> str:
    """Fixed translations and anchored templates preserve interpolated user data."""
    if ui_language(language) != "ru":
        return text
    exact = CATALOG.strings.get(text)
    if exact is not None:
        return exact
    for pattern, replacement in PATTERNS:
        matched = pattern.fullmatch(text)
        if matched is not None:
            return replacement.format_map(matched.groupdict())
    if "\n" in text:
        return "\n".join(localize(line, language) for line in text.split("\n"))
    return text
