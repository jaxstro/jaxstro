"""Dependency-light provenance-card validation and deterministic MyST rendering.

The installed API accepts already-parsed mappings. File formats, filesystem policy,
and YAML parsing belong to repository or application tooling rather than this module.
"""

from __future__ import annotations

import html
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

CardStatus = Literal["verified", "needs-check", "unverifiable-scanned"]

ALLOWED_STATUSES: tuple[CardStatus, ...] = (
    "verified",
    "needs-check",
    "unverifiable-scanned",
)

REQUIRED_FIELDS = (
    "id",
    "title",
    "summary",
    "scope",
    "conventions",
    "sources",
    "code_refs",
    "validation",
    "status",
    "deviations",
)

_SOURCE_FIELDS = ("reference", "locator", "supports")
_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class ProvenanceCardError(ValueError):
    """Raised when a provenance card violates the shared schema."""


@dataclass(frozen=True)
class SourceReference:
    """A source pointer with an exact locator and bounded supported claim."""

    reference: str
    locator: str
    supports: str


@dataclass(frozen=True)
class ProvenanceCard:
    """Validated, immutable provenance-card data."""

    id: str
    title: str
    summary: str
    scope: str
    conventions: tuple[str, ...]
    sources: tuple[SourceReference, ...]
    code_refs: tuple[str, ...]
    validation: tuple[str, ...]
    status: CardStatus
    deviations: tuple[str, ...]


def _error(context: str, message: str) -> ProvenanceCardError:
    prefix = f"{context}: " if context else ""
    return ProvenanceCardError(prefix + message)


def _require_string(value: object, *, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(context, f"{field} must be a non-empty string")
    return value.strip()


def _require_string_sequence(
    value: object, *, field: str, context: str
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _error(context, f"{field} must be a sequence of strings")
    return tuple(
        _require_string(item, field=f"{field}[{index}]", context=context)
        for index, item in enumerate(value)
    )


def _require_reference_format(
    refs: tuple[str, ...], *, field: str, expected: str, context: str
) -> None:
    malformed = [ref for ref in refs if ref.count("::") < 1]
    if malformed:
        raise _error(context, f"{field} must use {expected}: {malformed}")


def validate_card(raw: Mapping[str, object], *, context: str = "") -> ProvenanceCard:
    """Validate an already-parsed mapping and return an immutable card.

    The function deliberately performs no file I/O and imports no serialization
    library. Callers choose YAML, JSON, TOML, or another mapping-producing format.
    """

    if not isinstance(raw, Mapping):
        raise _error(context, "card must be a mapping")
    missing = [field for field in REQUIRED_FIELDS if field not in raw]
    if missing:
        raise _error(context, f"card missing required fields: {missing}")

    card_id = _require_string(raw["id"], field="id", context=context)
    if not _ID_PATTERN.fullmatch(card_id):
        raise _error(context, "id must be a lowercase kebab-case slug")

    status = _require_string(raw["status"], field="status", context=context)
    if status not in ALLOWED_STATUSES:
        raise _error(context, f"card {card_id!r} has unknown status {status!r}")

    raw_sources = raw["sources"]
    if isinstance(raw_sources, (str, bytes)) or not isinstance(raw_sources, Sequence):
        raise _error(context, "sources must be a sequence of mappings")
    sources: list[SourceReference] = []
    for index, source in enumerate(raw_sources):
        source_context = f"{context} card {card_id} source[{index}]".strip()
        if not isinstance(source, Mapping):
            raise _error(source_context, "source must be a mapping")
        missing_source = [field for field in _SOURCE_FIELDS if field not in source]
        if missing_source:
            raise _error(
                source_context, f"source missing required fields: {missing_source}"
            )
        sources.append(
            SourceReference(
                reference=_require_string(
                    source["reference"], field="reference", context=source_context
                ),
                locator=_require_string(
                    source["locator"], field="locator", context=source_context
                ),
                supports=_require_string(
                    source["supports"], field="supports", context=source_context
                ),
            )
        )

    code_refs = _require_string_sequence(
        raw["code_refs"], field="code_refs", context=context
    )
    validation = _require_string_sequence(
        raw["validation"], field="validation", context=context
    )
    _require_reference_format(
        code_refs,
        field="code reference",
        expected="path::qualname",
        context=context,
    )
    _require_reference_format(
        validation,
        field="validation reference",
        expected="path::node",
        context=context,
    )

    if status == "verified":
        if not sources:
            raise _error(context, "verified card requires at least one source")
        if not code_refs:
            raise _error(context, "verified card requires at least one code reference")
        if not validation:
            raise _error(
                context, "verified card requires at least one validation reference"
            )

    return ProvenanceCard(
        id=card_id,
        title=_require_string(raw["title"], field="title", context=context),
        summary=_require_string(raw["summary"], field="summary", context=context),
        scope=_require_string(raw["scope"], field="scope", context=context),
        conventions=_require_string_sequence(
            raw["conventions"], field="conventions", context=context
        ),
        sources=tuple(sources),
        code_refs=code_refs,
        validation=validation,
        status=status,
        deviations=_require_string_sequence(
            raw["deviations"], field="deviations", context=context
        ),
    )


def _as_card(card: Mapping[str, object] | ProvenanceCard) -> ProvenanceCard:
    return card if isinstance(card, ProvenanceCard) else validate_card(card)


def _render_reference(reference: str) -> str:
    """Escape a registry source reference for an HTML link target."""
    return html.escape(reference, quote=True)


def render_card(card: Mapping[str, object] | ProvenanceCard) -> str:
    """Render one validated card as deterministic MyST Markdown."""

    item = _as_card(card)
    lines = [
        f"(card-{item.id})=",
        f"## {item.title}",
        "",
        f"**Status:** `{item.status}`",
        "",
        item.summary,
        "",
        "### Scope",
        "",
        item.scope,
        "",
        "### Conventions",
        "",
    ]
    lines.extend(f"- {convention}" for convention in item.conventions)
    if not item.conventions:
        lines.append("- none recorded")

    lines.extend(["", "### Sources", ""])
    for source in item.sources:
        reference = _render_reference(source.reference)
        lines.append(
            f'- <a href="{reference}">source</a> - {source.supports}. '
            f"*Locator:* `{source.locator}`"
        )
    if not item.sources:
        lines.append("- none recorded")

    lines.extend(["", "### Code & validation", ""])
    lines.extend(f"- code: `{ref}`" for ref in item.code_refs)
    lines.extend(f"- validation: `{ref}`" for ref in item.validation)
    if not item.code_refs and not item.validation:
        lines.append("- none recorded")

    if item.deviations:
        lines.extend(
            [
                "",
                ":::{admonition} Deviations from the source",
                ":class: caution",
            ]
        )
        lines.extend(f"- {deviation}" for deviation in item.deviations)
        lines.append(":::")

    return "\n".join(lines).rstrip() + "\n"


def render_family(
    family: str,
    cards: Sequence[Mapping[str, object] | ProvenanceCard],
    *,
    title: str | None = None,
) -> str:
    """Render a deterministically ordered family page."""

    validated = sorted((_as_card(card) for card in cards), key=lambda item: item.id)
    ids = [card.id for card in validated]
    if len(ids) != len(set(ids)):
        raise ProvenanceCardError(f"{family}: duplicate card ids")
    display_title = title or family.replace("_", " ").title()
    lines = [
        "---",
        f'title: "{display_title}"',
        "description: >-",
        f"  Generated provenance cards for {display_title.lower()}.",
        "---",
        "",
        "<!-- GENERATED by scripts/build_provenance_registry.py; DO NOT EDIT. -->",
        "",
        f"# {display_title}",
        "",
    ]
    if validated:
        lines.append("\n\n".join(render_card(card).rstrip() for card in validated))
    else:
        lines.append("No source-verified cards are registered for this family yet.")
    return "\n".join(lines).rstrip() + "\n"


def render_index(
    families: Mapping[str, Sequence[Mapping[str, object] | ProvenanceCard]],
    *,
    family_titles: Mapping[str, str] | None = None,
) -> str:
    """Render the deterministic registry index."""

    titles = family_titles or {}
    lines = [
        "---",
        'title: "Provenance cards"',
        'description: "Generated index of source-backed jaxstro provenance cards."',
        "---",
        "",
        "<!-- GENERATED by scripts/build_provenance_registry.py; DO NOT EDIT. -->",
        "",
        "# Provenance cards",
        "",
    ]
    for family in sorted(families):
        title = titles.get(family, family.replace("_", " ").title())
        count = len(families[family])
        lines.append(
            f"- [{title}]({family}.md) - {count} card{'s' if count != 1 else ''}"
        )
    return "\n".join(lines).rstrip() + "\n"


def render_registry(
    families: Mapping[str, Sequence[Mapping[str, object] | ProvenanceCard]],
    *,
    family_titles: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Render all family pages and the index with stable ordering and bytes."""

    titles = family_titles or {}
    rendered = {
        f"{family}.md": render_family(
            family, families[family], title=titles.get(family)
        )
        for family in sorted(families)
    }
    rendered["index.md"] = render_index(families, family_titles=titles)
    return rendered


__all__ = [
    "ALLOWED_STATUSES",
    "CardStatus",
    "ProvenanceCard",
    "ProvenanceCardError",
    "REQUIRED_FIELDS",
    "SourceReference",
    "render_card",
    "render_family",
    "render_index",
    "render_registry",
    "validate_card",
]
