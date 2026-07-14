"""Public documentation language and mathematical-source contracts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

DOCS = Path(__file__).parents[2] / "docs"

FORBIDDEN_TERMS = re.compile(
    r"\b(course|courses|curriculum|instructor|instructors|"
    r"teaching assistant|teaching assistants|assessment rubric)\b",
    re.IGNORECASE,
)
ALLOWED_NON_ASCII_OCCURRENCES = {
    # Diataxis is a proper name. These are the four exact public occurrences.
    "70-project/decisions/0005-diataxis-docs-with-adr-meta.md": (
        "Di\u00e1taxis",
        "Di\u00e1taxis",
        "Di\u00e1taxis",
        "Di\u00e1taxis",
    ),
}


def _unexpected_non_ascii(relative: str, text: str) -> list[str]:
    remaining = text
    violations = []
    for literal in ALLOWED_NON_ASCII_OCCURRENCES.get(relative, ()):
        if literal not in remaining:
            violations.append(f"missing allowlisted occurrence {literal!r}")
            continue
        remaining = remaining.replace(literal, "", 1)
    violations.extend(
        f"{character!r} (U+{ord(character):04X})"
        for character in sorted(set(remaining))
        if not character.isascii()
    )
    return violations


@pytest.mark.parametrize(
    "text",
    [
        "curly \u201cquote\u201d",
        "Unicode ellipsis\u2026",
        "real space \u211d",
        "down \u2193 then right \u2192",
        "generic fa\u00e7ade prose",
    ],
    ids=("curly-quotes", "ellipsis", "real-symbol", "arrows", "accented-prose"),
)
def test_non_ascii_prose_and_math_are_rejected(text: str) -> None:
    assert _unexpected_non_ascii("fixture.md", text)


def test_non_ascii_allowlist_is_literal_and_file_scoped() -> None:
    relative = "70-project/decisions/0005-diataxis-docs-with-adr-meta.md"
    exact_occurrences = " ".join(ALLOWED_NON_ASCII_OCCURRENCES[relative])
    assert not _unexpected_non_ascii(relative, exact_occurrences)
    assert _unexpected_non_ascii("another-file.md", exact_occurrences)


def test_routed_pages_use_research_software_language_and_latex_math() -> None:
    manifest = json.loads((DOCS / "route-manifest.json").read_text())
    for relative in manifest:
        if not relative.endswith(".md"):
            continue
        text = (DOCS / relative).read_text(encoding="utf-8")
        assert FORBIDDEN_TERMS.search(text) is None, relative
        violations = _unexpected_non_ascii(relative, text)
        assert not violations, f"{relative}: {violations}"
