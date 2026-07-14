"""Public documentation language and mathematical-source contracts."""

from __future__ import annotations

import json
import re
from pathlib import Path

DOCS = Path(__file__).parents[2] / "docs"

FORBIDDEN_TERMS = re.compile(
    r"\b(course|courses|curriculum|instructor|instructors|"
    r"teaching assistant|teaching assistants|assessment rubric)\b",
    re.IGNORECASE,
)
FORBIDDEN_UNICODE = {
    "\u2192",
    "\u2190",
    "\u2194",
    "\u2264",
    "\u2265",
    "\u2260",
    "\u2248",
    "\u2208",
    "\u2209",
    "\u221e",
    "\u2211",
    "\u220f",
    "\u221a",
    "\u2202",
    "\u2207",
    "\u00d7",
    "\u00b7",
    "\u2212",
    "\u2013",
    "\u2014",
}


def test_routed_pages_use_research_software_language_and_latex_math() -> None:
    manifest = json.loads((DOCS / "route-manifest.json").read_text())
    for relative in manifest:
        if not relative.endswith(".md"):
            continue
        text = (DOCS / relative).read_text(encoding="utf-8")
        assert FORBIDDEN_TERMS.search(text) is None, relative
        assert not (set(text) & FORBIDDEN_UNICODE), relative
