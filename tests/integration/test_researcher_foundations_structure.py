"""Researcher-first structure and route contracts for Foundations."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"

FOUNDATION_PAGES = (
    "10-foundations/foundations.md",
    "10-foundations/mathematical-objects/functions-units-scales.md",
    "10-foundations/mathematical-objects/linear-algebra-language-of-change.md",
    "10-foundations/mathematical-objects/what-is-a-derivative.md",
    "10-foundations/mathematical-objects/probability-and-distributions.md",
    "10-foundations/models-and-computation/what-is-a-model.md",
    "10-foundations/models-and-computation/models-inference-information.md",
    "10-foundations/models-and-computation/sensitivity-conditioning-identifiability.md",
    "10-foundations/models-and-computation/from-relations-to-differentiable-programs.md",
)

PUBLIC_ROUTES = {
    "10-foundations/foundations.md": "/foundations",
    "10-foundations/mathematical-objects/functions-units-scales.md": (
        "/functions-units-scales"
    ),
    "10-foundations/mathematical-objects/linear-algebra-language-of-change.md": (
        "/linear-algebra-language-of-change"
    ),
    "10-foundations/mathematical-objects/what-is-a-derivative.md": (
        "/what-is-a-derivative"
    ),
    "10-foundations/mathematical-objects/probability-and-distributions.md": (
        "/probability-and-distributions"
    ),
    "10-foundations/models-and-computation/what-is-a-model.md": "/what-is-a-model",
    "10-foundations/models-and-computation/models-inference-information.md": (
        "/models-inference-information"
    ),
    (
        "10-foundations/models-and-computation/"
        "sensitivity-conditioning-identifiability.md"
    ): "/sensitivity-conditioning-identifiability",
    (
        "10-foundations/models-and-computation/"
        "from-relations-to-differentiable-programs.md"
    ): "/from-relations-to-differentiable-programs",
}


def _body_after_frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        return text
    return text.split("---\n", 2)[-1]


def test_foundations_use_two_semantic_subsections() -> None:
    myst = (DOCS / "myst.yml").read_text(encoding="utf-8")
    assert myst.count("title: Mathematical objects") == 1
    assert myst.count("title: Models and computation") == 1
    for page in FOUNDATION_PAGES:
        assert (DOCS / page).is_file(), page
        assert myst.count(f"file: {page}") == 1


def test_foundations_landing_explains_optional_connected_routes() -> None:
    landing = " ".join(
        (DOCS / FOUNDATION_PAGES[0]).read_text(encoding="utf-8").lower().split()
    )
    for phrase in (
        "optional does not mean unimportant",
        "connected concepts, not prerequisites to pass",
        "proceed linearly",
        "enter from any method page",
        "return when an audit exposes a conceptual gap",
        "mathematical objects, executable programs, and warranted scientific claims",
    ):
        assert phrase in landing


def test_each_foundation_page_opens_with_a_use_sentence() -> None:
    for page in FOUNDATION_PAGES:
        body = _body_after_frontmatter((DOCS / page).read_text(encoding="utf-8"))
        opening = "\n".join(body.strip().splitlines()[:12])
        assert "Use this page when" in opening, page


def test_foundations_source_is_ascii_and_researcher_facing() -> None:
    forbidden_framing = re.compile(
        r"\b(course|class|lecture|instructor|grade|graded|grading)\b",
        flags=re.IGNORECASE,
    )
    for page in FOUNDATION_PAGES:
        text = (DOCS / page).read_text(encoding="utf-8")
        assert text.isascii(), page
        assert forbidden_framing.search(text) is None, page


def test_old_foundations_tree_and_philosophy_route_are_retired() -> None:
    assert not tuple((DOCS / "05-foundations").glob("*.md"))
    manifest = json.loads((DOCS / "route-manifest.json").read_text(encoding="utf-8"))
    assert "/why-this-documentation-works-this-way" not in manifest.values()
    assert not any(page.startswith("05-foundations/") for page in manifest)


def test_meaningful_foundation_routes_are_preserved() -> None:
    manifest = json.loads((DOCS / "route-manifest.json").read_text(encoding="utf-8"))
    for page, route in PUBLIC_ROUTES.items():
        assert manifest[page] == route
    assert manifest[FOUNDATION_PAGES[0]] == "/foundations"
    assert (
        manifest[
            "10-foundations/models-and-computation/"
            "from-relations-to-differentiable-programs.md"
        ]
        == "/from-relations-to-differentiable-programs"
    )
