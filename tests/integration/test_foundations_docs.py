"""Navigation and pedagogy contracts for the optional foundations route."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


def test_foundations_and_usage_router_explain_the_evidence_first_structure() -> None:
    landing = (DOCS / "10-foundations" / "foundations.md").read_text(encoding="utf-8")
    router = (DOCS / "00-start-here" / "ways-to-use-these-docs.md").read_text(
        encoding="utf-8"
    )
    text = f"{landing}\n{router}"
    for phrase in (
        "# Foundations: the ideas we will not assume",
        "Optional does not mean unimportant",
        "predict -> compute -> audit -> state the warranted claim",
        "post-hoc storytelling",
        "The API reference records the exact supported surface",
        "independent evidence",
    ):
        assert phrase in text

    landing_prose = " ".join(landing.split())
    for phrase in (
        "Prior exposure does not guarantee that those concepts are active and "
        "connected for a new research problem",
        "calculus, statistics, linear algebra, programming, physical modeling, "
        "and inference separately",
        "Reconnecting them is substantive scientific work, not remediation",
    ):
        assert phrase in landing_prose


def test_choose_your_path_is_optional_research_routing() -> None:
    index = (DOCS / "10-foundations" / "foundations.md").read_text(encoding="utf-8")
    chooser = (DOCS / "00-start-here" / "choose-your-path.md").read_text(
        encoding="utf-8"
    )
    assert "Foundations: the ideas we will not assume" in index
    for phrase in (
        "route finder, not a gate",
        "complete first-principles path",
        "computation-first",
        "astronomy",
        "statistics and inference",
        "returning researcher",
    ):
        assert phrase in chooser
    assert "course" not in chooser.lower()
    assert "graded" not in chooser.lower()
    assert (
        "../10-foundations/mathematical-objects/"
        "linear-algebra-language-of-change.md" in chooser
    )
    assert "../10-foundations/mathematical-objects/what-is-a-derivative.md" in chooser
    assert (
        "../10-foundations/mathematical-objects/probability-and-distributions.md"
        in chooser
    )


def test_homepage_names_four_routes_without_stale_door_count() -> None:
    homepage = (DOCS / "index.md").read_text(encoding="utf-8")
    assert "## Choose your route" in homepage
    assert "## Three doors in" not in homepage


def test_foundations_navigation_preserves_module_theory_section() -> None:
    myst = (DOCS / "myst.yml").read_text(encoding="utf-8")
    manifest = json.loads((DOCS / "route-manifest.json").read_text(encoding="utf-8"))
    for page, route in (
        ("10-foundations/foundations.md", "/foundations"),
        ("00-start-here/choose-your-path.md", "/choose-your-path"),
    ):
        assert myst.count(f"file: {page}") == 1
        assert manifest[page] == route
    assert "/why-this-documentation-works-this-way" not in manifest.values()
    assert myst.count("title: Methods") == 1
    assert (
        myst.count("file: 20-methods/change-constraints-evolution/rootfinding.md") == 1
    )
    assert myst.count("file: 20-methods/probability-sampling/distributions.md") == 1
