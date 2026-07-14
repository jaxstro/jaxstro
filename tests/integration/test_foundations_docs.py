"""Navigation and pedagogy contracts for the optional foundations route."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


def test_documentation_design_page_explains_the_evidence_first_structure() -> None:
    path = DOCS / "05-foundations" / "why-this-documentation-works-this-way.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    for phrase in (
        "# Why this documentation works this way",
        "research students",
        "prerequisite",
        "current preparedness",
        "predict → compute → audit → state the warranted claim",
        "post-hoc storytelling",
        "module",
        "reference",
    ):
        assert phrase in text


def test_choose_your_path_is_optional_ungraded_and_task_routed() -> None:
    index = (DOCS / "05-foundations" / "foundations.md").read_text(encoding="utf-8")
    chooser = (DOCS / "00-start-here" / "choose-your-path.md").read_text(
        encoding="utf-8"
    )
    assert "Foundations: the ideas we will not assume" in index
    for phrase in (
        "ungraded",
        "not a placement test",
        "complete first-principles path",
        "computation-first",
        "astronomy",
        "statistics and inference",
        "returning researcher",
    ):
        assert phrase in chooser
    assert "beginner" not in chooser.lower()
    assert "advanced learner" not in chooser.lower()
    assert "../05-foundations/linear-algebra-language-of-change.md" in chooser
    assert "../05-foundations/what-is-a-derivative.md" in chooser
    assert "../05-foundations/probability-and-distributions.md" in chooser


def test_homepage_names_four_routes_without_stale_door_count() -> None:
    homepage = (DOCS / "index.md").read_text(encoding="utf-8")
    assert "## Choose your route" in homepage
    assert "## Three doors in" not in homepage


def test_foundations_navigation_preserves_module_theory_section() -> None:
    myst = (DOCS / "myst.yml").read_text(encoding="utf-8")
    manifest = json.loads((DOCS / "route-manifest.json").read_text(encoding="utf-8"))
    for page, route in (
        ("05-foundations/foundations.md", "/foundations"),
        ("00-start-here/choose-your-path.md", "/choose-your-path"),
        (
            "05-foundations/why-this-documentation-works-this-way.md",
            "/why-this-documentation-works-this-way",
        ),
    ):
        assert myst.count(f"file: {page}") == 1
        assert manifest[page] == route
    assert myst.count("title: Theory — AD-safe numerics") == 1
    assert myst.count("file: 10-theory/rootfinding.md") == 1
    assert myst.count("file: 10-theory/distributions.md") == 1
