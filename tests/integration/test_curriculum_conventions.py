"""Selective curriculum conventions for substantial scientific chapters."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHAPTERS = (
    "docs/10-theory/rootfinding.md",
    "docs/10-theory/interpolation.md",
    "docs/10-theory/regular-grid.md",
    "docs/10-theory/bsplines.md",
    "docs/10-theory/linear-algebra.md",
    "docs/10-theory/distributions.md",
    "docs/10-theory/spatial.md",
    "docs/10-theory/ode.md",
    "docs/10-theory/quadrature.md",
    "docs/10-theory/quantities.md",
    "docs/20-architecture/spectra-data-architecture.md",
)


def test_substantial_chapters_state_objectives_and_active_learning_prompt() -> None:
    for relative_path in CHAPTERS:
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert "## Learning objectives" in text, relative_path
        assert ("Predict → compute → audit" in text) or (
            "Concept check" in text
        ), relative_path


def test_homepage_names_scientific_capabilities_without_removing_entry_doors() -> None:
    text = (REPO_ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    for phrase in (
        "Explicit quantities and conventions",
        "Auditable numerical maps",
        "Events, equilibria, and inverse mappings",
        "Differentiable tabulated models",
        "Provenance-backed claims",
        "## Three doors in",
    ):
        assert phrase in text
