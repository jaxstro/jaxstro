"""Contracts for science-question routing across Jaxstro modules."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE = (
    REPO_ROOT
    / "docs"
    / "40-workflows"
    / "differentiable-research"
    / "science-patterns.md"
)


def test_science_patterns_cover_the_required_research_questions() -> None:
    text = PAGE.read_text(encoding="utf-8")
    headings = (
        "## Locate an event or equilibrium",
        "## Differentiate a certified equilibrium",
        "## Accumulate a conserved or integrated quantity",
        "## Interpolate a tabulated physical model",
        "## Cross a limiting distribution parameter smoothly",
        "## Transform coordinates, units, and spectra",
        "## Find local spatial interactions",
        "## Connect model PyTrees to inference parameters",
        "## Preserve provenance from artifact to claim",
    )
    for heading in headings:
        assert heading in text
    assert text.count("**Question.**") == len(headings)
    assert text.count("**Transform boundary.**") == len(headings)
    assert text.count("**Ownership.**") == len(headings)


def test_science_patterns_route_to_theory_api_and_validation() -> None:
    text = PAGE.read_text(encoding="utf-8")

    assert "[](../../20-methods/change-constraints-evolution/rootfinding.md)" in text
    assert "[](../../20-methods/probability-sampling/distributions.md)" in text
    assert "[](../../20-methods/discrete-space/spatial.md)" in text
    assert "[](../../50-api/api.md)" in text
    assert "[](../../60-validation/validation.md)" in text


def test_science_patterns_page_is_in_navigation_and_manifest() -> None:
    myst = (REPO_ROOT / "docs" / "myst.yml").read_text(encoding="utf-8")
    manifest = json.loads(
        (REPO_ROOT / "docs" / "route-manifest.json").read_text(encoding="utf-8")
    )

    path = "40-workflows/differentiable-research/science-patterns.md"
    assert myst.count(path) == 1
    assert manifest[path] == "/science-patterns"
