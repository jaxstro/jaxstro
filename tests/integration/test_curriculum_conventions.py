"""Selective curriculum conventions for substantial scientific chapters."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHAPTERS = (
    "docs/20-methods/change-constraints-evolution/rootfinding.md",
    "docs/20-methods/approximation-integration/interpolation.md",
    "docs/20-methods/approximation-integration/regular-grid.md",
    "docs/20-methods/approximation-integration/bsplines.md",
    "docs/20-methods/linear-structure/linear-algebra.md",
    "docs/20-methods/probability-sampling/distributions.md",
    "docs/20-methods/discrete-space/spatial.md",
    "docs/20-methods/change-constraints-evolution/ode.md",
    "docs/20-methods/approximation-integration/quadrature.md",
    "docs/30-representations/units-quantities/quantities.md",
    "docs/30-representations/spectra-atmospheres/spectra-data-architecture.md",
)


def test_substantial_chapters_state_objectives_and_active_learning_prompt() -> None:
    for relative_path in CHAPTERS:
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        if "30-representations" in relative_path:
            assert "## Representation contract" in text, relative_path
            assert "Use this page when" in text, relative_path
        else:
            assert "## Learning objectives" in text, relative_path
            assert ("Predict → compute → audit" in text) or ("Concept check" in text), (
                relative_path
            )


def test_homepage_names_scientific_capabilities_and_all_entry_routes() -> None:
    text = (REPO_ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    for phrase in (
        "Explicit quantities and conventions",
        "Auditable numerical maps",
        "Events, equilibria, and inverse mappings",
        "Differentiable tabulated models",
        "Provenance-backed claims",
        "## Choose your route",
        "Rebuild the foundations",
        "./10-foundations/foundations.md",
        "Learn the methods",
        "./20-methods/methods.md",
        "Audit the evidence",
        "./60-validation/index.md",
        "Look up the API",
        "./40-api/index.md",
    ):
        assert phrase in text


def test_regular_grid_activity_separates_reject_from_gradient_policies() -> None:
    text = (
        REPO_ROOT
        / "docs"
        / "20-methods"
        / "approximation-integration"
        / "regular-grid.md"
    ).read_text(encoding="utf-8")

    assert "eager failure for `reject`" in text
    assert "not a differentiable boundary rule" in text
    assert "gradient just outside a grid for clamp, fill,\nand reject" not in text
