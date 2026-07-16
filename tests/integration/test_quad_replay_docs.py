from pathlib import Path

ROOT = Path("docs")


def test_differentiating_integral_page_contains_required_derivations():
    text = (
        ROOT / "20-methods/approximation-integration/differentiating-an-integral.md"
    ).read_text()
    for required in (
        "## The exact integral derivative",
        "## The accepted fixed-formula derivative",
        "## Why the two derivatives can differ",
        "## Moving bounds",
        "## Units of a derivative",
        "## A complete analytic, AD, and finite-difference audit",
        "```{math}",
        ":::{warning}",
        ":::{admonition}",
    ):
        assert required in text
    assert "course" not in text.lower()
    assert "instructor" not in text.lower()


def test_myst_toc_places_derivative_page_after_adaptive_quadrature():
    text = (ROOT / "myst.yml").read_text()
    adaptive = text.index("20-methods/approximation-integration/adaptive-quadrature.md")
    derivative = text.index(
        "20-methods/approximation-integration/differentiating-an-integral.md"
    )
    assert adaptive < derivative
    assert "60-validation/numerical/quadrature-replay-derivatives.md" in text


def test_docs_link_foundations_workflow_api_and_evidence():
    text = (
        ROOT / "20-methods/approximation-integration/differentiating-an-integral.md"
    ).read_text()
    for route in (
        "../../00-start-here/why-jax.md",
        "../../10-foundations/mathematical-objects/what-is-a-derivative.md",
        "../../40-workflows/differentiable-research/what-jax-differentiates.md",
        "../../30-representations/units-quantities/quantities.md",
        "../../50-api/approximation-integration/quad.md",
        "../../60-validation/numerical/quadrature-replay-derivatives.md",
    ):
        assert route in text
