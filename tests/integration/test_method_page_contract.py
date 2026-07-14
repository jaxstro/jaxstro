"""Researcher-first contracts for current change and approximation methods."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[2] / "docs"

REQUIRED_HEADINGS = (
    "## The question this method answers",
    "## Before computation: what should be true?",
    "## Define the mathematical objects",
    "## Derive the method",
    "## What the algorithm actually does",
    "## What JAX differentiates",
    "## Using it in Jaxstro",
    "## How to audit the result",
    "## Where the claim stops",
    "## Connected ideas",
)

PAGE_SPECS = {
    "change-constraints-evolution/autodiff.md": (
        "jaxstro.numerics.autodiff",
        "../../50-api/change-constraints/autodiff.md",
        ("eq-autodiff-jvp", "eq-autodiff-vjp", "eq-autodiff-adjoint"),
    ),
    "change-constraints-evolution/rootfinding.md": (
        "jaxstro.numerics.rootfinding",
        "../../50-api/change-constraints/rootfinding.md",
        (
            "eq-root-bracket-invariant",
            "eq-root-safeguard-band",
            "eq-root-implicit-derivative",
        ),
    ),
    "change-constraints-evolution/optimization.md": (
        "jaxstro.numerics.optimization",
        "../../50-api/change-constraints/optimization.md",
        ("eq-gradient-descent", "eq-armijo-decrease", "eq-optimization-convergence"),
    ),
    "change-constraints-evolution/ode.md": (
        "jaxstro.numerics.ode",
        "../../50-api/change-constraints/ode.md",
        ("eq-ode-euler", "eq-ode-midpoint", "eq-ode-rk4", "eq-ode-local-global"),
    ),
    "approximation-integration/interpolation.md": (
        "jaxstro.numerics.interpolation",
        "../../50-api/approximation-integration/interpolation.md",
        ("eq-linear-interpolant", "eq-hermite-interpolant", "eq-pchip-slope"),
    ),
    "approximation-integration/regular-grid.md": (
        "jaxstro.numerics.regular_grid",
        "../../50-api/approximation-integration/regular-grid.md",
        ("eq-regular-grid-coordinate", "eq-multilinear-interpolant"),
    ),
    "approximation-integration/bsplines.md": (
        "jaxstro.numerics.splines",
        "../../50-api/approximation-integration/splines.md",
        (
            "eq-bspline-zero",
            "eq-cox-de-boor",
            "eq-bspline-derivative",
            "eq-bspline-roughness",
        ),
    ),
    "approximation-integration/cumulative-trapz.md": (
        "jaxstro.numerics.integration",
        "../../50-api/approximation-integration/integration.md",
        ("eq-trapezoid-panel", "eq-cumulative-trapezoid", "eq-trapezoid-error"),
    ),
    "approximation-integration/quadrature.md": (
        "jaxstro.numerics.quadrature",
        "../../50-api/approximation-integration/quadrature.md",
        (
            "eq-fixed-node-quadrature",
            "eq-gaussian-exactness",
            "eq-standard-normal-hermite",
        ),
    ),
}


def _page(relative: str) -> str:
    return (DOCS / "20-methods" / relative).read_text(encoding="utf-8")


def _first_python_block(relative: str) -> str:
    match = re.search(r"```python\n(?P<code>.*?)\n```", _page(relative), re.DOTALL)
    assert match is not None, relative
    return match.group("code")


@pytest.mark.parametrize("relative", PAGE_SPECS)
def test_current_method_pages_follow_the_shared_heading_contract(relative: str) -> None:
    text = _page(relative)
    headings = re.findall(r"^## .+$", text, flags=re.MULTILINE)

    assert tuple(headings) == REQUIRED_HEADINGS, relative


@pytest.mark.parametrize("relative", PAGE_SPECS)
def test_current_method_pages_expose_assumptions_boundaries_choice_and_links(
    relative: str,
) -> None:
    text = _page(relative)
    owner, api_route, _ = PAGE_SPECS[relative]

    for directive in ("important", "warning", "tip", "seealso"):
        assert f":::{{{directive}}}" in text, (relative, directive)
    assert f"from {owner} import" in text, relative
    assert "../../10-foundations/" in text, relative
    assert "../../30-representations/" in text, relative
    assert f"[]({api_route})" in text, relative
    assert "../../60-validation/" in text, relative


@pytest.mark.parametrize("relative", PAGE_SPECS)
def test_current_method_pages_keep_the_required_derivations_visible_and_labeled(
    relative: str,
) -> None:
    text = _page(relative)
    _, _, equation_labels = PAGE_SPECS[relative]

    derive_start = text.index("## Derive the method")
    algorithm_start = text.index("## What the algorithm actually does")
    derivation = text[derive_start:algorithm_start]
    for label in equation_labels:
        assert f":label: {label}" in derivation, (relative, label)
    assert "```{math}" in derivation, relative
    assert not re.search(r"````?\{dropdown\}", derivation), relative


def test_derivations_protect_the_scientific_relations_not_only_section_names() -> None:
    expected_relations = {
        "change-constraints-evolution/autodiff.md": (
            r"D f\(x\)\[v\]",
            r"w\^\\mathsf\{T\} D f\(x\)\[v\]",
        ),
        "change-constraints-evolution/rootfinding.md": (
            r"f\(a_k\).*f\(b_k\).*\\le 0",
            r"\\frac\{d x\^\\star\}\{d\\theta\}.*-.*\\frac\{\\partial f/\\partial\\theta\}",
        ),
        "change-constraints-evolution/optimization.md": (
            r"x_\{k\+1\}.*x_k.*-.*\\alpha_k.*\\nabla F\(x_k\)",
            r"F\(x_k \+ \\alpha_k p_k\).*\\le.*F\(x_k\)",
        ),
        "change-constraints-evolution/ode.md": (
            r"k_1.*k_2.*k_3.*k_4",
            r"O\(h\^\{p\+1\}\).*O\(h\^p\)",
        ),
        "approximation-integration/interpolation.md": (
            r"\(1-t\).*y_i.*t.*y_\{i\+1\}",
            r"h_\{00\}.*h_\{10\}.*h_\{01\}.*h_\{11\}",
        ),
        "approximation-integration/regular-grid.md": (
            r"t_d.*\\frac\{x_d-x_\{d,i_d\}\}",
            r"\\sum_\{\\boldsymbol\{b\}\\in\\\{0,1\\\}\^D\}",
        ),
        "approximation-integration/bsplines.md": (
            r"B_\{i,0\}.*\\begin\{cases\}",
            r"\\int.*\\left\[S\^\{\(m\)\}\(x\)\\right\]\^2.*dx",
        ),
        "approximation-integration/cumulative-trapz.md": (
            r"T_i.*\\frac\{x_\{i\+1\}-x_i\}\{2\}",
            r"O\(h\^3\).*O\(h\^2\)",
        ),
        "approximation-integration/quadrature.md": (
            r"\\sum_\{i=1\}\^n w_i p\(x_i\).*\\int.*p\(x\).*\\omega\(x\)",
            r"\\sqrt\{2\}.*z_i.*\\sqrt\{\\pi\}",
        ),
    }

    for relative, patterns in expected_relations.items():
        text = _page(relative)
        derive_start = text.index("## Derive the method")
        algorithm_start = text.index("## What the algorithm actually does")
        derivation = text[derive_start:algorithm_start]
        for pattern in patterns:
            assert re.search(pattern, derivation, flags=re.DOTALL), (relative, pattern)


def test_rootfinding_example_enables_precision_and_converges() -> None:
    relative = "change-constraints-evolution/rootfinding.md"
    block = _first_python_block(relative)
    assert block.index("enable_high_precision()") < block.index(
        "from jaxstro.numerics.rootfinding import"
    )

    namespace: dict[str, object] = {}
    exec(compile(block, relative, "exec"), namespace)

    result = namespace["result"]
    assert bool(result.bracketed)
    assert bool(result.converged)
