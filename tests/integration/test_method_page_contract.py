"""Researcher-first contracts for current change and approximation methods."""

from __future__ import annotations

import re
import subprocess
import sys
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
            "eq-bspline-antiderivative",
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

EXACT_RELATIONS = {
    "change-constraints-evolution/optimization.md": (
        r"F(x_k+\alpha_kp_k)\leF(x_k)+c_1\alpha_k\nablaF(x_k)^\mathsf{T}p_k",
    ),
    "approximation-integration/regular-grid.md": (
        r"f_{\boldsymbol{i}+\boldsymbol{b}}\prod_{d=1}^{D}t_d^{b_d}(1-t_d)^{1-b_d}",
    ),
    "approximation-integration/bsplines.md": (
        r"B_{i,p}(x)=\frac{x-t_i}{t_{i+p}-t_i}B_{i,p-1}(x)+\frac{t_{i+p+1}-x}{t_{i+p+1}-t_{i+1}}B_{i+1,p-1}(x)",
        r"c'_i=p\frac{c_{i+1}-c_i}{t_{i+p+1}-t_{i+1}}",
        r"d_{i+1}-d_i=c_i\frac{t_{i+p+1}-t_i}{p+1}",
    ),
    "approximation-integration/quadrature.md": (
        r"\widetilde{w}_i=\frac{w_i}{\sqrt{\pi}}",
    ),
}

RELATION_MUTATIONS = {
    EXACT_RELATIONS["change-constraints-evolution/optimization.md"][0]: (
        r"F(x_k+\alpha_kp_k)\leF(x_k)+c_1\alpha_k\nablaF(x_k)^\mathsf{T}+p_k"
    ),
    EXACT_RELATIONS["approximation-integration/regular-grid.md"][0]: (
        r"f_{\boldsymbol{i}+\boldsymbol{b}}+\prod_{d=1}^{D}t_d^{b_d}(1-t_d)^{1-b_d}"
    ),
    EXACT_RELATIONS["approximation-integration/bsplines.md"][0]: (
        r"B_{i,p}(x)=\frac{x-t_i}{t_{i+p}-t_i}B_{i,p-1}(x)-\frac{t_{i+p+1}-x}{t_{i+p+1}-t_{i+1}}B_{i+1,p-1}(x)"
    ),
    EXACT_RELATIONS["approximation-integration/bsplines.md"][1]: (
        r"c'_i=p+\frac{c_{i+1}-c_i}{t_{i+p+1}-t_{i+1}}"
    ),
    EXACT_RELATIONS["approximation-integration/bsplines.md"][2]: (
        r"d_{i+1}-d_i=c_i+\frac{t_{i+p+1}-t_i}{p+1}"
    ),
    EXACT_RELATIONS["approximation-integration/quadrature.md"][0]: (
        r"\widetilde{w}_i=w_i\sqrt{\pi}"
    ),
}


def _page(relative: str) -> str:
    return (DOCS / "20-methods" / relative).read_text(encoding="utf-8")


def _first_python_block(relative: str) -> str:
    match = re.search(r"```python\n(?P<code>.*?)\n```", _page(relative), re.DOTALL)
    assert match is not None, relative
    return match.group("code")


def _derivation(relative: str, text: str | None = None) -> str:
    if text is None:
        text = _page(relative)
    derive_start = text.index("## Derive the method")
    algorithm_start = text.index("## What the algorithm actually does")
    return text[derive_start:algorithm_start]


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _missing_exact_relations_from_compact(
    relative: str, compact: str
) -> tuple[str, ...]:
    return tuple(
        relation
        for relation in EXACT_RELATIONS.get(relative, ())
        if relation not in compact
    )


def _missing_exact_relations(relative: str, text: str) -> tuple[str, ...]:
    return _missing_exact_relations_from_compact(
        relative, _compact(_derivation(relative, text))
    )


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

    derivation = _derivation(relative, text)
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
        derivation = _derivation(relative, text)
        for pattern in patterns:
            assert re.search(pattern, derivation, flags=re.DOTALL), (relative, pattern)


@pytest.mark.parametrize("relative", PAGE_SPECS)
def test_current_method_page_examples_execute_in_isolation(relative: str) -> None:
    block = _first_python_block(relative)
    completed = subprocess.run(
        [sys.executable, "-c", block],
        cwd=DOCS.parent,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, (
        relative,
        completed.stdout,
        completed.stderr,
    )


def test_rootfinding_example_enables_precision_before_owner_import() -> None:
    block = _first_python_block("change-constraints-evolution/rootfinding.md")
    assert block.index("enable_high_precision()") < block.index(
        "from jaxstro.numerics.rootfinding import"
    )


def test_exact_relations_are_mutation_resistant() -> None:
    for relative, relations in EXACT_RELATIONS.items():
        text = _page(relative)
        assert not _missing_exact_relations(relative, text), relative

        compact = _compact(_derivation(relative, text))
        for relation in relations:
            mutated = compact.replace(relation, RELATION_MUTATIONS[relation], 1)
            assert relation in _missing_exact_relations_from_compact(
                relative, mutated
            ), (
                relative,
                relation,
            )


def test_reviewed_runtime_boundaries_are_stated_explicitly() -> None:
    integration = " ".join(
        _page("approximation-integration/cumulative-trapz.md").split()
    )
    assert "supported default-last-axis paths" in integration
    assert "nondefault `trapz` axes are not currently supported" in integration
    assert "nonuniform multidimensional cumulative integration" in integration
    assert "direct width broadcasting" in integration

    interpolation = " ".join(
        _page("approximation-integration/interpolation.md").split()
    )
    assert "scalar one-dimensional `y` only" in interpolation
    assert "Array-valued payloads are supported by the linear, Hermite, and PCHIP" in (
        interpolation
    )
    assert "does not cleanly reject an array-valued `y`" in interpolation

    optimization = " ".join(
        _page("change-constraints-evolution/optimization.md").split()
    )
    assert "Huber is $C^1$ but not $C^2$" in optimization
    assert "curvature and Hessian" in optimization
    assert "$s_0>0$" in optimization
    assert "`scale_floor=1e-12`" in optimization

    rootfinding = " ".join(_page("change-constraints-evolution/rootfinding.md").split())
    assert "default `max_steps=50`" in rootfinding
    assert "default `max_steps=30`" in rootfinding
    assert r"\le x_{\mathrm{trial}}\le" in rootfinding

    ode = " ".join(_page("change-constraints-evolution/ode.md").split())
    assert "first-order RHS integrators" in ode
    assert "second-order velocity-Verlet surface" in ode
    assert "separable conservative system" in ode


def _run_runtime_shape_status_and_failure_probes() -> None:
    import jax
    import jax.numpy as jnp

    from jaxstro.numerics.autodiff import jvp, vjp
    from jaxstro.numerics.integration import cumulative_trapz, trapz
    from jaxstro.numerics.interpolation import (
        cubic_hermite_interp,
        monotone_cubic_interp,
        natural_cubic_spline_coeffs,
    )
    from jaxstro.numerics.ode import solve_fixed_step, velocity_verlet
    from jaxstro.numerics.optimization import armijo_backtracking
    from jaxstro.numerics.quadrature import gauss_laguerre_nodes
    from jaxstro.numerics.regular_grid import regular_grid_interp
    from jaxstro.numerics.rootfinding import safeguarded_bracketed_root
    from jaxstro.numerics.splines import bspline_eval, open_uniform_knots

    x = jnp.array([0.4, -0.2])
    tangent = jnp.array([0.3, 0.7])
    cotangent = jnp.array([1.2, -0.5])

    def function(value):
        return jnp.array([value[0] ** 2, value[0] * value[1]])

    _, pushed = jvp(function, x, tangent)
    _, pulled = vjp(function, x, cotangent)
    assert pushed.shape == cotangent.shape
    assert pulled.shape == tangent.shape
    assert jnp.allclose(jnp.vdot(cotangent, pushed), jnp.vdot(tangent, pulled))

    missing_root = safeguarded_bracketed_root(
        lambda value: value**2 + 1.0,
        -1.0,
        1.0,
        max_steps=8,
    )
    assert not bool(missing_root.bracketed)
    assert not bool(missing_root.converged)
    assert bool(jnp.isnan(missing_root.root))

    def objective(value):
        return 0.5 * jnp.sum(value**2)

    point = jnp.array([1.0])
    rejected = armijo_backtracking(
        objective,
        point,
        direction=point,
        grad=point,
        max_steps=4,
    )
    assert not bool(rejected.accepted)
    assert int(rejected.iterations) == 4

    with pytest.raises(ValueError, match="unknown fixed-step ODE method"):
        solve_fixed_step(
            lambda y, t: y + t,
            y0=jnp.ones(2),
            t0=0.0,
            dt=0.1,
            num_steps=2,
            method="bogus",  # type: ignore[arg-type]
        )
    verlet = velocity_verlet(
        lambda q, t: -q + 0.0 * t,
        q0=jnp.ones(2),
        v0=jnp.zeros(2),
        t0=0.0,
        dt=0.1,
        num_steps=3,
    )
    assert verlet.q.shape == verlet.v.shape == (4, 2)

    grid = jnp.arange(3.0)
    payload = jnp.stack([grid, grid**2], axis=1)
    slopes = jnp.ones_like(payload)
    assert cubic_hermite_interp(grid, payload, slopes, 0.5, axis=0).shape == (2,)
    assert monotone_cubic_interp(grid, payload, 0.5, axis=0).shape == (2,)
    with pytest.raises(TypeError, match="different numbers of dimensions"):
        natural_cubic_spline_coeffs(grid, payload)

    with pytest.raises(jax.errors.TracerIntegerConversionError):
        trapz(jnp.ones((3, 4)), axis=-1)
    with pytest.raises(ValueError, match="Incompatible shapes for broadcasting"):
        cumulative_trapz(jnp.ones((3, 4)), grid, axis=0)

    with pytest.raises(ValueError, match="outside query points"):
        regular_grid_interp(
            (jnp.array([0.0, 1.0]),),
            jnp.array([2.0, 3.0]),
            jnp.array([[2.0]]),
            boundary="reject",
        )

    knots = open_uniform_knots(0.0, 1.0, n_basis=4, degree=3)
    with pytest.raises(ValueError, match="coefficient axis length"):
        bspline_eval(knots, jnp.ones(3), jnp.array([0.5]), degree=3)

    nodes, weights = gauss_laguerre_nodes(4)
    assert nodes.shape == weights.shape == (4,)
    with pytest.raises(ValueError, match="n >= 1"):
        gauss_laguerre_nodes(0)


def test_runtime_shape_status_and_failure_probes_match_the_pages() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tests.integration.test_method_page_contract import "
                "_run_runtime_shape_status_and_failure_probes as run; run()"
            ),
        ],
        cwd=DOCS.parent,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert probe.returncode == 0, probe.stderr
