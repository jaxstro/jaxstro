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

MULTIDIMENSIONAL_HEADINGS = (
    "## Scientific question",
    "## Geometric picture",
    "## Derivation",
    "## Computational cost",
    "## What the estimator means",
    "## JAX and differentiation",
    "## Quantities and units",
    "## Worked astrophysical example",
    "## Failure modes",
    "## Audit recipe",
    "## Warranted claim",
)

MULTIDIMENSIONAL_PAGES = (
    "hyperrectangles.md",
    "tensor-product.md",
    "adaptive-cubature.md",
    "sparse-grids.md",
    "randomized-qmc.md",
    "differentiating.md",
    "choosing-a-method.md",
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
    "change-constraints-evolution/lane-emden.md": (
        "jaxstro.numerics.lane_emden",
        "../../50-api/change-constraints/lane-emden.md",
        ("eq-lane-emden-isothermal", "eq-lane-emden-polytropic", "eq-lane-emden-mass"),
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
        "jaxstro.quad",
        "../../50-api/approximation-integration/quad.md",
        ("eq-trapezoid-panel", "eq-cumulative-trapezoid", "eq-trapezoid-error"),
    ),
    "approximation-integration/quadrature.md": (
        "jaxstro.quad",
        "../../50-api/approximation-integration/quad.md",
        (
            "eq-fixed-node-quadrature",
            "eq-gaussian-exactness",
            "eq-standard-normal-hermite",
        ),
    ),
    "approximation-integration/adaptive-quadrature.md": (
        "jaxstro.quad",
        "../../50-api/approximation-integration/quad.md",
        (
            "eq-adaptive-error-account",
            "eq-adaptive-tolerance",
            "eq-adaptive-gk",
            "eq-adaptive-clenshaw-curtis",
            "eq-adaptive-tanh-sinh",
            "eq-adaptive-romberg",
            "eq-adaptive-work",
        ),
    ),
    "linear-structure/linear-algebra.md": (
        "jaxstro.numerics.linear_algebra",
        "../../50-api/linear-structure/linear-algebra.md",
        ("eq-covariance-estimator", "eq-weighted-least-squares"),
    ),
    "linear-structure/operators.md": (
        "jaxstro.numerics.operators",
        "../../50-api/linear-structure/operators.md",
        ("eq-linear-operator", "eq-operator-composition", "eq-operator-adjoint"),
    ),
    "linear-structure/special-functions.md": (
        "jaxstro.numerics.special",
        "../../50-api/linear-structure/special.md",
        ("eq-planck-lambda", "eq-planck-coordinate-change", "eq-legendre-recurrence"),
    ),
    "probability-sampling/distributions.md": (
        "jaxstro.numerics.distributions",
        "../../50-api/randomness/distributions.md",
        ("eq-density-normalization", "eq-cdf-definition", "eq-powerlaw-integral"),
    ),
    "probability-sampling/random.md": (
        "jaxstro.numerics.random",
        "../../50-api/randomness/random.md",
        ("eq-key-split", "eq-key-fold-in"),
    ),
    "probability-sampling/sampling.md": (
        "jaxstro.numerics.sampling",
        "../../50-api/randomness/sampling.md",
        ("eq-inverse-cdf-sampling", "eq-stratified-uniform", "eq-residual-counts"),
    ),
    "discrete-space/grids.md": (
        "jaxstro.numerics.grids",
        "../../50-api/discrete-space/grids.md",
        ("eq-grid-overlap", "eq-conservative-rebin"),
    ),
    "discrete-space/meshes.md": (
        "jaxstro.numerics.meshes",
        "../../50-api/discrete-space/meshes.md",
        ("eq-finite-volume-divergence", "eq-mesh-telescoping", "eq-conservative-remap"),
    ),
    "discrete-space/spatial.md": (
        "jaxstro.spatial",
        "../../50-api/discrete-space/spatial.md",
        ("eq-morton-interleave", "eq-fixed-radius-set"),
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
    "linear-structure/linear-algebra.md": (
        r"C=\frac{1}{n-\mathrm{ddof}}\sum_{i=1}^{n}(x_i-\bar{x})(x_i-\bar{x})^\mathsf{T}",
        r"(X^\mathsf{T}WX)\widehat{\beta}=X^\mathsf{T}Wy",
    ),
    "linear-structure/operators.md": (
        r"(A\circB)x=A(Bx)",
        r"\langley,Ax\rangle=\langleA^\mathsf{T}y,x\rangle",
    ),
    "linear-structure/special-functions.md": (
        r"B_\nu=B_\lambda\left|\frac{d\lambda}{d\nu}\right|=B_\lambda\frac{\lambda^2}{c}",
    ),
    "probability-sampling/distributions.md": (
        r"\int_{\mathcal{S}}p(x)\,dx=1",
        r"F(F^{-1}(u))=u",
    ),
    "probability-sampling/random.md": (
        r"(K_{\mathrm{next}},K_1,\ldots,K_m)=\operatorname{split}(K,m+1)",
    ),
    "probability-sampling/sampling.md": (
        r"X=F^{-1}(U)",
        r"U_i=\frac{i+V_i}{n}",
    ),
    "discrete-space/grids.md": (r"v'_j=\sum_iv_i\frac{\ell_{ji}}{e_{i+1}-e_i}",),
    "discrete-space/meshes.md": (r"\sum_i\Deltax_i(\nabla\cdotF)_i=F_{N+1/2}-F_{1/2}",),
    "discrete-space/spatial.md": (
        r"\mathcal{N}_i=\{j:0<\lVertx_i-x_j\rVert_2\ler_{\mathrm{cut}}\}",
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
    EXACT_RELATIONS["linear-structure/linear-algebra.md"][0]: (
        r"C=\frac{1}{n+\mathrm{ddof}}\sum_{i=1}^{n}(x_i-\bar{x})(x_i-\bar{x})^\mathsf{T}"
    ),
    EXACT_RELATIONS["linear-structure/linear-algebra.md"][1]: (
        r"(X^\mathsf{T}WX)\widehat{\beta}=XWy"
    ),
    EXACT_RELATIONS["linear-structure/operators.md"][0]: r"(A\circB)x=B(Ax)",
    EXACT_RELATIONS["linear-structure/operators.md"][1]: (
        r"\langley,Ax\rangle=\langleAy,x\rangle"
    ),
    EXACT_RELATIONS["linear-structure/special-functions.md"][0]: (
        r"B_\nu=B_\lambda\left|\frac{d\lambda}{d\nu}\right|=B_\lambda\frac{c}{\lambda^2}"
    ),
    EXACT_RELATIONS["probability-sampling/distributions.md"][0]: (
        r"\int_{\mathcal{S}}p(x)\,dx=0"
    ),
    EXACT_RELATIONS["probability-sampling/distributions.md"][1]: r"F(F^{-1}(u))=x",
    EXACT_RELATIONS["probability-sampling/random.md"][0]: (
        r"(K_{\mathrm{next}},K_1,\ldots,K_m)=\operatorname{split}(K,m)"
    ),
    EXACT_RELATIONS["probability-sampling/sampling.md"][0]: r"X=F(U)",
    EXACT_RELATIONS["probability-sampling/sampling.md"][1]: r"U_i=\frac{i-V_i}{n}",
    EXACT_RELATIONS["discrete-space/grids.md"][0]: (
        r"v'_j=\sum_iv_i\ell_{ji}(e_{i+1}-e_i)"
    ),
    EXACT_RELATIONS["discrete-space/meshes.md"][0]: (
        r"\sum_i\Deltax_i(\nabla\cdotF)_i=F_{1/2}-F_{N+1/2}"
    ),
    EXACT_RELATIONS["discrete-space/spatial.md"][0]: (
        r"\mathcal{N}_i=\{j:0\le\lVertx_i-x_j\rVert_2<r_{\mathrm{cut}}\}"
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


def test_adaptive_quadrature_page_publishes_current_capability_and_boundaries() -> None:
    text = _page("approximation-integration/adaptive-quadrature.md")
    normalized = " ".join(text.split())

    for required in (
        "quad.integrate",
        "GaussKronrod",
        "AdaptiveClenshawCurtis",
        "AdaptiveTanhSinh",
        "Romberg",
        "RombergTanhSinh",
        "QuadStatus",
        "QuadWork",
        "ErrorKind",
        "INVALID_INPUT",
        "NONFINITE_INTEGRAND",
        "CONVERGED",
        "ROUNDOFF_LIMITED",
        "MAX_EVALUATIONS",
        "MAX_REGIONS",
        'gradient="stop"',
        "integrand evaluations",
        "not an exact error certificate",
        r"E_{\mathrm{asc}}",
        r"E_{\mathrm{asc}}\,\min",
        r"200\delta",
        r"E_{\mathrm{sum},k}",
        "floor-dominated error",
        "not by itself a status trigger",
        "all-zero `QuadWork` record",
        "tests/validation/test_quad_adaptive_reference.py",
        "docs/validation/quad-adaptive-envelope.json",
    ):
        assert required in normalized

    assert "Quadax owns adaptive quadrature" not in text
    assert "does not establish an adaptive Jaxstro API" not in text


@pytest.mark.parametrize("name", MULTIDIMENSIONAL_PAGES)
def test_multidimensional_pages_follow_the_researcher_contract(name: str) -> None:
    path = DOCS / "20-methods/approximation-integration/multidimensional" / name
    text = path.read_text(encoding="utf-8")
    headings = tuple(re.findall(r"^## .+$", text, flags=re.MULTILINE))

    assert headings == MULTIDIMENSIONAL_HEADINGS, name
    assert text.count("```{math}") >= 2, name
    assert ":::{warning}" in text or ":::{important}" in text, name
    assert "course" not in text.lower(), name
    assert "instructor" not in text.lower(), name
    assert text.isascii(), name


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
    expected_import = (
        "from jaxstro import quad"
        if owner == "jaxstro.quad"
        else f"from {owner} import"
    )
    assert expected_import in text, relative
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
        "linear-structure/linear-algebra.md": (
            r"C.*\\sum.*\(x_i-\\bar\{x\}\).*\(x_i-\\bar\{x\}\)\^\\mathsf\{T\}",
            r"X\^\\mathsf\{T\}WX.*\\widehat\{\\beta\}.*X\^\\mathsf\{T\}Wy",
        ),
        "linear-structure/operators.md": (
            r"A\(\\alpha x\+\\beta z\).*\\alpha A\(x\)\+\\beta A\(z\)",
            r"A\(Bx\).*=.*\(AB\)x",
        ),
        "linear-structure/special-functions.md": (
            r"B_\\lambda.*\\frac\{2hc\^2\}\{\\lambda\^5\}",
            r"B_\\nu.*B_\\lambda.*\\frac\{\\lambda\^2\}\{c\}",
        ),
        "probability-sampling/distributions.md": (
            r"\\int_\{\\mathcal\{S\}\}p\(x\).*dx.*=.*1",
            r"F\(x\).*\\int.*p\(t\).*dt",
        ),
        "probability-sampling/random.md": (
            r"K_\{\\mathrm\{next\}\}.*K_1.*K_m.*\\operatorname\{split\}",
            r"K_i.*\\operatorname\{fold\\_in\}",
        ),
        "probability-sampling/sampling.md": (
            r"X.*F\^\{-1\}\(U\)",
            r"N_i.*\\lfloor.*\\bar\{w\}_i.*\\rfloor",
        ),
        "discrete-space/grids.md": (
            r"\\ell_\{ji\}.*\\max.*\\min",
            r"v'_j.*\\sum_i.*v_i",
        ),
        "discrete-space/meshes.md": (
            r"\\frac\{F_\{i\+1/2\}-F_\{i-1/2\}\}\{\\Delta x_i\}",
            r"\\sum_i.*\\Delta x_i.*\\nabla\\cdot F",
        ),
        "discrete-space/spatial.md": (
            r"m.*x.*y.*z",
            r"\\mathcal\{N\}_i.*0<.*\\lVert x_i-x_j\\rVert_2.*\\le",
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
    assert "nondefault `trapezoid` axes are not currently supported" in integration
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

    linear = " ".join(_page("linear-structure/linear-algebra.md").split())
    assert (
        "value-dependent eager validation is skipped while inputs are traced" in linear
    )
    assert "`rowvar` and `ddof` are static" in linear
    assert "zero variance" in linear
    assert "`n_obs - ddof > 0` for unweighted covariance" in linear
    assert "`sum(weights) - ddof > 0` for weighted covariance" in linear
    assert "frequency-weight-style runtime semantics" in linear
    assert "not an effective-sample-size correction" in linear

    operators = " ".join(_page("linear-structure/operators.md").split())
    assert "Python structure is static" in operators
    assert "PyTree leaves" in operators
    assert "shape checks happen eagerly" in operators

    special = " ".join(_page("linear-structure/special-functions.md").split())
    assert "`degree` and `axis` are static" in special
    assert "value-dependent eager positivity checks are skipped while traced" in special

    distributions = " ".join(_page("probability-sampling/distributions.md").split())
    assert "do not validate parameter domains" in distributions
    assert "outside support" in distributions
    assert "CDF/PPF" in distributions

    random_page = " ".join(_page("probability-sampling/random.md").split())
    assert "`num` and `start` are static" in random_page
    assert "does not establish statistical independence" in random_page

    sampling = " ".join(_page("probability-sampling/sampling.md").split())
    assert "zero-total fallback" in sampling
    assert "eager validation is skipped while weights are traced" in sampling
    assert "integer indices" in sampling

    grids = " ".join(_page("discrete-space/grids.md").split())
    assert "value-dependent eager validation is skipped while traced" in grids
    assert "integrated bin totals" in grids

    meshes = " ".join(_page("discrete-space/meshes.md").split())
    assert "`n_cells` is static" in meshes
    assert "cell averages" in meshes

    spatial = " ".join(_page("discrete-space/spatial.md").split())
    assert "exact only when `did_overflow` is false" in spatial
    assert "host-side, discrete preprocessing" in spatial
    assert "candidate axis has length `27 * Bcap`" in spatial
    assert "`k_max <= 27 * Bcap` is required" in spatial
    assert "`Bcap=None` selects `min(N, max(k_max, 64))`" in spatial
    assert "does not guarantee `k_max <= 27 * Bcap` when `k_max > 27 * N`" in spatial
    assert (
        "`cell_size`, `cutoff`, `k_max`, `Bcap`, and `dims` must be static or "
        "closed over for `jax.jit`"
    ) in spatial


def _run_runtime_shape_status_and_failure_probes() -> None:
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom

    from jaxstro import constants, quad
    from jaxstro.numerics.autodiff import jvp, vjp
    from jaxstro.numerics.distributions import (
        powerlaw_cdf,
        powerlaw_logpdf,
        powerlaw_ppf,
    )
    from jaxstro.numerics.grids import conservative_rebin
    from jaxstro.numerics.interpolation import (
        cubic_hermite_interp,
        monotone_cubic_interp,
        natural_cubic_spline_coeffs,
    )
    from jaxstro.numerics.linear_algebra import (
        covariance_matrix,
        weighted_lstsq,
    )
    from jaxstro.numerics.meshes import (
        conservative_remap_1d,
        divergence_1d,
    )
    from jaxstro.numerics.ode import solve_fixed_step, velocity_verlet
    from jaxstro.numerics.operators import DenseOperator, compose
    from jaxstro.numerics.optimization import armijo_backtracking
    from jaxstro.numerics.random import (
        key_stream,
        residual_resample,
        systematic_resample,
    )
    from jaxstro.numerics.regular_grid import regular_grid_interp
    from jaxstro.numerics.rootfinding import safeguarded_bracketed_root
    from jaxstro.numerics.sampling import inverse_cdf_draw
    from jaxstro.numerics.special import (
        legendre_basis,
        normalize_log_weights,
        planck_lambda_cgs,
        planck_nu_cgs,
    )
    from jaxstro.numerics.splines import bspline_eval, open_uniform_knots
    from jaxstro.spatial import gather_pairs_within_radius

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

    assert jnp.array_equal(quad.trapezoid(jnp.ones((3, 4)), axis=-1), jnp.full(3, 3.0))
    with pytest.raises(ValueError, match="Incompatible shapes for broadcasting"):
        quad.cumulative_trapezoid(jnp.ones((3, 4)), grid, axis=0)

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

    nodes, weights = quad.gauss_laguerre_nodes(4)
    assert nodes.shape == weights.shape == (4,)
    with pytest.raises(ValueError, match="n >= 1"):
        quad.gauss_laguerre_nodes(0)

    fixed_value = quad.fixed(
        lambda x: x**4,
        quad.Interval(-1.0, 1.0),
        rule=quad.GaussianRule(3),
    )
    assert jnp.allclose(fixed_value, 2.0 / 5.0)

    observations = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 8.0]])
    centered = observations - jnp.mean(observations, axis=0)
    covariance = covariance_matrix(observations, ddof=1)
    assert jnp.allclose(covariance, centered.T @ centered / 2.0)
    weighted_singleton = covariance_matrix(
        jnp.array([[3.0, 4.0]]), weights=jnp.array([2.0]), ddof=1
    )
    assert weighted_singleton.shape == (2, 2)
    assert jnp.allclose(weighted_singleton, jnp.zeros((2, 2)))
    design = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    response = jnp.array([1.0, 3.0, 5.0])
    weights = jnp.array([1.0, 2.0, 1.0])
    beta = weighted_lstsq(design, response, weights)
    residual = design @ beta - response
    assert jnp.allclose(design.T @ (weights * residual), 0.0, atol=1e-5)

    left_matrix = jnp.array([[1.0, 2.0], [0.0, 1.0]])
    right_matrix = jnp.array([[2.0, 0.0], [1.0, 3.0]])
    operator = compose(DenseOperator(left_matrix), DenseOperator(right_matrix))
    vector = jnp.array([0.5, -1.0])
    cotangent = jnp.array([1.5, 0.25])
    assert jnp.allclose(operator.matvec(vector), left_matrix @ right_matrix @ vector)
    assert jnp.allclose(
        jnp.vdot(cotangent, operator.matvec(vector)),
        jnp.vdot(operator.rmatvec(cotangent), vector),
    )

    wavelength = jnp.array(5.0e-5)
    temperature = jnp.array(5800.0)
    frequency = constants.C_CGS / wavelength
    b_lambda = planck_lambda_cgs(wavelength, temperature)
    b_nu = planck_nu_cgs(frequency, temperature)
    assert jnp.allclose(b_nu, b_lambda * wavelength**2 / constants.C_CGS)
    probabilities = normalize_log_weights(jnp.array([3.0, 2.0, 1.0]))
    assert jnp.allclose(jnp.sum(probabilities), 1.0)
    basis = legendre_basis(jnp.array([0.2, 0.4]), degree=3)
    assert basis.shape == (2, 4)
    assert jnp.allclose(
        3.0 * basis[:, 3], 5.0 * basis[:, 1] * basis[:, 2] - 2.0 * basis[:, 1]
    )

    x_power = jnp.array([1.0, 2.0, 4.0])
    log_density = powerlaw_logpdf(x_power, alpha=-1.0, xmin=1.0, xmax=4.0)
    assert jnp.all(jnp.isfinite(log_density))
    assert jnp.isneginf(powerlaw_logpdf(jnp.array(0.5), xmin=1.0, xmax=4.0))
    probabilities = jnp.array([0.1, 0.5, 0.9])
    quantiles = powerlaw_ppf(probabilities, alpha=-1.0, xmin=1.0, xmax=4.0)
    assert jnp.allclose(
        powerlaw_cdf(quantiles, alpha=-1.0, xmin=1.0, xmax=4.0),
        probabilities,
    )

    key = jrandom.PRNGKey(23)
    next_key, subkeys = key_stream(key, 3)
    replay_next, replay_subkeys = key_stream(jrandom.PRNGKey(23), 3)
    assert subkeys.shape == (3, 2)
    assert jnp.array_equal(next_key, replay_next)
    assert jnp.array_equal(subkeys, replay_subkeys)
    assert not jnp.array_equal(subkeys[0], subkeys[1])

    tabulated_weight = jnp.ones(5)
    tabulated_grid = jnp.linspace(0.0, 1.0, 5)
    draw = inverse_cdf_draw(tabulated_weight, tabulated_grid, jnp.array(0.4))
    assert jnp.isfinite(draw)
    assert jnp.isfinite(
        jax.grad(
            lambda uniform: inverse_cdf_draw(tabulated_weight, tabulated_grid, uniform)
        )(jnp.array(0.4))
    )
    fallback = systematic_resample(key, jnp.zeros(3), num_samples=6)
    assert fallback.shape == (6,)
    assert jnp.all((fallback >= 0) & (fallback < 3))
    residual_indices = residual_resample(key, jnp.array([0.4, 0.4, 0.2]), num_samples=5)
    assert jnp.array_equal(
        jnp.bincount(residual_indices, length=3), jnp.array([2, 2, 1])
    )

    old_edges = jnp.array([0.0, 1.0, 3.0])
    old_totals = jnp.array([2.0, 6.0])
    new_edges = jnp.array([0.0, 0.5, 2.0, 3.0])
    new_totals = conservative_rebin(old_edges, old_totals, new_edges)
    assert jnp.allclose(jnp.sum(new_totals), jnp.sum(old_totals))
    assert jnp.allclose(new_totals, jnp.array([1.0, 4.0, 3.0]))

    face_flux = jnp.array([1.0, 3.0, 2.0, 5.0])
    mesh_edges = jnp.array([0.0, 0.5, 2.0, 4.0])
    divergence = divergence_1d(face_flux, mesh_edges)
    assert jnp.allclose(
        jnp.sum(jnp.diff(mesh_edges) * divergence), face_flux[-1] - face_flux[0]
    )
    old_averages = jnp.array([2.0, 4.0])
    remapped = conservative_remap_1d(
        jnp.array([0.0, 1.0, 3.0]), old_averages, new_edges
    )
    assert jnp.allclose(
        jnp.sum(remapped * jnp.diff(new_edges)),
        jnp.sum(old_averages * jnp.array([1.0, 2.0])),
    )

    positions = jnp.array(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.5, 0.0, 0.0], [1.1, 0.0, 0.0]]
    )
    cutoff = 0.5
    neighbors, mask, overflow = gather_pairs_within_radius(
        positions,
        origin=jnp.array([0.0, -0.5, -0.5]),
        cell_size=cutoff,
        cutoff=cutoff,
        k_max=3,
        Bcap=4,
        dims=(4, 2, 2),
    )
    assert not bool(overflow)
    for index in range(positions.shape[0]):
        distances = jnp.linalg.norm(positions - positions[index], axis=1)
        expected = set(
            map(int, jnp.where((distances > 0.0) & (distances <= cutoff))[0].tolist())
        )
        actual = set(map(int, neighbors[index][mask[index]].tolist()))
        assert actual == expected


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


def _run_spatial_top_k_bound_failure_probe() -> None:
    import jax.numpy as jnp

    from jaxstro.spatial import gather_pairs_within_radius

    with pytest.raises(
        ValueError,
        match="k argument to top_k must be no larger than size along axis",
    ):
        gather_pairs_within_radius(
            jnp.array([[0.0, 0.0, 0.0]]),
            origin=jnp.zeros(3),
            cell_size=1.0,
            cutoff=0.5,
            k_max=28,
            Bcap=1,
            dims=(1, 1, 1),
        )


def test_spatial_top_k_bound_failure_matches_the_documented_candidate_axis() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tests.integration.test_method_page_contract import "
                "_run_spatial_top_k_bound_failure_probe as run; run()"
            ),
        ],
        cwd=DOCS.parent,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert probe.returncode == 0, probe.stderr


def _run_supported_jitted_spatial_probe() -> None:
    import jax
    import jax.numpy as jnp

    from jaxstro.spatial import gather_pairs_within_radius

    positions = jnp.array(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.5, 0.0, 0.0], [1.1, 0.0, 0.0]]
    )

    def query(pos):
        return gather_pairs_within_radius(
            pos,
            origin=jnp.array([0.0, -0.5, -0.5]),
            cell_size=0.5,
            cutoff=0.5,
            k_max=3,
            Bcap=4,
            dims=(4, 2, 2),
        )

    neighbors, mask, overflow = jax.jit(query)(positions)
    assert neighbors.shape == mask.shape == (4, 3)
    assert not bool(overflow)
    assert set(map(int, neighbors[0][mask[0]].tolist())) == {1, 2}


def test_spatial_jit_succeeds_with_shape_and_float_controls_closed_over() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tests.integration.test_method_page_contract import "
                "_run_supported_jitted_spatial_probe as run; run()"
            ),
        ],
        cwd=DOCS.parent,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert probe.returncode == 0, probe.stderr
