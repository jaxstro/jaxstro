from __future__ import annotations

import math
import tomllib
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scripts.quad_benchmark_cases import (
    BEST_METHODS,
    CASES,
    METHOD_PAIRS,
    BestMethodChoice,
    ComparisonLabel,
    LibraryMethod,
    independent_gauss_legendre_reference,
)

ROOT = Path(__file__).resolve().parents[2]


def test_quadax_is_benchmark_only_and_exactly_pinned() -> None:
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert payload["dependency-groups"]["benchmark"] == ["quadax==0.2.13"]
    assert all("quadax" not in item for item in payload["project"]["dependencies"])
    assert all(
        "quadax" not in item
        for extra in payload["project"]["optional-dependencies"].values()
        for item in extra
    )


def test_case_truths_match_analytic_definitions() -> None:
    by_name = {case.name: case for case in CASES}
    assert math.isclose(by_name["smooth_exponential"].truth, math.e - 1.0)
    assert math.isclose(by_name["endpoint_sqrt"].truth, 2.0 / 3.0)
    assert math.isclose(by_name["semi_infinite_exponential"].truth, 1.0)
    assert math.isclose(by_name["full_line_gaussian"].truth, math.sqrt(math.pi))
    value = by_name["vector_polynomial_exponential"].fun(
        jnp.asarray(0.5), jnp.asarray(1.0)
    )
    assert value.shape == (2,)
    np.testing.assert_allclose(
        by_name["vector_polynomial_exponential"].derivative_truth,
        (0.0, 1.0),
    )


def test_method_pairs_and_best_choices_are_predeclared() -> None:
    assert {pair.label for pair in METHOD_PAIRS} == {
        ComparisonLabel.EXACT,
        ComparisonLabel.STRONG_MATCH,
        ComparisonLabel.NODE_MATCHED,
        ComparisonLabel.FAMILY_MATCHED,
        ComparisonLabel.CAPABILITY,
    }
    assert set(BEST_METHODS) == {case.name for case in CASES}
    for choice in BEST_METHODS.values():
        assert isinstance(choice, BestMethodChoice)
        assert isinstance(choice.jaxstro_method, LibraryMethod)
        assert isinstance(choice.quadax_method, LibraryMethod)
        assert choice.rationale
        assert choice.source
    assert BEST_METHODS["smooth_exponential"].jaxstro_config == (("pair", 21),)
    assert BEST_METHODS["smooth_exponential"].quadax_config == (("order", 21),)
    assert BEST_METHODS["localized_gaussian"].jaxstro_config == (("initial_order", 17),)
    assert BEST_METHODS["localized_gaussian"].quadax_config == (("order", 16),)
    assert BEST_METHODS["endpoint_sqrt"].jaxstro_config == (("initial_level", 3),)
    assert BEST_METHODS["endpoint_sqrt"].quadax_config == (("order", 61),)


def test_every_case_has_portable_truth_provenance() -> None:
    for case in CASES:
        assert case.truth_provenance.kind in {"analytic", "reference"}
        assert case.truth_provenance.expression
        assert case.truth_provenance.source
        assert case.truth_provenance.atol > 0.0
        assert case.truth_provenance.rtol >= 0.0
    by_name = {case.name: case for case in CASES}
    for name in ("expensive_identity", "narrow_gaussian"):
        provenance = by_name[name].truth_provenance
        assert provenance.kind == "reference"
        assert "NumPy Gauss-Legendre" in provenance.source
        assert provenance.reference_version
        assert provenance.reference_orders == (256, 512, 1024)
        assert provenance.convergence_delta is not None
        assert provenance.convergence_delta <= 1.0e-13


def test_independent_reference_converges_and_cross_checks_truth() -> None:
    by_name = {case.name: case for case in CASES}
    for name in ("expensive_identity", "narrow_gaussian"):
        case = by_name[name]
        reference = independent_gauss_legendre_reference(case)
        assert reference.orders == (256, 512, 1024)
        assert reference.convergence_delta <= 1.0e-13
        assert case.truth is not None
        assert math.isclose(reference.values[-1], float(case.truth), abs_tol=1.0e-14)


def test_case_and_pair_inventory_is_exact() -> None:
    assert tuple(case.name for case in CASES) == (
        "smooth_exponential",
        "vector_polynomial_exponential",
        "localized_gaussian",
        "breakpoint_kink",
        "endpoint_sqrt",
        "semi_infinite_exponential",
        "full_line_gaussian",
        "oscillatory_cosine",
        "expensive_identity",
        "narrow_gaussian",
        "nonfinite_band",
    )
    assert tuple(pair.variant for pair in METHOD_PAIRS) == (
        "pair21",
        "nodes17",
        "closest_work",
        "native_default",
        "divmax10",
        "divmax10",
    )
