from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from scripts.quad_benchmark_adapters import (
    RunControls,
    matched_capacities,
    normalize_quadax_evaluations,
    normalize_result,
    portable_numeric,
    raw_jaxstro,
    raw_quadax,
)
from scripts.quad_benchmark_cases import CASES, LibraryMethod


def _case(name: str):
    return next(case for case in CASES if case.name == name)


def test_exact_gk_pair_converges_to_the_same_truth() -> None:
    case = _case("smooth_exponential")
    controls = RunControls(epsabs=1.0e-10, epsrel=1.0e-10, max_regions=64)
    ours_raw = raw_jaxstro(case, LibraryMethod.GAUSS_KRONROD, controls)(
        jnp.asarray(case.theta)
    )
    theirs_raw = raw_quadax(case, LibraryMethod.GAUSS_KRONROD, controls)(
        jnp.asarray(case.theta)
    )
    ours = normalize_result(
        ours_raw,
        library="jaxstro",
        family=LibraryMethod.GAUSS_KRONROD,
    )
    theirs = normalize_result(
        theirs_raw,
        library="quadax",
        family=LibraryMethod.GAUSS_KRONROD,
    )
    assert ours.converged and theirs.converged
    assert math.isclose(float(ours.value), float(case.truth), rel_tol=1.0e-10)
    assert math.isclose(float(theirs.value), float(case.truth), rel_tol=1.0e-10)


def test_raw_adapters_are_jittable_array_only_pytrees() -> None:
    case = _case("smooth_exponential")
    controls = RunControls(epsabs=1.0e-8, epsrel=1.0e-8, max_regions=32)
    for factory in (raw_jaxstro, raw_quadax):
        result = jax.jit(factory(case, LibraryMethod.GAUSS_KRONROD, controls))(
            jnp.asarray(case.theta)
        )
        assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(result))


def test_clenshaw_curtis_normalizes_actual_node_count() -> None:
    assert (
        normalize_quadax_evaluations(
            LibraryMethod.CLENSHAW_CURTIS,
            reported=32,
            order=16,
        )
        == 34
    )


def test_nonfinite_semantics_are_not_collapsed() -> None:
    case = _case("nonfinite_band")
    controls = RunControls(epsabs=1.0e-8, epsrel=1.0e-8, max_regions=32)
    ours = normalize_result(
        raw_jaxstro(case, LibraryMethod.GAUSS_KRONROD, controls)(
            jnp.asarray(case.theta)
        ),
        library="jaxstro",
        family=LibraryMethod.GAUSS_KRONROD,
    )
    theirs = normalize_result(
        raw_quadax(case, LibraryMethod.GAUSS_KRONROD, controls)(
            jnp.asarray(case.theta)
        ),
        library="quadax",
        family=LibraryMethod.GAUSS_KRONROD,
    )
    assert ours.semantic_status == "nonfinite_integrand"
    assert theirs.semantic_status != ours.semantic_status


def test_nonfinite_values_have_portable_explicit_classifications() -> None:
    assert portable_numeric(jnp.asarray(jnp.nan)) == {
        "finite": False,
        "classification": "nan",
    }
    assert portable_numeric(jnp.asarray(jnp.inf)) == {
        "finite": False,
        "classification": "posinf",
    }
    assert portable_numeric(jnp.asarray(-jnp.inf)) == {
        "finite": False,
        "classification": "neginf",
    }


def test_breakpoint_region_and_evaluation_capacities_are_matched() -> None:
    case = _case("breakpoint_kink")
    controls = RunControls(epsabs=1.0e-8, epsrel=1.0e-8, max_regions=64)
    matched = matched_capacities(
        case,
        LibraryMethod.GAUSS_KRONROD,
        node_cost=21,
        controls=controls,
    )
    assert matched.jaxstro_max_regions == matched.quadax_max_ninter == 64
    assert matched.initial_segments == 2
    assert matched.jaxstro_max_evaluations >= 21 * (2 * 64 - 2)


def test_quadax_unknown_status_is_not_given_jaxstro_semantics() -> None:
    raw = raw_quadax(
        _case("smooth_exponential"),
        LibraryMethod.GAUSS_KRONROD,
        RunControls(epsabs=1.0e-10, epsrel=1.0e-10, max_regions=1),
    )(jnp.asarray(1.0))
    normalized = normalize_result(
        raw._replace(status=jnp.asarray(16)),
        library="quadax",
        family=LibraryMethod.GAUSS_KRONROD,
    )
    assert normalized.semantic_status == "quadax_status_16"


def test_quadax_status_two_is_family_specific() -> None:
    case = _case("smooth_exponential")
    controls = RunControls(epsabs=1.0e-10, epsrel=1.0e-10, max_regions=1)
    regional = raw_quadax(case, LibraryMethod.GAUSS_KRONROD, controls)(
        jnp.asarray(1.0)
    )._replace(status=jnp.asarray(2))
    extrapolation = raw_quadax(case, LibraryMethod.ROMBERG, controls)(
        jnp.asarray(1.0)
    )._replace(status=jnp.asarray(2))

    assert (
        normalize_result(
            regional,
            library="quadax",
            family=LibraryMethod.GAUSS_KRONROD,
        ).semantic_status
        == "max_regions"
    )
    assert (
        normalize_result(
            extrapolation,
            library="quadax",
            family=LibraryMethod.ROMBERG,
        ).semantic_status
        == "tolerance_not_met"
    )


def test_romberg_pair_accepts_evaluation_capacity_as_adapter_control() -> None:
    case = _case("smooth_exponential")
    raw = raw_jaxstro(
        case,
        LibraryMethod.ROMBERG,
        RunControls(epsabs=1.0e-8, epsrel=1.0e-8, max_regions=64),
        (("initial_level", 1), ("max_evaluations", 1025)),
    )(jnp.asarray(case.theta))
    normalized = normalize_result(
        raw,
        library="jaxstro",
        family=LibraryMethod.ROMBERG,
    )
    assert normalized.converged
