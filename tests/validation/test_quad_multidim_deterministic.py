from __future__ import annotations

import copy
import gc
import hashlib
import importlib.util
import itertools
import json
import math
import tomllib
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType

import jax.numpy as jnp
import pytest

from jaxstro import quad
from jaxstro.quad import _cubature, _tensor
from jaxstro.quad._cubature import (
    genz_malik_point_count,
    validate_cubature_capacity,
)
from jaxstro.quad._tensor import validate_adaptive_tensor_capacity

ROOT = Path(__file__).parents[2]
REFERENCE_PATH = ROOT / "tests/validation/data/quad-b1-genz-reference.json"
GENERATOR_PATH = ROOT / "scripts/generate_quad_b1_reference.py"
FORMULA_SET_ID = "genz-unit-hypercube-six-family-v1"
REFERENCE_FORMULA_ID_BY_FAMILY = MappingProxyType(
    {
        "oscillatory": "genz-unit-hypercube-six-family-v1:oscillatory",
        "product_peak": "genz-unit-hypercube-six-family-v1:product-peak",
        "corner_peak": "genz-unit-hypercube-six-family-v1:corner-peak",
        "gaussian": "genz-unit-hypercube-six-family-v1:gaussian",
        "continuous": "genz-unit-hypercube-six-family-v1:continuous",
        "discontinuous": (
            "genz-unit-hypercube-six-family-v1:discontinuous-first-two-axes"
        ),
    }
)

GENZ_FAMILIES = (
    "oscillatory",
    "product_peak",
    "corner_peak",
    "gaussian",
    "continuous",
    "discontinuous",
)

GENZ_MANIFEST = MappingProxyType(
    {
        "dimensions": (2, 4, 6, 8),
        "families": GENZ_FAMILIES,
        "a_rule": "0.35 + 0.05 * arange(1, dimension + 1)",
        "u_rule": "arange(1, dimension + 1) / (dimension + 1)",
        "dtype": "float64",
        "method_controls": MappingProxyType(
            {
                "fixed_tensor": MappingProxyType(
                    {
                        "dimensions": (2, 4),
                        "families": (
                            "oscillatory",
                            "product_peak",
                            "corner_peak",
                            "gaussian",
                        ),
                        "family_dimensions": MappingProxyType(
                            {
                                "oscillatory": (2, 4),
                                "product_peak": (2, 4),
                                "corner_peak": (2, 4),
                                "gaussian": (2, 4),
                            }
                        ),
                        "method": "TensorProduct(GaussianRule(12))",
                        "max_evaluations": "12 ** dimension",
                    }
                ),
                "adaptive_tensor": MappingProxyType(
                    {
                        "dimensions": (2, 4),
                        "families": (
                            "oscillatory",
                            "product_peak",
                            "corner_peak",
                            "gaussian",
                            "continuous",
                        ),
                        "family_dimensions": MappingProxyType(
                            {
                                "oscillatory": (2, 4),
                                "product_peak": (2, 4),
                                "corner_peak": (2, 4),
                                "gaussian": (2, 4),
                                "continuous": (2,),
                            }
                        ),
                        "method": ("AdaptiveTensorClenshawCurtis(initial_level=2)"),
                        "max_evaluations": 32_768,
                        "epsabs": 1.0e-8,
                        "epsrel": 1.0e-8,
                    }
                ),
                "adaptive_cubature": MappingProxyType(
                    {
                        "dimensions": (2, 4, 6, 8),
                        "families": GENZ_FAMILIES,
                        "family_dimensions": MappingProxyType(
                            {
                                "oscillatory": (2, 4, 6, 8),
                                "product_peak": (2, 4, 6, 8),
                                "corner_peak": (2, 4, 6, 8),
                                "gaussian": (2, 4, 6, 8),
                                "continuous": (2, 4, 8),
                                "discontinuous": (2, 4, 6, 8),
                            }
                        ),
                        "method": "AdaptiveCubature(GenzMalik())",
                        "max_evaluations": 500_000,
                        "max_regions": 4_096,
                        "epsabs": 1.0e-8,
                        "epsrel": 1.0e-8,
                    }
                ),
            }
        ),
        "structural_preflight": MappingProxyType(
            {
                "dimensions": (2, 3, 4, 5, 6, 7, 8),
                "adaptive_tensor_initial_evaluations": MappingProxyType(
                    {
                        2: 65,
                        3: 425,
                        4: 2_625,
                        5: 15_625,
                        6: 90_625,
                        7: 515_625,
                        8: 2_890_625,
                    }
                ),
                "adaptive_tensor_exact_capacity_status": "accepted",
                "adaptive_tensor_under_capacity_status": "ValueError",
                "adaptive_cubature_initial_evaluations": MappingProxyType(
                    {
                        2: 17,
                        3: 33,
                        4: 57,
                        5: 93,
                        6: 149,
                        7: 241,
                        8: 401,
                    }
                ),
                "adaptive_cubature_exact_capacity_status": "accepted",
                "adaptive_cubature_under_capacity_status": "ValueError",
            }
        ),
        "b4_carry_forward": MappingProxyType(
            {
                "adaptive_tensor": MappingProxyType(
                    {
                        "dimensions": (5, 6, 7, 8),
                        "required_metrics": (
                            "compile_time",
                            "warm_runtime",
                            "process_memory",
                            "device_memory",
                            "dtype",
                            "payload",
                            "capacity",
                        ),
                        "claim_boundary": (
                            "structural acceptance is not practical runtime "
                            "certification; disclose intrinsic tensor frontier "
                            "and fixed-capacity O(C d) storage growth"
                        ),
                    }
                ),
                "adaptive_cubature": MappingProxyType(
                    {
                        "dimensions": (2, 4, 6, 8),
                        "required_metrics": (
                            "compile_time",
                            "warm_runtime",
                            "process_memory",
                            "device_memory",
                            "dtype",
                            "payload_shape",
                            "reachable_store_capacity",
                        ),
                        "claim_boundary": (
                            "B1 certifies bounded declared cases, not universal "
                            "payload/dtype/store memory safety"
                        ),
                    }
                ),
            }
        ),
        "stress_records": MappingProxyType(
            {
                "adaptive_tensor_250000": MappingProxyType(
                    {
                        "dimensions": (2, 4),
                        "max_evaluations": 250_000,
                        "status": "incomplete_non_default_stress",
                        "fresh_d2_peak_rss_bytes": 12_413_124_608,
                        "fresh_d4_peak_rss_bytes": 842_678_272,
                        "combined_peak_rss_bytes": 16_738_811_904,
                        "combined_elapsed_seconds": 684.01,
                        "completed_cases": 11,
                        "threshold_misses": 0,
                    }
                ),
            }
        ),
        "threshold_by_family": MappingProxyType(
            {
                "fixed_tensor": MappingProxyType(
                    {
                        "oscillatory": 2.0e-8,
                        "product_peak": 2.0e-8,
                        "corner_peak": 2.0e-8,
                        "gaussian": 2.0e-8,
                    }
                ),
                "adaptive_tensor": MappingProxyType(
                    {
                        "oscillatory": 5.0e-7,
                        "product_peak": 5.0e-7,
                        "corner_peak": 5.0e-7,
                        "gaussian": 5.0e-7,
                        "continuous": 5.0e-5,
                        "discontinuous": 5.0e-5,
                    }
                ),
                "adaptive_cubature": MappingProxyType(
                    {
                        "oscillatory": 5.0e-7,
                        "product_peak": 5.0e-7,
                        "corner_peak": 5.0e-7,
                        "gaussian": 5.0e-7,
                        "continuous": 5.0e-5,
                        "discontinuous": 5.0e-5,
                    }
                ),
            }
        ),
    }
)

FIXED_TENSOR_LIMITATION_RESIDUALS = MappingProxyType(
    {
        ("continuous", 2): 4.020689850376957e-4,
        ("continuous", 4): 3.3746643580634395e-4,
        ("discontinuous", 2): 1.455241594837775e-2,
        ("discontinuous", 4): 4.33358357839128e-2,
    }
)

ADAPTIVE_TENSOR_LIMITATIONS = MappingProxyType(
    {
        ("discontinuous", 2): MappingProxyType(
            {
                "residual": 1.21599337392575e-3,
                "evaluations": 24_961,
                "refinements": 9,
                "level": 7,
                "frontier_norm": 5.352858148793382e-3,
            }
        ),
        ("continuous", 4): MappingProxyType(
            {
                "residual": 3.9738172472236766e-4,
                "evaluations": 32_385,
                "refinements": 4,
                "level": 4,
                "frontier_norm": 6.916251122028871e-4,
            }
        ),
        ("discontinuous", 4): MappingProxyType(
            {
                "residual": 3.218803677795348e-2,
                "evaluations": 32_385,
                "refinements": 4,
                "level": 4,
                "frontier_norm": 2.3432820553677375e-2,
            }
        ),
    }
)

CUBATURE_LIMITATIONS = MappingProxyType(
    {
        ("continuous", 6): MappingProxyType(
            {
                "residual": 1.0946951546741968e-4,
                "error_norm": 2.654924527754773e-4,
                "evaluations": 499_895,
                "level": 14,
                "refinements": 1_677,
                "regions": 1_678,
            }
        )
    }
)

ANALYTIC_CASES = (
    (
        "constant",
        lambda x: jnp.ones(x.shape[0], dtype=x.dtype),
        lambda dimension: 1.0,
    ),
    (
        "product_moment",
        lambda x: jnp.prod(x**2, axis=-1),
        lambda dimension: (1.0 / 3.0) ** dimension,
    ),
    (
        "separable_exponential",
        lambda x: jnp.exp(-jnp.sum(x, axis=-1)),
        lambda dimension: (1.0 - math.exp(-1.0)) ** dimension,
    ),
)


def _parameters(dimension: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    indices = jnp.arange(1, dimension + 1, dtype=jnp.float64)
    return 0.35 + 0.05 * indices, indices / (dimension + 1)


def _parameter_text(dimension: int) -> tuple[list[str], list[str]]:
    def text(value: Fraction) -> str:
        if value.denominator == 1:
            return str(value.numerator)
        return f"{value.numerator}/{value.denominator}"

    a = [text(Fraction(35 + 5 * index, 100)) for index in range(1, dimension + 1)]
    u = [text(Fraction(index, dimension + 1)) for index in range(1, dimension + 1)]
    return a, u


def _genz_integrand(family: str, dimension: int):
    a, u = _parameters(dimension)
    if family == "oscillatory":
        return lambda x: jnp.cos(2.0 * jnp.pi * u[0] + x @ a)
    if family == "product_peak":
        return lambda x: jnp.prod(
            1.0 / (a**-2 + (x - u) ** 2),
            axis=-1,
        )
    if family == "corner_peak":
        return lambda x: (1.0 + x @ a) ** (-(dimension + 1))
    if family == "gaussian":
        return lambda x: jnp.exp(-jnp.sum((a * (x - u)) ** 2, axis=-1))
    if family == "continuous":
        return lambda x: jnp.exp(-jnp.sum(a * jnp.abs(x - u), axis=-1))
    if family == "discontinuous":
        return lambda x: jnp.where(
            (x[:, 0] <= u[0]) & (x[:, 1] <= u[1]),
            jnp.exp(x @ a),
            jnp.asarray(0.0, dtype=x.dtype),
        )
    raise ValueError(f"unknown Genz family: {family}")


def _genz_closed_form(family: str, dimension: int) -> jnp.ndarray:
    a, u = _parameters(dimension)
    if family == "oscillatory":
        amplitude = jnp.prod(2.0 * jnp.sin(0.5 * a) / a)
        return amplitude * jnp.cos(2.0 * jnp.pi * u[0] + 0.5 * jnp.sum(a))
    if family == "product_peak":
        return jnp.prod(a * (jnp.arctan(a * (1.0 - u)) + jnp.arctan(a * u)))
    if family == "corner_peak":
        alternating_sum = jnp.asarray(0.0, dtype=jnp.float64)
        for mask in itertools.product((0, 1), repeat=dimension):
            mask_array = jnp.asarray(mask, dtype=jnp.float64)
            alternating_sum = alternating_sum + (
                (-1.0) ** sum(mask) / (1.0 + jnp.sum(a * mask_array))
            )
        return alternating_sum / (math.factorial(dimension) * jnp.prod(a))
    if family == "gaussian":
        return jnp.prod(
            jnp.sqrt(jnp.pi) / (2.0 * a) * (jax_erf(a * (1.0 - u)) + jax_erf(a * u))
        )
    if family == "continuous":
        return jnp.prod((2.0 - jnp.exp(-a * u) - jnp.exp(-a * (1.0 - u))) / a)
    if family == "discontinuous":
        upper = jnp.ones(dimension, dtype=jnp.float64).at[:2].set(u[:2])
        return jnp.prod(jnp.expm1(a * upper) / a)
    raise ValueError(f"unknown Genz family: {family}")


def jax_erf(value: jnp.ndarray) -> jnp.ndarray:
    from jax.scipy.special import erf

    return erf(value)


def _method_case(method_name: str, dimension: int):
    controls = GENZ_MANIFEST["method_controls"][method_name]
    if method_name == "fixed_tensor":
        return quad.TensorProduct(quad.GaussianRule(12)), {
            "epsabs": 0.0,
            "epsrel": 0.0,
            "max_evaluations": 12**dimension,
        }
    if method_name == "adaptive_tensor":
        return quad.AdaptiveTensorClenshawCurtis(initial_level=2), {
            "epsabs": controls["epsabs"],
            "epsrel": controls["epsrel"],
            "max_evaluations": controls["max_evaluations"],
        }
    if method_name == "adaptive_cubature":
        return quad.AdaptiveCubature(quad.GenzMalik()), {
            "epsabs": controls["epsabs"],
            "epsrel": controls["epsrel"],
            "max_evaluations": controls["max_evaluations"],
            "max_regions": controls["max_regions"],
        }
    raise ValueError(f"unknown method: {method_name}")


def _integrate(method_name: str, dimension: int, fun):
    method, controls = _method_case(method_name, dimension)
    return quad.integrate(
        fun,
        quad.Hyperrectangle(
            jnp.zeros(dimension, dtype=jnp.float64),
            jnp.ones(dimension, dtype=jnp.float64),
        ),
        method=method,
        gradient="stop",
        **controls,
    )


def _assert_work_and_status(method_name: str, dimension: int, result) -> None:
    status = int(result.status)
    if method_name == "fixed_tensor":
        assert status == quad.QuadStatus.ERROR_ESTIMATE_UNAVAILABLE
        assert int(result.work.evaluations) == 12**dimension
        assert int(result.work.refinements) == 0
        return

    assert status in (
        quad.QuadStatus.CONVERGED,
        quad.QuadStatus.MAX_EVALUATIONS,
        quad.QuadStatus.MAX_REGIONS,
        quad.QuadStatus.ROUNDOFF_LIMITED,
    )
    if status == quad.QuadStatus.CONVERGED:
        assert float(result.error.norm) <= float(result.tolerance)
    if method_name == "adaptive_tensor":
        assert (
            int(result.work.evaluations)
            <= GENZ_MANIFEST["method_controls"][method_name]["max_evaluations"]
        )
        assert int(result.work.active_regions) == 0
        assert int(result.work.levels) >= 2
        return

    point_count = genz_malik_point_count(dimension)
    refinements = int(result.work.refinements)
    assert int(result.work.evaluations) == point_count * (1 + 2 * refinements)
    assert int(result.work.active_regions) == refinements + 1
    controls = GENZ_MANIFEST["method_controls"][method_name]
    assert int(result.work.evaluations) <= controls["max_evaluations"]
    assert int(result.work.active_regions) <= controls["max_regions"]


def _load_reference() -> dict:
    assert GENERATOR_PATH.is_file(), "Task 5 reference generator is missing"
    assert REFERENCE_PATH.is_file(), "Task 5 reference artifact is missing"
    artifact = json.loads(REFERENCE_PATH.read_text())
    assert artifact["schema_version"] == 1
    assert artifact["formula_set_id"] == FORMULA_SET_ID
    assert artifact["generator"]["precision_decimal_digits"] == 80
    assert (
        artifact["generator"]["source_sha256"]
        == hashlib.sha256(GENERATOR_PATH.read_bytes()).hexdigest()
    )
    return artifact


def _assert_reference_formula_ids(records: list[dict]) -> None:
    mismatches = [
        (
            record["family"],
            record["formula_id"],
            REFERENCE_FORMULA_ID_BY_FAMILY.get(record["family"]),
        )
        for record in records
        if record["formula_id"] != REFERENCE_FORMULA_ID_BY_FAMILY.get(record["family"])
    ]
    assert not mismatches, f"formula ID mismatch: {mismatches}"


def _runtime_cases():
    return tuple(
        (method_name, dimension)
        for method_name, controls in GENZ_MANIFEST["method_controls"].items()
        for dimension in controls["dimensions"]
    )


def _runtime_family_cases():
    return tuple(
        (method_name, dimension, family)
        for method_name, controls in GENZ_MANIFEST["method_controls"].items()
        for family, dimensions in controls["family_dimensions"].items()
        for dimension in dimensions
    )


@pytest.fixture(autouse=True)
def _bounded_runtime_cache_lifetime():
    yield
    import jax

    jax.clear_caches()
    _tensor._adaptive_tensor_capacity_cached.cache_clear()
    _tensor._represented_cc_axis_metadata_cached.cache_clear()
    _cubature._genz_malik_data_cached.cache_clear()
    gc.collect()


@pytest.mark.parametrize(
    ("dimension", "minimum"),
    GENZ_MANIFEST["structural_preflight"][
        "adaptive_tensor_initial_evaluations"
    ].items(),
)
def test_adaptive_tensor_structural_preflight_has_exact_minimum(
    dimension: int,
    minimum: int,
):
    accepted = validate_adaptive_tensor_capacity(
        initial_level=2,
        dimension=dimension,
        max_evaluations=minimum,
        dtype=jnp.float64,
    )
    assert accepted.initial_evaluations == minimum
    with pytest.raises(
        ValueError,
        match=rf"requires (?:at least )?{minimum} evaluations",
    ):
        validate_adaptive_tensor_capacity(
            initial_level=2,
            dimension=dimension,
            max_evaluations=minimum - 1,
            dtype=jnp.float64,
        )


@pytest.mark.parametrize(
    ("dimension", "minimum"),
    GENZ_MANIFEST["structural_preflight"][
        "adaptive_cubature_initial_evaluations"
    ].items(),
)
def test_adaptive_cubature_structural_preflight_has_exact_minimum(
    dimension: int,
    minimum: int,
):
    assert genz_malik_point_count(dimension) == minimum
    accepted = validate_cubature_capacity(
        dimension=dimension,
        max_evaluations=minimum,
        max_regions=1,
    )
    assert accepted.point_count == minimum
    assert accepted.store_capacity == 1
    with pytest.raises(
        ValueError,
        match=rf"requires {minimum} evaluations",
    ):
        validate_cubature_capacity(
            dimension=dimension,
            max_evaluations=minimum - 1,
            max_regions=1,
        )


def test_manifest_keeps_runtime_certification_distinct_from_b4_obligations():
    assert GENZ_MANIFEST["method_controls"]["adaptive_tensor"]["dimensions"] == (
        2,
        4,
    )
    assert GENZ_MANIFEST["b4_carry_forward"]["adaptive_tensor"]["dimensions"] == (
        5,
        6,
        7,
        8,
    )
    assert GENZ_MANIFEST["b4_carry_forward"]["adaptive_cubature"][
        "required_metrics"
    ] == (
        "compile_time",
        "warm_runtime",
        "process_memory",
        "device_memory",
        "dtype",
        "payload_shape",
        "reachable_store_capacity",
    )


def test_reference_dependencies_are_development_only_and_exactly_pinned():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert pyproject["dependency-groups"]["reference"] == [
        "mpmath==1.3.0",
        "scipy==1.16.0",
    ]
    runtime = tuple(pyproject["project"]["dependencies"])
    assert all("mpmath" not in dependency for dependency in runtime)
    assert all("scipy" not in dependency for dependency in runtime)


def test_reference_generator_owns_precision_and_restores_caller_context():
    spec = importlib.util.spec_from_file_location(
        "quad_b1_reference_generator",
        GENERATOR_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    original_dps = generator.mp.mp.dps
    try:
        generator.mp.mp.dps = 15
        records_from_low_precision_caller = generator._records()
        assert generator.mp.mp.dps == 15

        generator.mp.mp.dps = 37
        records_from_higher_precision_caller = generator._records()
        assert generator.mp.mp.dps == 37
    finally:
        generator.mp.mp.dps = original_dps

    assert records_from_low_precision_caller == records_from_higher_precision_caller
    oscillatory_d2 = next(
        record["truth_decimal"]
        for record in records_from_low_precision_caller
        if record["family"] == "oscillatory" and record["dimension"] == 2
    )
    assert oscillatory_d2 == (
        "-0.80039965815688459973909807494292399787055795739660590014938079512729989307212925"
    )


def test_reference_artifact_records_reported_and_working_precision():
    artifact = _load_reference()
    assert artifact["generator"]["precision_decimal_digits"] == 80
    assert artifact["generator"]["working_precision_decimal_digits"] == 100


def test_reference_artifact_is_fresh_complete_and_formula_identified():
    artifact = _load_reference()
    expected = {
        (family, dimension)
        for family in GENZ_FAMILIES
        for dimension in GENZ_MANIFEST["dimensions"]
    }
    records = artifact["records"]
    assert {(record["family"], record["dimension"]) for record in records} == expected
    assert tuple(REFERENCE_FORMULA_ID_BY_FAMILY) == GENZ_FAMILIES
    _assert_reference_formula_ids(records)
    assert artifact["b4_carry_forward"]["adaptive_tensor"]["dimensions"] == [
        5,
        6,
        7,
        8,
    ]


@pytest.mark.parametrize(
    ("family", "replacement_formula_id"),
    (
        (
            "oscillatory",
            "genz-unit-hypercube-six-family-v1:oscillatory-v0",
        ),
        (
            "product_peak",
            "genz-unit-hypercube-six-family-v1:unrelated",
        ),
        (
            "corner_peak",
            "genz-unit-hypercube-six-family-v1:gaussian",
        ),
    ),
)
def test_reference_formula_ids_reject_stale_wrong_and_swapped_family_identity(
    family: str,
    replacement_formula_id: str,
):
    records = copy.deepcopy(_load_reference()["records"])
    record = next(item for item in records if item["family"] == family)
    record["formula_id"] = replacement_formula_id

    with pytest.raises(AssertionError, match="formula ID"):
        _assert_reference_formula_ids(records)


@pytest.mark.parametrize("family", GENZ_FAMILIES)
@pytest.mark.parametrize("dimension", GENZ_MANIFEST["dimensions"])
def test_direct_closed_forms_match_independent_80_digit_artifact(
    family: str,
    dimension: int,
):
    artifact = _load_reference()
    record = next(
        item
        for item in artifact["records"]
        if item["family"] == family and item["dimension"] == dimension
    )
    a_text, u_text = _parameter_text(dimension)
    assert record["a"] == a_text
    assert record["u"] == u_text
    assert math.isclose(
        float(_genz_closed_form(family, dimension)),
        float(record["truth_decimal"]),
        rel_tol=2.0e-13,
        abs_tol=2.0e-14,
    )


@pytest.mark.parametrize(
    ("case_name", "fun", "truth"),
    ANALYTIC_CASES,
    ids=[case[0] for case in ANALYTIC_CASES],
)
@pytest.mark.parametrize(
    ("method_name", "dimension"),
    _runtime_cases(),
    ids=lambda value: str(value),
)
def test_b1_methods_match_analytic_anchors(
    method_name: str,
    dimension: int,
    case_name: str,
    fun,
    truth,
):
    del case_name
    result = _integrate(method_name, dimension, fun)
    threshold = GENZ_MANIFEST["threshold_by_family"][method_name]["gaussian"]
    assert abs(float(result.value) - truth(dimension)) <= threshold
    _assert_work_and_status(method_name, dimension, result)


@pytest.mark.parametrize(
    ("method_name", "dimension", "family"),
    _runtime_family_cases(),
    ids=lambda value: str(value),
)
def test_b1_methods_match_six_family_genz_truth(
    method_name: str,
    dimension: int,
    family: str,
):
    artifact = _load_reference()
    record = next(
        item
        for item in artifact["records"]
        if item["family"] == family and item["dimension"] == dimension
    )
    result = _integrate(method_name, dimension, _genz_integrand(family, dimension))
    absolute_error = abs(float(result.value) - float(record["truth_decimal"]))
    threshold = GENZ_MANIFEST["threshold_by_family"][method_name][family]
    diagnostic = (
        f"absolute_error={absolute_error!r}, "
        f"error_norm={float(result.error.norm)!r}, "
        f"evaluations={int(result.work.evaluations)}, "
        f"level={int(result.work.levels)}, "
        f"refinements={int(result.work.refinements)}, "
        f"regions={int(result.work.active_regions)}, "
        f"status={int(result.status)}, threshold={threshold!r}, "
        f"tolerance={float(result.tolerance)!r}, "
        f"value={float(result.value)!r}"
    )
    assert absolute_error <= threshold, diagnostic
    _assert_work_and_status(method_name, dimension, result)


@pytest.mark.parametrize(
    ("family", "dimension", "expected_residual"),
    tuple(
        (family, dimension, residual)
        for (family, dimension), residual in FIXED_TENSOR_LIMITATION_RESIDUALS.items()
    ),
)
def test_fixed_tensor_records_nonsmooth_limitation_without_accuracy_claim(
    family: str,
    dimension: int,
    expected_residual: float,
):
    artifact = _load_reference()
    record = next(
        item
        for item in artifact["records"]
        if item["family"] == family and item["dimension"] == dimension
    )
    result = _integrate(
        "fixed_tensor",
        dimension,
        _genz_integrand(family, dimension),
    )
    residual = abs(float(result.value) - float(record["truth_decimal"]))
    assert residual == pytest.approx(expected_residual, rel=2.0e-12, abs=2.0e-14)
    assert residual > 2.0e-5
    assert int(result.status) == quad.QuadStatus.ERROR_ESTIMATE_UNAVAILABLE
    assert int(result.error.kind) == quad.ErrorKind.UNAVAILABLE
    assert int(result.work.evaluations) == 12**dimension
    assert jnp.isnan(result.error.norm)


@pytest.mark.parametrize(
    ("family", "dimension", "expected"),
    tuple(
        (family, dimension, expected)
        for (family, dimension), expected in ADAPTIVE_TENSOR_LIMITATIONS.items()
    ),
)
def test_adaptive_tensor_records_nonsmooth_capacity_limitation(
    family: str,
    dimension: int,
    expected,
):
    artifact = _load_reference()
    record = next(
        item
        for item in artifact["records"]
        if item["family"] == family and item["dimension"] == dimension
    )
    result = _integrate(
        "adaptive_tensor",
        dimension,
        _genz_integrand(family, dimension),
    )
    residual = abs(float(result.value) - float(record["truth_decimal"]))
    assert residual == pytest.approx(expected["residual"], rel=2.0e-12, abs=2.0e-14)
    assert int(result.status) == quad.QuadStatus.MAX_EVALUATIONS
    assert int(result.work.evaluations) == expected["evaluations"]
    assert int(result.work.refinements) == expected["refinements"]
    assert int(result.work.levels) == expected["level"]
    assert float(result.error.norm) == pytest.approx(
        expected["frontier_norm"],
        rel=2.0e-12,
        abs=2.0e-14,
    )
    assert float(result.tolerance) == pytest.approx(1.0e-8)


@pytest.mark.parametrize(
    ("family", "dimension", "expected"),
    tuple(
        (family, dimension, expected)
        for (family, dimension), expected in CUBATURE_LIMITATIONS.items()
    ),
)
def test_adaptive_cubature_records_nonsmooth_capacity_limitation(
    family: str,
    dimension: int,
    expected,
):
    artifact = _load_reference()
    record = next(
        item
        for item in artifact["records"]
        if item["family"] == family and item["dimension"] == dimension
    )
    result = _integrate(
        "adaptive_cubature",
        dimension,
        _genz_integrand(family, dimension),
    )
    residual = abs(float(result.value) - float(record["truth_decimal"]))
    assert residual == pytest.approx(expected["residual"], rel=2.0e-12, abs=2.0e-14)
    assert residual > GENZ_MANIFEST["threshold_by_family"]["adaptive_cubature"][family]
    assert int(result.status) == quad.QuadStatus.MAX_EVALUATIONS
    assert int(result.work.evaluations) == expected["evaluations"]
    assert int(result.work.levels) == expected["level"]
    assert int(result.work.refinements) == expected["refinements"]
    assert int(result.work.active_regions) == expected["regions"]
    assert float(result.error.norm) == pytest.approx(
        expected["error_norm"],
        rel=2.0e-12,
        abs=2.0e-14,
    )
    assert float(result.tolerance) == pytest.approx(1.0e-8)
