#!/usr/bin/env python3
"""Emit and verify adaptive-quadrature replay derivative evidence."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jaxstro.jaxconfig import enable_high_precision

enable_high_precision()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jaxstro import __version__, quad  # noqa: E402
from jaxstro import quantity as q  # noqa: E402
from jaxstro.evidence import (  # noqa: E402
    ComparisonRecord,
    ComparisonRelation,
    EnvironmentRecord,
    EvidenceArtifact,
    EvidenceStatus,
    MetricRecord,
    artifact_from_dict,
    artifact_to_dict,
    check_artifact,
    emit_artifact,
)
from jaxstro.quad._replay import IntegrateConfig, replay_value  # noqa: E402
from jaxstro.quad.adaptive import _solve_raw  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "docs" / "validation" / "quad-replay-derivatives.json"
REPORT = (
    REPO_ROOT
    / "docs"
    / "60-validation"
    / "numerical"
    / "quadrature-replay-derivatives.md"
)
FD_STEP = 2.0e-5
RELATIVE_GATE = 1.0e-8
ABSOLUTE_GATE = 2.0e-9


@dataclass(frozen=True)
class MethodSpec:
    name: str
    method: Any
    max_evaluations: int
    max_regions: int


METHODS = (
    MethodSpec("gauss_kronrod", quad.GaussKronrod(21), 147, 4),
    MethodSpec(
        "adaptive_clenshaw_curtis",
        quad.AdaptiveClenshawCurtis(17),
        153,
        4,
    ),
    MethodSpec("adaptive_tanh_sinh", quad.AdaptiveTanhSinh(3), 600, 8),
    MethodSpec("romberg", quad.Romberg(2), 257, 1),
    MethodSpec("romberg_tanh_sinh", quad.RombergTanhSinh(2), 801, 1),
)


def central_difference(fun, x, step):
    return (fun(x + step) - fun(x - step)) / (2.0 * step)


def relative_error(actual, expected):
    scale = max(_max_abs(expected), 1.0e-300)
    return _max_abs(jax.tree.map(lambda x, y: x - y, actual, expected)) / scale


def gate(name, observed, threshold, unit="dimensionless"):
    return {
        "name": name,
        "observed": float(observed),
        "threshold": float(threshold),
        "unit": unit,
        "passed": bool(observed <= threshold),
    }


def _max_abs(value) -> float:
    leaves = jax.tree.leaves(value)
    return max(float(jnp.max(jnp.abs(leaf))) for leaf in leaves)


def _portable(value):
    array = np.asarray(value)
    if np.iscomplexobj(array):
        if array.ndim == 0:
            return {"real": float(array.real), "imag": float(array.imag)}
        return [
            {"real": float(item.real), "imag": float(item.imag)}
            for item in array.reshape(-1)
        ]
    if array.ndim == 0:
        return float(array)
    return [float(item) for item in array.reshape(-1)]


def _exp_integral(theta):
    return jnp.expm1(theta) / theta


def _exp_derivative(theta):
    return ((theta - 1.0) * jnp.exp(theta) + 1.0) / theta**2


def _complex_integral(theta):
    z = 1j * theta
    return jnp.expm1(z) / z


def _complex_derivative(theta):
    z = 1j * theta
    return 1j * (((z - 1.0) * jnp.exp(z) + 1.0) / z**2)


def _config(spec: MethodSpec, fun, measure):
    return IntegrateConfig(
        fun=fun,
        method=spec.method,
        measure=measure,
        max_evaluations=spec.max_evaluations,
        max_regions=spec.max_regions,
        error_norm=quad.MaxNorm(),
    )


def _parameter_case(
    spec: MethodSpec,
    *,
    family: str,
    fun,
    analytic_value,
    analytic_derivative,
    theta: float = 0.4,
    domain=None,
    measure=None,
    epsabs: float = 1.0e-10,
    epsrel: float = 1.0e-10,
    suffix: str = "",
):
    domain = quad.Interval(0.0, 1.0) if domain is None else domain
    measure = quad.LebesgueMeasure() if measure is None else measure
    config = _config(spec, fun, measure)
    solve = _solve_raw(config, domain, theta, epsabs, epsrel)

    def public(parameter):
        return quad.integrate(
            fun,
            domain,
            args=parameter,
            method=spec.method,
            measure=measure,
            epsabs=epsabs,
            epsrel=epsrel,
            max_evaluations=spec.max_evaluations,
            max_regions=spec.max_regions,
            gradient="replay",
        ).value

    def frozen(parameter):
        return replay_value(
            config,
            domain,
            parameter,
            jax.tree.map(jax.lax.stop_gradient, solve.evidence),
            solve.result.value,
        )

    replay_ad = jax.jvp(public, (theta,), (1.0,))[1]
    frozen_fd = central_difference(frozen, theta, FD_STEP)
    rerun_fd = central_difference(public, theta, FD_STEP)
    expected_value = analytic_value(theta)
    expected_derivative = analytic_derivative(theta)
    primal_relative = relative_error(solve.result.value, expected_value)
    derivative_relative = relative_error(replay_ad, expected_derivative)
    frozen_relative = relative_error(replay_ad, frozen_fd)
    case_gates = [
        gate("primal_relative_error", primal_relative, RELATIVE_GATE),
        gate("analytic_derivative_relative_error", derivative_relative, RELATIVE_GATE),
        gate("frozen_formula_relative_error", frozen_relative, RELATIVE_GATE),
    ]
    name = f"{spec.name}.{family}{suffix}"
    return {
        "name": name,
        "method": spec.name,
        "family": family,
        "dtype": str(jnp.asarray(theta).dtype),
        "primal_value": _portable(solve.result.value),
        "analytic_value": _portable(expected_value),
        "observed_primal_error": float(_max_abs(solve.result.value - expected_value)),
        "reported_primal_error": float(_max_abs(solve.result.error.norm)),
        "replay_ad_derivative": _portable(replay_ad),
        "analytic_derivative": _portable(expected_derivative),
        "frozen_formula_fd": _portable(frozen_fd),
        "adaptive_rerun_fd": _portable(rerun_fd),
        "accepted_regions": int(solve.result.work.active_regions),
        "accepted_level": int(solve.result.work.levels),
        "parameter_unit": "dimensionless",
        "integral_unit": "dimensionless",
        "derivative_unit": "dimensionless",
        "status": int(solve.result.status),
        "gates": case_gates,
    }


def _domain_parameter_case(family, domain_builder, analytic_value, analytic_derivative):
    spec = METHODS[0]

    def fun(x):
        return x

    parameter = 0.8 if family != "coincident_bounds" else 0.4
    domain = domain_builder(parameter)
    config = _config(spec, fun, quad.LebesgueMeasure())
    solve = _solve_raw(config, domain, (), 1.0e-10, 1.0e-10)

    def public(value):
        return quad.integrate(
            fun,
            domain_builder(value),
            method=spec.method,
            epsabs=1.0e-10,
            epsrel=1.0e-10,
            max_evaluations=spec.max_evaluations,
            max_regions=spec.max_regions,
            gradient="replay",
        ).value

    def frozen(value):
        return replay_value(
            config,
            domain_builder(value),
            (),
            jax.tree.map(jax.lax.stop_gradient, solve.evidence),
            solve.result.value,
        )

    replay_ad = jax.grad(public)(parameter)
    frozen_fd = central_difference(frozen, parameter, FD_STEP)
    rerun_fd = central_difference(public, parameter, FD_STEP)
    expected_value = analytic_value(parameter)
    expected_derivative = analytic_derivative(parameter)
    case_gates = [
        gate(
            "primal_absolute_error",
            abs(float(solve.result.value - expected_value)),
            ABSOLUTE_GATE,
        ),
        gate(
            "analytic_derivative_absolute_error",
            abs(float(replay_ad - expected_derivative)),
            ABSOLUTE_GATE,
        ),
        gate(
            "frozen_formula_absolute_error",
            abs(float(replay_ad - frozen_fd)),
            2.0e-8,
        ),
    ]
    return {
        "name": f"{spec.name}.{family}",
        "method": spec.name,
        "family": family,
        "dtype": str(jnp.asarray(parameter).dtype),
        "primal_value": _portable(solve.result.value),
        "analytic_value": _portable(expected_value),
        "observed_primal_error": abs(float(solve.result.value - expected_value)),
        "reported_primal_error": float(solve.result.error.norm),
        "replay_ad_derivative": _portable(replay_ad),
        "analytic_derivative": _portable(expected_derivative),
        "frozen_formula_fd": _portable(frozen_fd),
        "adaptive_rerun_fd": _portable(rerun_fd),
        "accepted_regions": int(solve.result.work.active_regions),
        "accepted_level": int(solve.result.work.levels),
        "parameter_unit": "coordinate",
        "integral_unit": "coordinate^2",
        "derivative_unit": "coordinate",
        "status": int(solve.result.status),
        "gates": case_gates,
    }


def _failure_case(spec, family, domain, fun, expected_status):
    result = quad.integrate(
        fun,
        domain,
        method=spec.method,
        epsabs=1.0e-9,
        epsrel=1.0e-9,
        max_evaluations=spec.max_evaluations,
        max_regions=spec.max_regions,
        gradient="replay",
    )
    mismatch = int(result.status != expected_status)
    finite_flag = int(bool(jnp.all(jnp.isfinite(result.value))))
    return {
        "name": f"{spec.name}.{family}",
        "method": spec.name,
        "family": family,
        "dtype": str(jnp.asarray(result.value).dtype),
        "primal_value": "nonfinite",
        "analytic_value": "not_applicable",
        "observed_primal_error": "not_applicable",
        "reported_primal_error": "not_applicable",
        "replay_ad_derivative": "undefined",
        "analytic_derivative": "undefined",
        "frozen_formula_fd": "undefined",
        "adaptive_rerun_fd": "undefined",
        "accepted_regions": int(result.work.active_regions),
        "accepted_level": int(result.work.levels),
        "parameter_unit": "not_applicable",
        "integral_unit": "not_applicable",
        "derivative_unit": "undefined",
        "status": int(result.status),
        "gates": [
            gate("status_mismatch", mismatch, 0),
            gate("finite_primal_flag", finite_flag, 0),
        ],
    }


def _semi_infinite_bound_case(spec: MethodSpec, side: str):
    parameter = 0.2

    def fun(x):
        return jnp.exp(-x) if side == "right" else jnp.exp(x)

    def domain_builder(bound):
        if side == "right":
            return quad.RightInfinite(bound)
        return quad.LeftInfinite(bound)

    def analytic_value(bound):
        return jnp.exp(-bound) if side == "right" else jnp.exp(bound)

    def analytic_derivative(bound):
        return -jnp.exp(-bound) if side == "right" else jnp.exp(bound)

    domain = domain_builder(parameter)
    config = _config(spec, fun, quad.LebesgueMeasure())
    solve = _solve_raw(config, domain, (), 1.0e-10, 1.0e-10)

    def public(bound):
        return quad.integrate(
            fun,
            domain_builder(bound),
            method=spec.method,
            epsabs=1.0e-10,
            epsrel=1.0e-10,
            max_evaluations=spec.max_evaluations,
            max_regions=spec.max_regions,
            gradient="replay",
        ).value

    def frozen(bound):
        return replay_value(
            config,
            domain_builder(bound),
            (),
            jax.tree.map(jax.lax.stop_gradient, solve.evidence),
            solve.result.value,
        )

    replay_ad = jax.grad(public)(parameter)
    frozen_fd = central_difference(frozen, parameter, FD_STEP)
    rerun_fd = central_difference(public, parameter, FD_STEP)
    expected_value = analytic_value(parameter)
    expected_derivative = analytic_derivative(parameter)
    return {
        "name": f"{spec.name}.semi_infinite_bound.{side}",
        "method": spec.name,
        "family": "semi_infinite_bound",
        "variant": side,
        "dtype": str(jnp.asarray(parameter).dtype),
        "primal_value": _portable(solve.result.value),
        "analytic_value": _portable(expected_value),
        "observed_primal_error": abs(float(solve.result.value - expected_value)),
        "reported_primal_error": float(solve.result.error.norm),
        "replay_ad_derivative": _portable(replay_ad),
        "analytic_derivative": _portable(expected_derivative),
        "frozen_formula_fd": _portable(frozen_fd),
        "adaptive_rerun_fd": _portable(rerun_fd),
        "accepted_regions": int(solve.result.work.active_regions),
        "accepted_level": int(solve.result.work.levels),
        "parameter_unit": "coordinate",
        "integral_unit": "coordinate times integrand",
        "derivative_unit": "integrand",
        "status": int(solve.result.status),
        "gates": [
            gate(
                "primal_absolute_error",
                abs(float(solve.result.value - expected_value)),
                ABSOLUTE_GATE,
            ),
            gate(
                "analytic_derivative_absolute_error",
                abs(float(replay_ad - expected_derivative)),
                ABSOLUTE_GATE,
            ),
            gate(
                "frozen_formula_absolute_error",
                abs(float(replay_ad - frozen_fd)),
                2.0e-8,
            ),
        ],
    }


def _quantity_case():
    def physical(bound_value, unit, output_unit):
        return quad.integrate(
            lambda x: x,
            quad.Interval(q.Quantity(0.0, unit), q.Quantity(bound_value, unit)),
            method=quad.GaussKronrod(21),
            epsabs=q.Quantity(1.0e-10, unit**2),
            epsrel=1.0e-10,
            max_evaluations=147,
            max_regions=4,
            gradient="replay",
        ).value.to_value(output_unit)

    metre_value = physical(2.0, q.m, q.cm**2)
    centimetre_value = physical(200.0, q.cm, q.cm**2)
    metre_derivative = jax.grad(lambda x: physical(x, q.m, q.cm**2))(2.0)
    centimetre_derivative = jax.grad(lambda x: physical(x, q.cm, q.cm**2))(200.0)
    common_metre_derivative = metre_derivative / 100.0
    value_error = abs(float(metre_value - centimetre_value))
    derivative_error = abs(float(common_metre_derivative - centimetre_derivative))
    return {
        "name": "gauss_kronrod.quantity_rescaling",
        "method": "gauss_kronrod",
        "family": "quantity_rescaling",
        "dtype": str(jnp.asarray(metre_value).dtype),
        "primal_value": {
            "metre_parameterization_in_cm2": float(metre_value),
            "centimetre_parameterization_in_cm2": float(centimetre_value),
        },
        "analytic_value": 20000.0,
        "observed_primal_error": value_error,
        "reported_primal_error": "see raw method cases",
        "replay_ad_derivative": {
            "cm2_per_m": float(metre_derivative),
            "cm2_per_cm": float(centimetre_derivative),
        },
        "analytic_derivative": {"cm2_per_m": 20000.0, "cm2_per_cm": 200.0},
        "frozen_formula_fd": "covered by dimensionless method cases",
        "adaptive_rerun_fd": "covered by dimensionless method cases",
        "accepted_regions": 1,
        "accepted_level": 0,
        "parameter_unit": "m and cm",
        "integral_unit": "cm^2",
        "derivative_unit": "cm^2/m converted to cm^2/cm",
        "status": int(quad.QuadStatus.CONVERGED),
        "gates": [
            gate(
                "physical_value_absolute_error_cm2",
                value_error,
                2.0e-8,
                "cm^2",
            ),
            gate(
                "physical_derivative_absolute_error_cm2_per_cm",
                derivative_error,
                2.0e-8,
                "cm^2/cm",
            ),
        ],
    }


def _stability_ladder(spec: MethodSpec):
    theta = 0.4

    def fun(x, args):
        return jnp.exp(args * x)

    def run(tolerance, max_evaluations, max_regions):
        result = quad.integrate(
            fun,
            quad.Interval(0.0, 1.0),
            args=theta,
            method=spec.method,
            epsabs=tolerance,
            epsrel=tolerance,
            max_evaluations=max_evaluations,
            max_regions=max_regions,
            gradient="replay",
        )
        derivative = jax.grad(
            lambda parameter: (
                quad.integrate(
                    fun,
                    quad.Interval(0.0, 1.0),
                    args=parameter,
                    method=spec.method,
                    epsabs=tolerance,
                    epsrel=tolerance,
                    max_evaluations=max_evaluations,
                    max_regions=max_regions,
                    gradient="replay",
                ).value
            )
        )(theta)
        primal_error = relative_error(result.value, _exp_integral(theta))
        derivative_error = relative_error(derivative, _exp_derivative(theta))
        return {
            "epsabs": float(tolerance),
            "epsrel": float(tolerance),
            "max_evaluations": max_evaluations,
            "max_regions": max_regions,
            "primal_relative_error": primal_error,
            "derivative_relative_error": derivative_error,
            "accepted_regions": int(result.work.active_regions),
            "accepted_level": int(result.work.levels),
            "passed": bool(
                primal_error <= RELATIVE_GATE and derivative_error <= RELATIVE_GATE
            ),
        }

    tolerances = [
        run(value, spec.max_evaluations, spec.max_regions)
        for value in (1.0e-6, 1.0e-8, 1.0e-10)
    ]
    capacities = [
        run(1.0e-10, spec.max_evaluations, spec.max_regions),
        run(
            1.0e-10,
            spec.max_evaluations * 2,
            spec.max_regions if spec.max_regions == 1 else spec.max_regions * 2,
        ),
    ]
    return {
        "method": spec.name,
        "tolerances": tolerances,
        "capacities": capacities,
    }


def _additional_cases():
    improper = _parameter_case(
        METHODS[2],
        family="improper_tail",
        fun=lambda x, args: jnp.exp(-args * x),
        domain=quad.RightInfinite(0.0),
        theta=1.3,
        analytic_value=lambda theta: 1.0 / theta,
        analytic_derivative=lambda theta: -1.0 / theta**2,
    )
    endpoint = _parameter_case(
        METHODS[2],
        family="endpoint_singularity",
        fun=lambda x, args: args / jnp.sqrt(x),
        theta=0.7,
        analytic_value=lambda theta: 2.0 * theta,
        analytic_derivative=lambda _theta: 2.0,
    )
    weighted = _parameter_case(
        METHODS[0],
        family="weighted_density",
        fun=lambda x, _args: x,
        measure=quad.WeightedMeasure(
            lambda x, args: args * (1.0 + x),
            density_unit=q.dimensionless,
        ),
        analytic_value=lambda theta: (5.0 / 6.0) * theta,
        analytic_derivative=lambda _theta: 5.0 / 6.0,
    )
    exhausted_spec = MethodSpec("gauss_kronrod", quad.GaussKronrod(15), 15, 1)
    exhausted = _parameter_case(
        exhausted_spec,
        family="exhausted_finite",
        fun=lambda x, args: jnp.exp(args * x),
        analytic_value=_exp_integral,
        analytic_derivative=_exp_derivative,
        epsabs=0.0,
        epsrel=0.0,
    )
    exhausted["gates"].append(
        gate(
            "expected_exhaustion_status_mismatch",
            int(exhausted["status"] != int(quad.QuadStatus.MAX_EVALUATIONS)),
            0,
        )
    )
    semi_infinite_bounds = [
        _semi_infinite_bound_case(spec, side)
        for spec in (METHODS[2], METHODS[4])
        for side in ("right", "left")
    ]
    return [improper, endpoint, weighted, exhausted, *semi_infinite_bounds]


def run_evidence() -> tuple[dict[str, Any], dict[str, str]]:
    cases = []
    for spec in METHODS:
        cases.append(
            _parameter_case(
                spec,
                family="smooth_parameter",
                fun=lambda x, args: jnp.exp(args * x),
                analytic_value=_exp_integral,
                analytic_derivative=_exp_derivative,
            )
        )
        cases.append(
            _parameter_case(
                spec,
                family="vector_payload",
                fun=lambda x, args: jnp.stack((jnp.exp(args * x), args * x), axis=-1),
                analytic_value=lambda theta: jnp.asarray(
                    [_exp_integral(theta), theta / 2.0]
                ),
                analytic_derivative=lambda theta: jnp.asarray(
                    [_exp_derivative(theta), 0.5]
                ),
            )
        )
        cases.append(
            _parameter_case(
                spec,
                family="complex_payload",
                fun=lambda x, args: jnp.exp(1j * args * x),
                analytic_value=_complex_integral,
                analytic_derivative=_complex_derivative,
            )
        )
    cases.extend(
        [
            _domain_parameter_case(
                "moving_bounds",
                lambda upper: quad.Interval(0.0, upper),
                lambda upper: upper**2 / 2.0,
                lambda upper: upper,
            ),
            _domain_parameter_case(
                "reversed_bounds",
                lambda lower: quad.Interval(lower, 0.0),
                lambda lower: -(lower**2) / 2.0,
                lambda lower: -lower,
            ),
            _domain_parameter_case(
                "coincident_bounds",
                lambda upper: quad.Interval(0.4, upper),
                lambda upper: (upper**2 - 0.4**2) / 2.0,
                lambda upper: upper,
            ),
        ]
    )
    cases.extend(_additional_cases())
    cases.append(_quantity_case())
    for spec in METHODS:
        cases.extend(
            [
                _failure_case(
                    spec,
                    "invalid_input",
                    quad.Interval(jnp.nan, 1.0),
                    lambda x: x,
                    quad.QuadStatus.INVALID_INPUT,
                ),
                _failure_case(
                    spec,
                    "nonfinite_integrand",
                    quad.Interval(0.0, 1.0),
                    lambda x: jnp.where(x > 0.5, jnp.nan, x),
                    quad.QuadStatus.NONFINITE_INTEGRAND,
                ),
            ]
        )
    for case in cases:
        case["status_name"] = quad.QuadStatus(case["status"]).name
    payload = {
        "claim": "replay-differentiable adaptive one-dimensional quadrature",
        "report_mode": "progressive",
        "controls": {
            "fd_step": FD_STEP,
            "relative_gate": RELATIVE_GATE,
            "absolute_gate": ABSOLUTE_GATE,
            "precision": "float64",
        },
        "cases": cases,
        "stability_ladders": [_stability_ladder(spec) for spec in METHODS],
    }
    environment = {
        "device": str(jax.devices()[0]),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "jax_backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "working_tree_dirty": str(
            bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
                ).strip()
            )
        ),
    }
    return payload, environment


def _validate(payload: dict[str, Any]) -> None:
    required_families = {
        "smooth_parameter",
        "vector_payload",
        "complex_payload",
        "moving_bounds",
        "reversed_bounds",
        "coincident_bounds",
        "semi_infinite_bound",
        "improper_tail",
        "endpoint_singularity",
        "weighted_density",
        "exhausted_finite",
        "quantity_rescaling",
        "invalid_input",
        "nonfinite_integrand",
    }
    families = {case["family"] for case in payload["cases"]}
    if not required_families <= families:
        raise ValueError("quad replay evidence case families are incomplete")
    failed = [
        (case["name"], item["name"])
        for case in payload["cases"]
        for item in case["gates"]
        if not item["passed"]
    ]
    if failed:
        raise ValueError(f"quad replay evidence gates failed: {failed}")
    for ladder in payload["stability_ladders"]:
        if not all(item["passed"] for item in ladder["tolerances"][-2:]):
            raise ValueError(f"unstable tolerance ladder: {ladder['method']}")
        if not all(item["passed"] for item in ladder["capacities"]):
            raise ValueError(f"unstable capacity ladder: {ladder['method']}")


def build_artifact(
    payload: dict[str, Any], environment: dict[str, str]
) -> EvidenceArtifact:
    metrics = []
    comparisons = []
    for case in payload["cases"]:
        for item in case["gates"]:
            identity = f"{case['name']}.{item['name']}"
            metrics.append(
                MetricRecord(
                    identity,
                    item["name"],
                    item["observed"],
                    item["unit"],
                    EvidenceStatus.INFO,
                )
            )
            comparisons.append(
                ComparisonRecord(
                    identity + ".gate",
                    identity,
                    ComparisonRelation.LESS_EQUAL,
                    item["threshold"],
                    item["unit"],
                    EvidenceStatus.PASS,
                    note="Predeclared Phase A3 replay evidence threshold.",
                )
            )
    return EvidenceArtifact(
        schema_version="1",
        artifact_id="quad.replay-derivatives",
        artifact_version="1",
        package_version=__version__,
        source_revision=environment["git_revision"],
        generation_command=(
            "uv run --no-sync python scripts/generate_quad_replay_evidence.py --emit"
        ),
        precision="float64",
        deterministic_config=tuple(sorted(payload["controls"].items())),
        environment=EnvironmentRecord(
            "Environment fields are emission snapshots; freshness gates deterministic algorithmic measurements.",
            tuple(sorted(environment.items())),
        ),
        metrics=tuple(metrics),
        comparisons=tuple(comparisons),
        limitations=(
            "First-order replay differentiates the accepted fixed formula, not adaptive control flow.",
            "Adaptive-rerun finite differences are diagnostic near partition or accepted-level changes.",
            "Invalid and nonfinite result derivatives are undefined.",
            "Quantity derivative units are declared for raw-value parameterizations; direct Quantity-PyTree quotient units are not inferred.",
        ),
        method_payload=payload,
    )


def _algorithmic_payload(artifact: EvidenceArtifact) -> str:
    payload = artifact_to_dict(artifact)["method_payload"]
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()

    payload, environment = run_evidence()
    _validate(payload)
    current = build_artifact(payload, environment)
    if args.emit:
        emit_artifact(OUTPUT, current)
        emit_artifact(REPORT, current)
        print(f"wrote {OUTPUT.relative_to(REPO_ROOT)}")
        return 0
    if not OUTPUT.exists():
        print("quad replay derivative evidence is missing")
        return 1
    stored = artifact_from_dict(json.loads(OUTPUT.read_text(encoding="utf-8")))
    check_artifact(REPORT, stored)
    if _algorithmic_payload(stored) != _algorithmic_payload(current):
        print("quad replay derivative algorithmic metrics are stale")
        return 1
    print("quad replay derivative evidence healthy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
