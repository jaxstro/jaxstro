#!/usr/bin/env python3
"""Generate deterministic Phase B4 multidimensional truth evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, cast

from jaxstro.jaxconfig import enable_high_precision

enable_high_precision()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jaxstro import quad  # noqa: E402
from jaxstro.quantity import Msun, Myr, Quantity, dimensionless, pc, rad  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRUTH_OUTPUT = ROOT / "docs/validation/quad-multidim-truth.json"
REPLAY_OUTPUT = ROOT / "docs/validation/quad-multidim-replay.json"
GENZ_REFERENCE = ROOT / "tests/validation/data/quad-b1-genz-reference.json"

VALIDATION_FAMILIES = (
    "tensor_polynomial",
    "beta_product",
    "separable_exponential",
    "rotated_smooth",
    "genz_oscillatory",
    "genz_product_peak",
    "genz_corner_peak",
    "genz_gaussian",
    "genz_continuous",
    "genz_discontinuous",
    "localized_peak",
    "boundary_layer",
)


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _environment() -> dict[str, object]:
    return {
        "jax": jax.__version__,
        "jax_backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "jaxlib": jax.lib.__version__,
        "machine": platform.machine(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def _work(result) -> dict[str, int]:
    return {
        name: int(np.asarray(getattr(result.work, name)))
        for name in (
            "evaluations",
            "refinements",
            "active_regions",
            "levels",
            "replicates",
        )
    }


def _controls(method) -> dict[str, object]:
    controls: dict[str, object] = {
        "epsabs": 1.0e-9,
        "epsrel": 1.0e-9,
        "gradient": "replay",
        "max_evaluations": 65_536,
    }
    if isinstance(method, quad.AdaptiveCubature):
        controls["max_regions"] = 1_024
    if isinstance(method, (quad.Smolyak, quad.AdaptiveSmolyak)):
        controls.update(max_indices=128, max_frontier=2_049, max_nodes=65_536)
    return controls


def _method(method_id: str):
    methods = {
        "tensor_gauss_3": quad.TensorProduct(quad.GaussianRule(3)),
        "tensor_gauss_12": quad.TensorProduct(quad.GaussianRule(12)),
        "tensor_gauss_20": quad.TensorProduct(quad.GaussianRule(20)),
        "adaptive_tensor": quad.AdaptiveTensorClenshawCurtis(initial_level=2),
        "cubature": quad.AdaptiveCubature(),
        "smolyak_5": quad.Smolyak(level=5),
        "adaptive_smolyak": quad.AdaptiveSmolyak(initial_level=1),
    }
    return methods[method_id]


def _fraction(value: str) -> float:
    return float(Fraction(value))


def _genz_reference(family: str) -> tuple[dict[str, object], float]:
    artifact = json.loads(GENZ_REFERENCE.read_text())
    record = next(
        item
        for item in artifact["records"]
        if item["dimension"] == 2 and item["family"] == family
    )
    provenance = {
        "absolute_uncertainty": 1.0e-79,
        "external_owner": "Genz test-integral family; closed form evaluated by mpmath",
        "formula_id": record["formula_id"],
        "generator": "scripts/generate_quad_b1_reference.py",
        "generator_source_sha256": artifact["generator"]["source_sha256"],
        "parameters": {"a": record["a"], "u": record["u"]},
        "precision_decimal_digits": artifact["generator"]["precision_decimal_digits"],
        "reference_artifact_sha256": _sha256(GENZ_REFERENCE),
    }
    return provenance, float(record["truth_decimal"])


def _genz_fun(family: str):
    a = jnp.asarray([0.4, 0.45], dtype=jnp.float64)
    u = jnp.asarray([1.0 / 3.0, 2.0 / 3.0], dtype=jnp.float64)
    if family == "oscillatory":
        return lambda x: jnp.cos(2.0 * jnp.pi * u[0] + x @ a)
    if family == "product_peak":
        return lambda x: jnp.prod(1.0 / (a**-2 + (x - u) ** 2), axis=-1)
    if family == "corner_peak":
        return lambda x: (1.0 + x @ a) ** -3
    if family == "gaussian":
        return lambda x: jnp.exp(-jnp.sum((a * (x - u)) ** 2, axis=-1))
    if family == "continuous":
        return lambda x: jnp.exp(-jnp.sum(a * jnp.abs(x - u), axis=-1))
    if family == "discontinuous":
        return lambda x: jnp.where(
            (x[..., 0] <= u[0]) & (x[..., 1] <= u[1]),
            jnp.exp(x @ a),
            0.0,
        )
    raise ValueError(f"unknown Genz family {family}")


def _registry() -> list[dict[str, object]]:
    records: list[dict[str, object]] = [
        {
            "family": "tensor_polynomial",
            "dimension": 2,
            "method_id": "tensor_gauss_3",
            "fun": lambda x: jnp.prod(1.0 + x + x**2, axis=-1),
            "truth": (1.0 + 0.5 + 1.0 / 3.0) ** 2,
            "truth_source": {"kind": "analytic", "formula": "(11/6)^2"},
            "tolerance": 2.0e-13,
            "expected_claim": "exact tensor polynomial moment",
        },
        {
            "family": "beta_product",
            "dimension": 3,
            "method_id": "smolyak_5",
            "fun": lambda x: jnp.prod(x**2 * (1.0 - x), axis=-1),
            "truth": (1.0 / 12.0) ** 3,
            "truth_source": {"kind": "analytic", "formula": "(1/12)^3"},
            "tolerance": 2.0e-13,
            "expected_claim": "sparse polynomial moment",
        },
        {
            "family": "separable_exponential",
            "dimension": 4,
            "method_id": "smolyak_5",
            "fun": lambda x: jnp.exp(-jnp.sum(x, axis=-1)),
            "truth": math.expm1(-1.0) ** 4,
            "truth_source": {
                "kind": "analytic",
                "formula": "(1-exp(-1))^4",
            },
            "tolerance": 2.0e-7,
            "expected_claim": (
                "fixed sparse accuracy threshold; estimator does not claim convergence"
            ),
        },
        {
            "family": "rotated_smooth",
            "dimension": 4,
            "method_id": "cubature",
            "fun": lambda x: 1.0 + 0.1 * (jnp.sum(x, axis=-1) - 2.0) ** 2,
            "truth": 1.0 + 0.1 * 4.0 / 12.0,
            "truth_source": {
                "kind": "analytic",
                "formula": "1 + 0.1 Var(sum(X_i))",
            },
            "tolerance": 2.0e-12,
            "expected_claim": "rotated smooth cubature convergence",
        },
    ]
    for family in (
        "oscillatory",
        "product_peak",
        "corner_peak",
        "gaussian",
        "continuous",
        "discontinuous",
    ):
        source, truth = _genz_reference(family)
        records.append(
            {
                "family": f"genz_{family}",
                "dimension": 2,
                "method_id": (
                    "cubature"
                    if family in {"continuous", "discontinuous"}
                    else "tensor_gauss_12"
                ),
                "fun": _genz_fun(family),
                "truth": truth,
                "truth_source": {"kind": "reference", **source},
                "tolerance": 5.0e-5,
                "expected_claim": "Genz reference accuracy threshold",
            }
        )
    localized_alpha = 25.0
    localized_1d = (
        math.sqrt(math.pi)
        / (2.0 * math.sqrt(localized_alpha))
        * (
            math.erf(math.sqrt(localized_alpha) * 0.4)
            + math.erf(math.sqrt(localized_alpha) * 0.6)
        )
    )
    records.extend(
        (
            {
                "family": "localized_peak",
                "dimension": 2,
                "method_id": "adaptive_tensor",
                "fun": lambda x: jnp.exp(
                    -localized_alpha * jnp.sum((x - 0.4) ** 2, axis=-1)
                ),
                "truth": localized_1d**2,
                "truth_source": {
                    "kind": "analytic",
                    "formula": "product of finite Gaussian error-function factors",
                },
                "tolerance": 2.0e-7,
                "expected_claim": "localized adaptive-tensor convergence",
            },
            {
                "family": "boundary_layer",
                "dimension": 2,
                "method_id": "tensor_gauss_20",
                "fun": lambda x: jnp.exp(-20.0 * jnp.sum(x, axis=-1)),
                "truth": ((1.0 - math.exp(-20.0)) / 20.0) ** 2,
                "truth_source": {
                    "kind": "analytic",
                    "formula": "((1-exp(-20))/20)^2",
                },
                "tolerance": 2.0e-7,
                "expected_claim": "fixed boundary-layer accuracy threshold",
            },
        )
    )
    return records


def _truth_record(case: dict[str, object]) -> dict[str, object]:
    method = _method(str(case["method_id"]))
    dimension = int(cast(Any, case["dimension"]))
    domain = quad.Hyperrectangle(jnp.zeros(dimension), jnp.ones(dimension))
    controls = _controls(method)
    fun = cast(Callable, case["fun"])

    def solve(amplitude):
        return quad.integrate(
            lambda x, scale: scale * fun(x),
            domain,
            args=amplitude,
            method=method,
            **controls,
        )

    result, tangent = jax.jvp(
        solve,
        (jnp.asarray(1.0),),
        (jnp.asarray(1.0),),
    )
    value = result.value
    replay_gradient = tangent.value
    truth = float(cast(Any, case["truth"]))
    value_float = float(np.asarray(value))
    absolute_error = abs(value_float - truth)
    return {
        "absolute_error": absolute_error,
        "controls": controls,
        "dimension": dimension,
        "dtype": str(np.asarray(result.value).dtype),
        "error_kind": int(np.asarray(result.error.kind)),
        "expected_claim": case["expected_claim"],
        "family": case["family"],
        "method": type(method).__name__,
        "method_id": case["method_id"],
        "relative_error": absolute_error / max(abs(truth), np.finfo(float).tiny),
        "replay_gradient": float(np.asarray(replay_gradient)),
        "status": int(np.asarray(result.status)),
        "tolerance": case["tolerance"],
        "truth": truth,
        "truth_source": case["truth_source"],
        "value": value_float,
        "work": _work(result),
    }


def _raw_solve(fun, lower, upper, *, method, args=()):
    controls = _controls(method)
    return quad.integrate(
        fun,
        quad.Hyperrectangle(jnp.asarray(lower), jnp.asarray(upper)),
        args=args,
        method=method,
        **controls,
    )


def _quantity_solve(fun, axes, *, result_unit, method, args=()):
    controls = _controls(method)
    return quad.integrate(
        fun,
        quad.Hyperrectangle.from_axes(tuple(axes)),
        args=args,
        method=method,
        epsabs=Quantity(controls.pop("epsabs"), result_unit),
        **controls,
    )


def _amplitude_replay_gradient(fun, lower, upper, *, method):
    return jax.grad(
        lambda amplitude: (
            _raw_solve(
                lambda x, scale: scale * fun(x),
                lower,
                upper,
                method=method,
                args=amplitude,
            ).value
        )
    )(jnp.asarray(1.0))


def _astro_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    radius = 3.0
    scale = 1.2
    method = _method("tensor_gauss_20")
    plummer = lambda x, a: (  # noqa: E731
        1.0 / (math.pi * a**2 * (1.0 + (x[..., 0] / a) ** 2) ** 2) * x[..., 0]
    )
    truth = radius**2 / (radius**2 + scale**2)
    raw = _raw_solve(
        plummer,
        [0.0, 0.0],
        [radius, 2.0 * math.pi],
        method=method,
        args=scale,
    )
    quantity = _quantity_solve(
        lambda x, a: Quantity(
            plummer(
                jnp.stack((x.axis(0).to_value(pc), x.axis(1).to_value(rad)), axis=-1),
                a,
            ),
            pc**-1,
        ),
        (
            quad.Axis(Quantity(0.0, pc), Quantity(radius, pc)),
            quad.Axis(Quantity(0.0, rad), Quantity(2.0 * math.pi, rad)),
        ),
        result_unit=dimensionless,
        method=method,
        args=scale,
    )
    gradient = jax.grad(
        lambda a: (
            _raw_solve(
                plummer,
                [0.0, 0.0],
                [radius, 2.0 * math.pi],
                method=method,
                args=a,
            ).value
        )
    )(jnp.asarray(scale))
    gradient_truth = -2.0 * radius**2 * scale / (radius**2 + scale**2) ** 2
    records.append(
        _astro_record(
            "projected_plummer_aperture",
            raw,
            quantity,
            truth,
            gradient,
            gradient_truth,
            "Plummer scale a",
            {
                "bounds": "[0,R] x [0,2*pi]",
                "equation": "Sigma(R_p) R_p; Sigma=1/(pi a^2 (1+(R_p/a)^2)^2)",
                "truth": "R^2/(R^2+a^2)",
                "units": ["pc", "rad"],
            },
        )
    )

    limits = jnp.asarray([1.5, 2.0])
    sigma = jnp.asarray([0.7, 1.1])
    method = _method("tensor_gauss_12")

    def gaussian(x):
        z = x / sigma
        norm = jnp.prod(1.0 / (jnp.sqrt(2.0 * jnp.pi) * sigma))
        return norm * jnp.exp(-0.5 * jnp.sum(z**2, axis=-1))

    mass_factors = jax.scipy.special.erf(limits / (jnp.sqrt(2.0) * sigma))
    mass_truth = float(jnp.prod(mass_factors))
    for moment_axis in (None, 0, 1):
        if moment_axis is None:
            raw_fun = gaussian
        else:
            selected_axis = moment_axis

            def raw_fun(x, axis=selected_axis):
                return x[..., axis] ** 2 * gaussian(x)

        raw = _raw_solve(
            raw_fun,
            -limits,
            limits,
            method=method,
        )
        if moment_axis is None:
            case_truth = mass_truth
            result_unit = dimensionless
            label = "diagonal_gaussian_mass"
        else:
            standardized = limits[moment_axis] / sigma[moment_axis]
            second = sigma[moment_axis] ** 2 * (
                mass_factors[moment_axis]
                - 2.0
                * standardized
                * jnp.exp(-0.5 * standardized**2)
                / jnp.sqrt(2.0 * jnp.pi)
            )
            case_truth = float(second * jnp.prod(mass_factors.at[moment_axis].set(1.0)))
            result_unit = pc**2
            label = f"diagonal_gaussian_second_moment_axis_{moment_axis}"
        density_unit = pc**-2
        quantity = _quantity_solve(
            lambda x, axis=moment_axis: Quantity(
                (
                    gaussian(x.as_quantity(pc).value)
                    if axis is None
                    else x.axis(axis).to_value(pc) ** 2
                    * gaussian(x.as_quantity(pc).value)
                ),
                density_unit if axis is None else dimensionless,
            ),
            tuple(
                quad.Axis(Quantity(-float(limit), pc), Quantity(float(limit), pc))
                for limit in limits
            ),
            result_unit=result_unit,
            method=method,
        )
        records.append(
            _astro_record(
                label,
                raw,
                quantity,
                case_truth,
                _amplitude_replay_gradient(
                    raw_fun,
                    -limits,
                    limits,
                    method=method,
                ),
                case_truth,
                "multiplicative amplitude",
                {
                    "bounds": "product_i [-L_i,L_i]",
                    "equation": (
                        "N(x|0,diag(sigma^2))"
                        if moment_axis is None
                        else f"x_{moment_axis}^2 N(x|0,diag(sigma^2))"
                    ),
                    "truth": "product of erf mass factors and selected truncated second moment",
                    "units": ["pc", "pc"],
                },
            )
        )

    lower = jnp.asarray([0.8, -0.5, 1.0, 2.0])
    upper = jnp.asarray([2.0, 0.3, 4.0, 5.0])
    method = _method("tensor_gauss_3")

    def population(x):
        mass, metallicity, age, distance = jnp.moveaxis(x, -1, 0)
        return mass**2 * (1.0 + metallicity) * age * distance**2

    factors = jnp.asarray(
        [
            (upper[0] ** 3 - lower[0] ** 3) / 3.0,
            (upper[1] - lower[1]) + 0.5 * (upper[1] ** 2 - lower[1] ** 2),
            (upper[2] ** 2 - lower[2] ** 2) / 2.0,
            (upper[3] ** 3 - lower[3] ** 3) / 3.0,
        ]
    )
    truth = float(jnp.prod(factors))
    raw = _raw_solve(population, lower, upper, method=method)
    population_unit = Msun**3 * Myr**2 * pc**3
    quantity = _quantity_solve(
        lambda x: Quantity(
            population(x.values),
            Msun**2 * Myr * pc**2,
        ),
        (
            quad.Axis(Quantity(lower[0], Msun), Quantity(upper[0], Msun)),
            quad.Axis(
                Quantity(lower[1], dimensionless),
                Quantity(upper[1], dimensionless),
            ),
            quad.Axis(Quantity(lower[2], Myr), Quantity(upper[2], Myr)),
            quad.Axis(Quantity(lower[3], pc), Quantity(upper[3], pc)),
        ),
        result_unit=population_unit,
        method=method,
    )
    records.append(
        _astro_record(
            "population_moment",
            raw,
            quantity,
            truth,
            _amplitude_replay_gradient(
                population,
                lower,
                upper,
                method=method,
            ),
            truth,
            "multiplicative amplitude",
            {
                "bounds": "[0.8,2] Msun x [-0.5,0.3] x [1,4] Myr x [2,5] pc",
                "equation": "M^2 (1+Z) t d^2",
                "truth": "product of four polynomial antiderivative factors",
                "units": ["Msun", "1", "Myr", "pc"],
            },
        )
    )

    lower = jnp.asarray([-1.0, 0.0])
    upper = jnp.asarray([2.0, 4.0])
    center = jnp.asarray([0.4, 1.8])
    width = jnp.asarray([0.3, 0.7])
    method = _method("tensor_gauss_20")
    selection = lambda x: jnp.prod(  # noqa: E731
        jax.nn.sigmoid((center - x) / width), axis=-1
    )

    def antiderivative(value):
        return -width * jax.nn.softplus((center - value) / width)

    truth = float(jnp.prod(antiderivative(upper) - antiderivative(lower)))
    raw = _raw_solve(selection, lower, upper, method=method)
    quantity = _quantity_solve(
        lambda x: Quantity(selection(x.values), dimensionless),
        (
            quad.Axis(
                Quantity(lower[0], dimensionless), Quantity(upper[0], dimensionless)
            ),
            quad.Axis(Quantity(lower[1], Myr), Quantity(upper[1], Myr)),
        ),
        result_unit=Myr,
        method=method,
    )
    records.append(
        _astro_record(
            "separable_selection",
            raw,
            quantity,
            truth,
            _amplitude_replay_gradient(
                selection,
                lower,
                upper,
                method=method,
            ),
            truth,
            "multiplicative amplitude",
            {
                "bounds": "[-1,2] x [0,4] Myr",
                "equation": "product_i sigmoid((c_i-x_i)/w_i)",
                "truth": "product of softplus antiderivative differences",
                "units": ["1", "Myr"],
            },
        )
    )
    return records


def _astro_record(
    case_id: str,
    raw,
    quantity,
    truth: float,
    replay_gradient,
    replay_gradient_truth: float,
    replay_parameter: str,
    specification: dict[str, object],
) -> dict[str, object]:
    raw_value = float(np.asarray(raw.value))
    quantity_value = float(np.asarray(quantity.value.value))
    return {
        "absolute_error": abs(raw_value - truth),
        "case_id": case_id,
        "error_kind": int(np.asarray(raw.error.kind)),
        "quantity_unit": str(quantity.value.unit),
        "quantity_value": quantity_value,
        "raw_quantity_absolute_difference": abs(raw_value - quantity_value),
        "raw_value": raw_value,
        "relative_error": abs(raw_value - truth)
        / max(abs(truth), np.finfo(float).tiny),
        "replay_gradient": float(np.asarray(replay_gradient)),
        "replay_gradient_absolute_error": abs(
            float(np.asarray(replay_gradient)) - replay_gradient_truth
        ),
        "replay_gradient_truth": replay_gradient_truth,
        "replay_parameter": replay_parameter,
        "specification": specification,
        "status": int(np.asarray(raw.status)),
        "truth": truth,
        "work": _work(raw),
    }


def build_artifacts() -> tuple[dict[str, object], dict[str, object]]:
    truth_records = [_truth_record(case) for case in _registry()]
    astro_records = _astro_records()
    common = {
        "environment": _environment(),
        "generator": "scripts/generate_quad_multidim_evidence.py",
        "generator_sha256": _sha256(Path(__file__)),
        "schema_version": 1,
    }
    truth = {
        **common,
        "artifact_id": "quad.multidim.truth",
        "claim_boundary": (
            "The artifact certifies only the frozen dimensions, controls, "
            "integrands, and tolerances; it is not a universal accuracy claim."
        ),
        "validation_families": list(VALIDATION_FAMILIES),
        "records": truth_records,
    }
    replay = {
        **common,
        "artifact_id": "quad.multidim.replay-and-astro",
        "claim_boundary": (
            "Replay evidence is first-order only. Astronomy fixtures are "
            "domain-neutral closed-form integrals, not sibling-package models."
        ),
        "astro_records": astro_records,
        "truth_replay_records": [
            {
                "family": record["family"],
                "replay_gradient": record["replay_gradient"],
                "value": record["value"],
            }
            for record in truth_records
        ],
    }
    return truth, replay


def _emit() -> None:
    truth, replay = build_artifacts()
    TRUTH_OUTPUT.write_text(_canonical_json(truth))
    REPLAY_OUTPUT.write_text(_canonical_json(replay))
    print(f"wrote: {TRUTH_OUTPUT.relative_to(ROOT)}")
    print(f"wrote: {REPLAY_OUTPUT.relative_to(ROOT)}")


def _check() -> None:
    truth, replay = build_artifacts()
    expected = {
        TRUTH_OUTPUT: _canonical_json(truth),
        REPLAY_OUTPUT: _canonical_json(replay),
    }
    for path, content in expected.items():
        if not path.is_file() or path.read_text() != content:
            raise ValueError(f"stale: {path.relative_to(ROOT)}")
        print(f"fresh: {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    _emit() if args.emit else _check()


if __name__ == "__main__":
    main()
