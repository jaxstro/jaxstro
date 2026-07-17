"""Matched Jaxstro/Quadax correctness, work, AD, and performance evidence."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import subprocess
import sys
import warnings
from contextlib import nullcontext
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from jaxstro.jaxconfig import enable_high_precision

enable_high_precision()

_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jaxlib  # noqa: E402
import numpy as np  # noqa: E402
import quadax  # noqa: E402
from scripts.quad_benchmark_adapters import (  # noqa: E402
    NormalizedResult,
    RunControls,
    normalize_result,
    portable_numeric,
    raw_jaxstro,
    raw_quadax,
)
from scripts.quad_benchmark_cases import (  # noqa: E402
    BEST_METHODS,
    CASES,
    METHOD_PAIRS,
    BenchmarkCase,
    ComparisonLabel,
    LibraryMethod,
)
from scripts.quad_benchmark_timing import (  # noqa: E402
    make_grad_kernel,
    make_jvp_kernel,
    make_vmap_kernel,
    measure_callable,
    measure_pair_interleaved,
)

from jaxstro import __version__, quad  # noqa: E402
from jaxstro.evidence import (  # noqa: E402
    ComparisonRecord,
    ComparisonRelation,
    EnvironmentRecord,
    EvidenceArtifact,
    EvidenceStatus,
    MetricRecord,
    artifact_from_dict,
    artifact_to_json,
)

REPO_ROOT = _IMPORT_ROOT
OUTPUT = REPO_ROOT / "docs" / "validation" / "quad-performance.json"
REPORT = (
    REPO_ROOT / "docs" / "60-validation" / "numerical" / "quadrature-performance.md"
)
GENERATION_COMMAND = "uv run --group benchmark python scripts/benchmark_quad.py --emit"
FLOAT32_CASES = (
    "smooth_exponential",
    "breakpoint_kink",
    "endpoint_sqrt",
    "semi_infinite_exponential",
    "oscillatory_cosine",
)
PRECISIONS = ("float32", "float64")
VMAP_BATCHES = (16, 128)
TIMING_REPEATS = 21

_CONTROL_VALUES = {
    "float32": RunControls(1.0e-5, 1.0e-5, 64, 16384),
    "float64": RunControls(1.0e-12, 1.0e-12, 64, 16384),
}


def lane_dtype(precision: str) -> jnp.dtype:
    """Return a real enabled JAX dtype for a declared precision lane."""
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision lane: {precision}")
    dtype = jnp.dtype(precision)
    if precision == "float64" and dtype != jnp.dtype("float64"):
        raise RuntimeError("float64 benchmark lane is not enabled")
    return dtype


def _git(*args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def _source_revision() -> str:
    return _git("rev-parse", "HEAD")


def _tree_is_clean() -> bool:
    return not _git("status", "--porcelain")


def _environment() -> dict[str, str]:
    device = jax.devices()[0]
    return {
        "backend": jax.default_backend(),
        "device": str(device),
        "device_kind": getattr(device, "device_kind", "unknown"),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "machine": platform.machine(),
        "numpy_version": np.__version__,
        "operating_system": platform.platform(),
        "processor": platform.processor() or "unreported",
        "python_version": platform.python_version(),
        "quadax_version": quadax.__version__,
    }


def _controls_payload() -> dict[str, Any]:
    return {
        "precision_lanes": {
            precision: asdict(controls)
            for precision, controls in _CONTROL_VALUES.items()
        },
        "timing_repeats": TIMING_REPEATS,
        "vmap_batches": list(VMAP_BATCHES),
        "error_norm": "infinity_norm",
        "timing_policy": "lower_compile_warm_synchronized_interleaved",
        "float32_context": (
            "scoped_x64_disabled_for_quadax_default_width_probe_compatibility"
        ),
    }


def _case_for_precision(case: BenchmarkCase, precision: str) -> BenchmarkCase:
    dtype = lane_dtype(precision)

    def cast(value):
        return jnp.asarray(value, dtype=dtype)

    domain = case.domain
    if isinstance(domain, quad.Interval):
        domain = quad.Interval(
            cast(domain.lower),
            cast(domain.upper),
            breakpoints=tuple(cast(item) for item in domain.breakpoints),
        )
    elif isinstance(domain, quad.RightInfinite):
        domain = quad.RightInfinite(
            cast(domain.lower),
            scale=cast(1.0 if domain.scale is None else domain.scale),
        )
    elif isinstance(domain, quad.LeftInfinite):
        domain = quad.LeftInfinite(
            cast(domain.upper),
            scale=cast(1.0 if domain.scale is None else domain.scale),
        )
    elif isinstance(domain, quad.Infinite):
        domain = quad.Infinite(
            unit=domain.unit,
            scale=cast(1.0 if domain.scale is None else domain.scale),
        )
    return replace(case, domain=domain, theta=float(cast(case.theta)))


def _record_specs(precision: str):
    for original in CASES:
        if precision == "float32" and original.name not in FLOAT32_CASES:
            continue
        case = _case_for_precision(original, precision)
        for pair in METHOD_PAIRS:
            if pair.family not in case.supported_methods:
                continue
            yield {
                "lane": "family_matched",
                "case": case,
                "family": pair.family,
                "pair_variant": pair.variant,
                "comparison_label": pair.label.value,
                "jaxstro_config": pair.jaxstro_config,
                "quadax_config": pair.quadax_config,
                "rationale": pair.note,
            }
        choice = BEST_METHODS[case.name]
        yield {
            "lane": "best_method",
            "case": case,
            "family": choice.jaxstro_method,
            "pair_variant": "practical_choice",
            "comparison_label": "best_method",
            "jaxstro_config": choice.jaxstro_config,
            "quadax_config": choice.quadax_config,
            "rationale": choice.rationale,
        }


def expected_record_keys() -> set[tuple[str, str, str, str, str]]:
    """Return the exact predeclared identity set for deterministic evidence."""
    return {
        (
            spec["lane"],
            spec["case"].name,
            spec["family"].value,
            spec["pair_variant"],
            precision,
        )
        for precision in PRECISIONS
        for spec in _record_specs(precision)
    }


def _infinity_error(measured: Any, truth: Any) -> float:
    measured_array = np.asarray(measured)
    truth_array = np.asarray(truth, dtype=measured_array.dtype)
    return float(np.max(np.abs(measured_array - truth_array)))


def _dtype_tolerances(case: BenchmarkCase, precision: str) -> tuple[float, float]:
    eps = np.finfo(np.dtype(precision)).eps
    return (
        float(max(case.truth_provenance.atol, 64.0 * eps)),
        float(max(case.truth_provenance.rtol, 64.0 * eps)),
    )


def derivative_gate(
    *,
    measured: Any,
    truth: Any,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Judge a derivative independently against declared mathematical truth."""
    measured_array = np.asarray(measured)
    truth_array = np.asarray(truth, dtype=measured_array.dtype)
    finite = bool(np.all(np.isfinite(measured_array)))
    absolute_error = _infinity_error(measured_array, truth_array) if finite else None
    truth_norm = float(np.max(np.abs(truth_array)))
    threshold = float(atol + rtol * truth_norm)
    return {
        "measured": portable_numeric(measured_array),
        "truth": portable_numeric(truth_array),
        "absolute_error": absolute_error,
        "threshold": threshold,
        "passed": bool(
            finite and absolute_error is not None and absolute_error <= threshold
        ),
    }


def _reported_calibration(reported: Any, observed: float | None, precision: str):
    reported_array = np.asarray(reported)
    if reported_array.size:
        reported_scalar = float(np.max(np.abs(reported_array)))
    else:
        reported_scalar = float("nan")
    if not math.isfinite(reported_scalar):
        return {
            "ratio": None,
            "classification": "nonfinite_reported_error",
        }
    if observed is None:
        return {"ratio": None, "classification": "no_finite_truth_error"}
    denominator = float(max(observed, np.finfo(np.dtype(precision)).tiny))
    return {
        "ratio": float(reported_scalar / denominator),
        "classification": "finite_ratio_not_a_bound_claim",
    }


def _normalized_payload(
    result: NormalizedResult,
    truth: Any,
    precision: str,
) -> dict[str, Any]:
    value_array = np.asarray(result.value)
    finite = bool(np.all(np.isfinite(value_array)))
    absolute_error = (
        _infinity_error(value_array, truth) if finite and truth is not None else None
    )
    truth_norm = float(np.max(np.abs(np.asarray(truth)))) if truth is not None else None
    relative_error = (
        absolute_error / max(1.0, truth_norm)
        if absolute_error is not None and truth_norm is not None
        else None
    )
    return {
        "value": portable_numeric(result.value),
        "dtype": str(value_array.dtype),
        "value_finite": finite,
        "reported_error": portable_numeric(result.error),
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "reported_error_calibration": _reported_calibration(
            result.error, absolute_error, precision
        ),
        "converged": result.converged,
        "raw_status": result.raw_status,
        "semantic_status": result.semantic_status,
        "reported_evaluations": result.reported_evaluations,
        "normalized_evaluations": result.normalized_evaluations,
        "refinements": result.refinements,
        "active_regions": result.active_regions,
        "levels": result.levels,
    }


def _library_gate(
    payload: dict[str, Any],
    case: BenchmarkCase,
    precision: str,
) -> dict[str, Any]:
    if case.expected == "fail_closed":
        passed = (
            payload["semantic_status"] == "nonfinite_integrand"
            and not payload["value_finite"]
        )
        return {
            "passed": passed,
            "performance_interpretable": False,
            "classification": (
                "expected_fail_closed" if passed else "false_finite_success"
            ),
            "criterion": "nonfinite input must not be reported as a finite success",
        }
    atol, rtol = _dtype_tolerances(case, precision)
    truth_norm = float(np.max(np.abs(np.asarray(case.truth))))
    threshold = float(atol + rtol * truth_norm)
    accurate = bool(
        payload["value_finite"]
        and payload["absolute_error"] is not None
        and payload["absolute_error"] <= threshold
    )
    honest_nonconvergence = not payload["converged"]
    passed = bool(accurate or honest_nonconvergence)
    classification = "false_success"
    if accurate and payload["converged"]:
        classification = "accurate_convergence"
    elif honest_nonconvergence:
        classification = "honest_nonconvergence"
    return {
        "passed": passed,
        "performance_interpretable": bool(accurate and payload["converged"]),
        "classification": classification,
        "criterion": (
            "accurate convergence warrants timing; honest nonconvergence is valid "
            "failure evidence; false success fails"
        ),
        "threshold": threshold,
    }


def _derivatives(
    case: BenchmarkCase,
    precision: str,
    ours_raw,
    theirs_raw,
    family: LibraryMethod,
) -> dict[str, Any]:
    if case.derivative_truth is None:
        return {
            "available": False,
            "reason": "no_declared_derivative_truth",
        }
    theta = jnp.asarray(case.theta, dtype=lane_dtype(precision))
    atol, rtol = _dtype_tolerances(case, precision)
    jaxstro_jvp = make_jvp_kernel(ours_raw)(theta)[1]
    quadax_jvp = make_jvp_kernel(theirs_raw)(theta)[1]
    payload: dict[str, Any] = {
        "available": True,
        "jaxstro_policy": "accepted_formula_replay",
        "quadax_policy": "adaptive_loop_ad",
        "jvp": {
            "jaxstro": derivative_gate(
                measured=jaxstro_jvp,
                truth=case.derivative_truth,
                atol=atol,
                rtol=rtol,
            ),
            "quadax": derivative_gate(
                measured=quadax_jvp,
                truth=case.derivative_truth,
                atol=atol,
                rtol=rtol,
            ),
        },
    }
    if np.asarray(case.truth).ndim:
        payload["reverse"] = {
            "jaxstro": {"supported": False, "reason": "vector_output"},
            "quadax": {"supported": False, "reason": "vector_output"},
        }
        return payload
    jaxstro_grad = make_grad_kernel(ours_raw)(theta)[2]
    reverse: dict[str, Any] = {
        "jaxstro": {
            "supported": True,
            **derivative_gate(
                measured=jaxstro_grad,
                truth=case.derivative_truth,
                atol=atol,
                rtol=rtol,
            ),
        }
    }
    if family in {LibraryMethod.ROMBERG, LibraryMethod.ROMBERG_TANH_SINH}:
        reverse["quadax"] = {
            "supported": False,
            "reason": "forward_mode_only",
        }
    else:
        quadax_grad = make_grad_kernel(theirs_raw)(theta)[2]
        reverse["quadax"] = {
            "supported": True,
            **derivative_gate(
                measured=quadax_grad,
                truth=case.derivative_truth,
                atol=atol,
                rtol=rtol,
            ),
        }
    payload["reverse"] = reverse
    return payload


def _evaluate_spec_enabled(spec: dict[str, Any], precision: str) -> dict[str, Any]:
    case = spec["case"]
    family = spec["family"]
    controls = _CONTROL_VALUES[precision]
    ours_raw = raw_jaxstro(
        case,
        family,
        controls,
        spec["jaxstro_config"],
    )
    theirs_raw = raw_quadax(
        case,
        family,
        controls,
        spec["quadax_config"],
    )
    theta = jnp.asarray(case.theta, dtype=lane_dtype(precision))
    ours = normalize_result(ours_raw(theta), library="jaxstro", family=family)
    theirs = normalize_result(theirs_raw(theta), library="quadax", family=family)
    ours_payload = _normalized_payload(ours, case.truth, precision)
    theirs_payload = _normalized_payload(theirs, case.truth, precision)
    ours_gate = _library_gate(ours_payload, case, precision)
    theirs_gate = _library_gate(theirs_payload, case, precision)
    derivatives = _derivatives(
        case,
        precision,
        ours_raw,
        theirs_raw,
        family,
    )
    derivatives_passed = not derivatives["available"] or all(
        item["passed"] for item in derivatives["jvp"].values()
    )
    return {
        "lane": spec["lane"],
        "case": case.name,
        "family": family.value,
        "pair_variant": spec["pair_variant"],
        "comparison_label": spec["comparison_label"],
        "precision": precision,
        "rationale": spec["rationale"],
        "truth": portable_numeric(case.truth) if case.truth is not None else None,
        "truth_provenance": asdict(case.truth_provenance),
        "jaxstro": ours_payload,
        "quadax": theirs_payload,
        "derivatives": derivatives,
        "warranted": {
            "jaxstro": ours_gate,
            "quadax": theirs_gate,
            "derivatives_passed": derivatives_passed,
            "performance_interpretable": bool(
                ours_gate["performance_interpretable"]
                and theirs_gate["performance_interpretable"]
                and derivatives_passed
            ),
        },
    }


def _evaluate_spec(spec: dict[str, Any], precision: str) -> dict[str, Any]:
    x64_before = jax.config.x64_enabled
    context = jax.enable_x64(False) if precision == "float32" else nullcontext()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Explicitly requested dtype float64 requested in dtype is not available.*",
            category=UserWarning,
        )
        with context:
            result = _evaluate_spec_enabled(spec, precision)
    if jax.config.x64_enabled != x64_before:
        raise RuntimeError("precision context did not restore the JAX x64 policy")
    return result


@lru_cache(maxsize=1)
def run_deterministic_suite() -> dict[str, Any]:
    """Run every predeclared correctness, work, status, and AD comparison."""
    records = []
    evaluated: dict[tuple[Any, ...], dict[str, Any]] = {}
    for precision in PRECISIONS:
        for spec in _record_specs(precision):
            key = (
                precision,
                spec["case"].name,
                spec["family"].value,
                spec["jaxstro_config"],
                spec["quadax_config"],
            )
            if key not in evaluated:
                if os.environ.get("JAXSTRO_BENCHMARK_PROGRESS") == "1":
                    print(
                        "evaluating",
                        precision,
                        spec["case"].name,
                        spec["family"].value,
                        flush=True,
                    )
                evaluated[key] = _evaluate_spec(spec, precision)
            record = copy.deepcopy(evaluated[key])
            record.update(
                lane=spec["lane"],
                pair_variant=spec["pair_variant"],
                comparison_label=spec["comparison_label"],
                rationale=spec["rationale"],
            )
            records.append(record)
    keys = {
        (
            record["lane"],
            record["case"],
            record["family"],
            record["pair_variant"],
            record["precision"],
        )
        for record in records
    }
    if len(keys) != len(records) or keys != expected_record_keys():
        raise ValueError("quadrature benchmark record identity schema is incomplete")
    return _portable_tree(
        {
            "schema_version": 1,
            "source_revision": _source_revision(),
            "controls": _controls_payload(),
            "records": records,
            "timings": [],
        }
    )


def _portable_tree(value: Any) -> Any:
    """Convert a completed evidence tree to finite built-in JSON scalars."""
    if isinstance(value, dict):
        return {str(key): _portable_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable_tree(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("deterministic evidence contains an unclassified nonfinite")
    return value


def _timing_payload(record) -> dict[str, Any]:
    return asdict(record)


def _pair_timing(functions, argument) -> dict[str, Any]:
    return {
        name: _timing_payload(record)
        for name, record in measure_pair_interleaved(
            functions,
            argument,
            repeats=TIMING_REPEATS,
        ).items()
    }


def _timing_spec_enabled(spec: dict[str, Any], precision: str) -> dict[str, Any]:
    case = spec["case"]
    family = spec["family"]
    controls = _CONTROL_VALUES[precision]
    ours = raw_jaxstro(case, family, controls, spec["jaxstro_config"])
    theirs = raw_quadax(case, family, controls, spec["quadax_config"])
    theta = jax.device_put(jnp.asarray(case.theta, dtype=lane_dtype(precision)))
    timings: dict[str, Any] = {
        "scalar": _pair_timing({"jaxstro": ours, "quadax": theirs}, theta),
        "vmap": {},
        "jvp": _pair_timing(
            {
                "jaxstro": make_jvp_kernel(ours),
                "quadax": make_jvp_kernel(theirs),
            },
            theta,
        ),
        "memory": {
            "status": "unavailable",
            "reason": (
                "No backend-portable peak device-memory metric is available "
                "in this CPU artifact."
            ),
        },
    }
    for batch in VMAP_BATCHES:
        argument = jax.device_put(jnp.full((batch,), theta, dtype=theta.dtype))
        timings["vmap"][str(batch)] = _pair_timing(
            {
                "jaxstro": make_vmap_kernel(ours),
                "quadax": make_vmap_kernel(theirs),
            },
            argument,
        )
    if np.asarray(case.truth).ndim:
        timings["reverse"] = {
            "jaxstro": {"supported": False, "reason": "vector_output"},
            "quadax": {"supported": False, "reason": "vector_output"},
        }
    elif family in {LibraryMethod.ROMBERG, LibraryMethod.ROMBERG_TANH_SINH}:
        timings["reverse"] = {
            "jaxstro": {
                "supported": True,
                **_timing_payload(
                    measure_callable(
                        make_grad_kernel(ours),
                        theta,
                        repeats=TIMING_REPEATS,
                    )
                ),
            },
            "quadax": {"supported": False, "reason": "forward_mode_only"},
        }
    else:
        timings["reverse"] = {
            name: {"supported": True, **payload}
            for name, payload in _pair_timing(
                {
                    "jaxstro": make_grad_kernel(ours),
                    "quadax": make_grad_kernel(theirs),
                },
                theta,
            ).items()
        }
    return {
        "lane": spec["lane"],
        "case": case.name,
        "family": family.value,
        "pair_variant": spec["pair_variant"],
        "precision": precision,
        **timings,
    }


def _timing_spec(spec: dict[str, Any], precision: str) -> dict[str, Any]:
    context = jax.enable_x64(False) if precision == "float32" else nullcontext()
    with context:
        return _timing_spec_enabled(spec, precision)


def run_timing_suite() -> dict[str, Any]:
    """Run informational synchronized timings for every deterministic record."""
    return {
        "schema_version": 1,
        "source_revision": _source_revision(),
        "environment": _environment(),
        "records": [
            _timing_spec(spec, precision)
            for precision in PRECISIONS
            for spec in _record_specs(precision)
        ],
    }


def merge_optimized(
    *,
    baseline: dict[str, Any],
    optimized: dict[str, Any],
    ratios: dict[str, Any],
    contract_parity: bool,
) -> dict[str, Any]:
    """Attach optimized evidence without mutating the immutable baseline."""
    return {
        "schema_version": 1,
        "controls": copy.deepcopy(baseline["controls"]),
        "baseline": copy.deepcopy(baseline),
        "optimized": copy.deepcopy(optimized),
        "ratios": copy.deepcopy(ratios),
        "contract_parity": contract_parity,
        "optimization_decision": {
            "status": "optimized_result_recorded",
            "triggered": True,
        },
    }


def _baseline_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "controls": copy.deepcopy(payload["controls"]),
        "baseline": copy.deepcopy(payload),
        "optimized": None,
        "ratios": None,
        "contract_parity": None,
        "optimization_decision": {
            "status": "baseline_pending_independent_review",
            "triggered": None,
            "reason": "Approved optimization gates are evaluated after fairness review.",
        },
    }


def _metric_identity(record: dict[str, Any], library: str, name: str) -> str:
    parts = (
        record["lane"],
        record["case"],
        record["family"],
        record["pair_variant"],
        record["precision"],
        library,
        name,
    )
    return ".".join(parts)


def build_artifact(payload: dict[str, Any]) -> EvidenceArtifact:
    """Wrap quadrature measurements in the shared portable evidence envelope."""
    metrics: list[MetricRecord] = []
    comparisons: list[ComparisonRecord] = []
    for record in payload["records"]:
        for library in ("jaxstro", "quadax"):
            result = record[library]
            finite_id = _metric_identity(record, library, "value_finite")
            metrics.append(
                MetricRecord(
                    finite_id,
                    "I_finite",
                    int(result["value_finite"]),
                    "dimensionless",
                )
            )
            if record["case"] == "nonfinite_band":
                expected_finite = 0
                status = (
                    EvidenceStatus.PASS
                    if int(result["value_finite"]) == expected_finite
                    else EvidenceStatus.FAIL
                )
                comparisons.append(
                    ComparisonRecord(
                        finite_id + ".gate",
                        finite_id,
                        ComparisonRelation.EQUAL,
                        expected_finite,
                        "dimensionless",
                        status,
                        note="Nonfinite integrand samples must not become finite success evidence.",
                    )
                )
            if result["absolute_error"] is None:
                continue
            error_id = _metric_identity(record, library, "absolute_error")
            threshold = record["warranted"][library]["threshold"]
            error_value = float(result["absolute_error"])
            metrics.append(
                MetricRecord(
                    error_id,
                    "abs(I_hat-I)",
                    error_value,
                    "integral units",
                )
            )
            comparisons.append(
                ComparisonRecord(
                    error_id + ".gate",
                    error_id,
                    ComparisonRelation.LESS_EQUAL,
                    threshold,
                    "integral units",
                    (
                        EvidenceStatus.PASS
                        if error_value <= threshold
                        else EvidenceStatus.FAIL
                    ),
                    note="Compared independently with declared analytic or converged truth.",
                )
            )
    environment = _environment()
    method_payload = _baseline_payload(payload)
    return EvidenceArtifact(
        schema_version="1",
        artifact_id="quad.performance",
        artifact_version="1",
        package_version=__version__,
        source_revision=payload["source_revision"],
        generation_command=GENERATION_COMMAND,
        precision="float32,float64",
        deterministic_config=tuple(sorted(payload["controls"].items())),
        environment=EnvironmentRecord(
            "Machine fields and timings are informational; freshness gates deterministic numerical evidence.",
            tuple(sorted(environment.items())),
        ),
        metrics=tuple(metrics),
        comparisons=tuple(comparisons),
        limitations=(
            "CPU wall time is hardware- and load-dependent and is not a CI gate.",
            "Family-matched labels do not imply identical algorithms or failure semantics.",
            "Jaxstro replay derivatives and Quadax adaptive-loop derivatives have different policies.",
            "No backend-portable peak device-memory metric is claimed.",
            "The nonfinite case intentionally exposes Quadax zero-substitution behavior.",
        ),
        method_payload=method_payload,
    )


def _thaw(value: Any) -> Any:
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _portable_close(stored: Any, current: Any) -> bool:
    stored = _thaw(stored)
    current = _thaw(current)
    if isinstance(stored, dict) and isinstance(current, dict):
        if set(stored) != set(current):
            return False
        return all(_portable_close(stored[key], current[key]) for key in stored)
    if isinstance(stored, list) and isinstance(current, list):
        return len(stored) == len(current) and all(
            _portable_close(left, right) for left, right in zip(stored, current)
        )
    if (
        isinstance(stored, (int, float))
        and not isinstance(stored, bool)
        and isinstance(current, (int, float))
        and not isinstance(current, bool)
    ):
        return math.isclose(
            float(stored),
            float(current),
            rel_tol=1.0e-11,
            abs_tol=1.0e-13,
        )
    return stored == current


def algorithmic_metrics_match(stored: dict[str, Any], current: dict[str, Any]) -> bool:
    """Compare deterministic evidence while ignoring revisions and all timings."""
    try:
        stored_view = {
            "schema_version": stored["schema_version"],
            "controls": stored["controls"],
            "records": stored["records"],
        }
        current_view = {
            "schema_version": current["schema_version"],
            "controls": current["controls"],
            "records": current["records"],
        }
    except (KeyError, TypeError):
        return False
    return _portable_close(stored_view, current_view)


def render_report(artifact: EvidenceArtifact) -> str:
    """Render the authored researcher-facing MyST benchmark report."""
    payload = artifact.method_payload
    active = payload["optimized"] or payload["baseline"]
    records = active["records"]
    warranted = sum(
        bool(record["warranted"]["performance_interpretable"]) for record in records
    )
    labels = {
        label.value: sum(
            record["comparison_label"] == label.value for record in records
        )
        for label in ComparisonLabel
    }
    lines = [
        "# Quadrature performance and comparison evidence",
        "",
        "## Purpose",
        "",
        "This report separates numerical correctness from performance. Every library is judged against declared mathematical truth before any timing ratio is interpreted.",
        "",
        "```{admonition} Reading rule",
        ":class: important",
        "A faster result is not a better result unless its value, status, work accounting, and derivative checks are warranted.",
        "```",
        "",
        "## Comparison label definitions",
        "",
        "| Label | Meaning | Records |",
        "| --- | --- | ---: |",
    ]
    definitions = {
        "exact": "Same embedded rule family and order.",
        "strong_match": "Closely matched global refinement capacity.",
        "node_matched": "Same local node count with different estimators.",
        "family_matched": "Same broad method family; algorithms differ.",
        "capability": "Related capability only; no algorithmic equivalence claim.",
    }
    lines.extend(
        f"| `{label}` | {definitions[label]} | {labels[label]} |"
        for label in definitions
    )
    lines.extend(
        [
            "| `best_method` | Independent practical choice for each library. | "
            + str(
                sum(record["comparison_label"] == "best_method" for record in records)
            )
            + " |",
            "",
            "## Cases and truth",
            "",
            "The catalog includes smooth, vector-valued, localized, nonsmooth, endpoint-singular, improper, oscillatory, expensive, narrow-feature, and nonfinite cases. Truth comes from analytic derivations or an independent NumPy Gauss-Legendre convergence ladder.",
            "",
            "```{math}",
            "\\varepsilon_{\\mathrm{obs}} = \\lVert \\widehat{I} - I \\rVert_{\\infty}.",
            "```",
            "",
            "## Accuracy and calibration",
            "",
            f"{warranted} of {len(records)} records warrant direct performance interpretation. Reported-error ratios are calibration diagnostics, not automatic bound claims.",
            "",
            "## Work",
            "",
            "Reported and normalized evaluations are retained separately. In particular, Quadax Clenshaw-Curtis interval work is converted to actual node evaluations before comparable-work analysis.",
            "",
            "## Compile, warm, VMAP, and AD timing",
            "",
            "Lowering, compilation, warm scalar execution, VMAP batches of 16 and 128, JVP, and supported reverse mode are measured separately with synchronized outputs and interleaved library order.",
            "",
            "```{admonition} Timing scope",
            ":class: note",
            "Wall time is informational for this recorded CPU environment and is never used as a deterministic freshness gate.",
            "```",
            "",
            "## Failure semantics",
            "",
            "Jaxstro fails closed on nonfinite integrand samples. Quadax 0.2.13 masks nonfinite samples to zero, so that case is recorded as a semantic difference and excluded from performance claims.",
            "",
            "## Environment",
            "",
            f"Source revision: `{artifact.source_revision}`",
            "",
        ]
    )
    lines.extend(f"- `{key}`: `{value}`" for key, value in artifact.environment.values)
    decision = payload["optimization_decision"]
    lines.extend(
        [
            "",
            "## Optimization decision",
            "",
            f"Status: `{decision['status']}`.",
            "",
            str(decision.get("reason", "See recorded trigger ratios.")),
            "",
            "## Warranted limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in artifact.limitations)
    lines.append("")
    return "\n".join(lines)


def _active_suite(artifact: EvidenceArtifact) -> dict[str, Any]:
    payload = artifact.method_payload
    return payload["optimized"] or payload["baseline"]


def _write_artifact(artifact: EvidenceArtifact) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(artifact_to_json(artifact), encoding="utf-8")
    REPORT.write_text(render_report(artifact), encoding="utf-8")


def _check_mode() -> int:
    if not OUTPUT.exists() or not REPORT.exists():
        print("quadrature performance evidence is missing or stale")
        return 1
    try:
        stored = artifact_from_dict(json.loads(OUTPUT.read_text(encoding="utf-8")))
        current = run_deterministic_suite()
        healthy = algorithmic_metrics_match(_active_suite(stored), current)
        healthy = healthy and REPORT.read_text(encoding="utf-8") == render_report(
            stored
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        healthy = False
    if not healthy:
        print("quadrature performance evidence is missing or stale")
        return 1
    print(
        "quadrature performance evidence healthy: deterministic metrics match fresh run"
    )
    return 0


def _require_clean_tree() -> None:
    if not _tree_is_clean():
        raise RuntimeError(
            "authoritative quadrature timings require a clean source tree"
        )


def _timing_only(path: Path) -> int:
    _require_clean_tree()
    destination = path.expanduser().resolve()
    try:
        destination.relative_to(REPO_ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("--timing-only output must be outside the repository")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(run_timing_suite(), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote timing confirmation to {destination}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--timing-only", type=Path, metavar="PATH")
    args = parser.parse_args()
    if args.check:
        return _check_mode()
    if args.timing_only is not None:
        return _timing_only(args.timing_only)
    _require_clean_tree()
    baseline = copy.deepcopy(run_deterministic_suite())
    timing = run_timing_suite()
    baseline["timings"] = timing["records"]
    artifact = build_artifact(baseline)
    _write_artifact(artifact)
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)} and {REPORT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
