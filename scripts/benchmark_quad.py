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
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

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
CONFIRMATION_OUTPUT = (
    REPO_ROOT / "docs" / "validation" / "quad-performance-confirmation.json"
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
REPRESENTATIVE_CASES = (
    "smooth_exponential",
    "localized_gaussian",
    "breakpoint_kink",
    "endpoint_sqrt",
    "semi_infinite_exponential",
    "oscillatory_cosine",
    "expensive_identity",
)
RATIO_ELIGIBLE_LABELS = ("exact", "strong_match", "node_matched")
WARM_RATIO_TRIGGER = 1.25
COMPILE_RATIO_TRIGGER = 2.0
WORK_RATIO_TRIGGER = 1.50
VMAP_RATIO_TRIGGER = 1.50
AD_RATIO_TRIGGER = 1.50
MIN_WARM_CASES = 3
MIN_COMPILE_CASES = 2
MIN_OTHER_CASES = 3

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


def _cpu_model() -> str:
    identifiers = []
    for key in ("machdep.cpu.brand_string", "hw.model"):
        completed = subprocess.run(
            ("sysctl", "-n", key),
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            identifiers.append(completed.stdout.strip())
    if identifiers:
        return " / ".join(identifiers)
    return platform.processor() or platform.uname().processor or "unreported"


def _environment() -> dict[str, str]:
    device = jax.devices()[0]
    return {
        "backend": jax.default_backend(),
        "cpu_model": _cpu_model(),
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
        "timing_policy": (
            "fresh_process_per_record_lower_compile_warm_synchronized_interleaved"
        ),
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
    derivatives_passed = bool(
        derivatives["available"]
        and all(item["passed"] for item in derivatives["jvp"].values())
    )
    primal_performance_interpretable = bool(
        ours_gate["performance_interpretable"]
        and theirs_gate["performance_interpretable"]
    )
    jvp_performance_interpretable = bool(
        primal_performance_interpretable and derivatives_passed
    )
    reverse = derivatives.get("reverse", {})
    reverse_performance_interpretable = bool(
        primal_performance_interpretable
        and derivatives["available"]
        and reverse
        and all(
            item.get("supported", False) and item.get("passed", False)
            for item in reverse.values()
        )
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
            "primal_performance_interpretable": (primal_performance_interpretable),
            "jvp_performance_interpretable": jvp_performance_interpretable,
            "reverse_performance_interpretable": (reverse_performance_interpretable),
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


def _timing_record(precision: str, index: int) -> dict[str, Any]:
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision lane: {precision}")
    specs = list(_record_specs(precision))
    if not 0 <= index < len(specs):
        raise IndexError(f"timing record index {index} is out of range")
    return {
        **_timing_spec(specs[index], precision),
        "process_isolation": "fresh_process_per_record",
    }


def _timing_record_subprocess(precision: str, index: int) -> dict[str, Any]:
    completed = subprocess.run(
        (
            sys.executable,
            str(Path(__file__).resolve()),
            "--timing-record",
            precision,
            str(index),
        ),
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(completed.stdout)


def run_timing_suite() -> dict[str, Any]:
    """Measure every record in a fresh process to isolate compilation caches."""
    run_id = str(uuid4())
    started_utc = datetime.now(timezone.utc).isoformat()
    indexed_specs = [
        (precision, index, spec)
        for precision in PRECISIONS
        for index, spec in enumerate(_record_specs(precision))
    ]
    records = []
    for sequence, (precision, index, spec) in enumerate(indexed_specs, start=1):
        if os.environ.get("JAXSTRO_BENCHMARK_PROGRESS") == "1":
            print(
                "timing",
                f"{sequence}/{len(indexed_specs)}",
                precision,
                spec["case"].name,
                spec["family"].value,
                flush=True,
            )
        records.append(_timing_record_subprocess(precision, index))
    return {
        "schema_version": 1,
        "run_id": run_id,
        "started_utc": started_utc,
        "source_revision": _source_revision(),
        "controls": _controls_payload(),
        "environment": _environment(),
        "process_isolation": "fresh_process_per_record",
        "records": records,
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
            "reason": (
                "Optimized evidence preserves the reviewed baseline and passes "
                "the scientific contract-parity gate."
            ),
        },
    }


def _optimization_ratio_summary(
    baseline: dict[str, Any],
    optimized: dict[str, Any],
) -> dict[str, Any]:
    baseline_timings = {
        _record_identity(record): record for record in baseline["timings"]
    }
    optimized_timings = {
        _record_identity(record): record for record in optimized["timings"]
    }
    targets = []
    for record in baseline["records"]:
        if not (
            record["precision"] == "float64"
            and record["lane"] == "family_matched"
            and record["family"] == "romberg"
            and record["case"]
            in {"smooth_exponential", "oscillatory_cosine", "expensive_identity"}
        ):
            continue
        key = _record_identity(record)
        before = baseline_timings[key]
        after = optimized_timings[key]
        modes = {
            "scalar": (before["scalar"], after["scalar"]),
            "vmap_16": (before["vmap"]["16"], after["vmap"]["16"]),
            "vmap_128": (before["vmap"]["128"], after["vmap"]["128"]),
            "jvp": (before["jvp"], after["jvp"]),
        }
        targets.append(
            {
                "record": _trigger_identity(record),
                "modes": {
                    mode: {
                        "baseline_jaxstro_seconds": pair_before["jaxstro"][
                            "median_warm_seconds"
                        ],
                        "optimized_jaxstro_seconds": pair_after["jaxstro"][
                            "median_warm_seconds"
                        ],
                        "jaxstro_speedup": pair_before["jaxstro"]["median_warm_seconds"]
                        / pair_after["jaxstro"]["median_warm_seconds"],
                        "optimized_jaxstro_to_quadax_ratio": pair_after["jaxstro"][
                            "median_warm_seconds"
                        ]
                        / pair_after["quadax"]["median_warm_seconds"],
                    }
                    for mode, (pair_before, pair_after) in modes.items()
                },
            }
        )
    return {
        "policy": "reviewed_baseline_over_optimized_same_host",
        "targets": targets,
        "baseline_trigger_assessment": baseline.get("trigger_assessment"),
        "optimized_trigger_assessment": optimized.get("trigger_assessment"),
    }


def _measurement_owner_equivalent(before: str, after: str) -> bool:
    completed = subprocess.run(
        (
            "git",
            "diff",
            "--quiet",
            before,
            after,
            "--",
            "src/jaxstro/quad",
            "scripts/benchmark_quad.py",
            "scripts/quad_benchmark_adapters.py",
            "scripts/quad_benchmark_cases.py",
            "scripts/quad_benchmark_timing.py",
            "pyproject.toml",
            "uv.lock",
        ),
        cwd=REPO_ROOT,
        check=False,
    )
    return completed.returncode == 0


def _timing_materially_slower(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    before = float(baseline["median_warm_seconds"])
    after = float(candidate["median_warm_seconds"])
    return bool(
        after > 1.25 * before
        and after - before
        > 2.0
        * max(
            float(baseline["mad_warm_seconds"]),
            float(candidate["mad_warm_seconds"]),
        )
    )


def _optimized_confirmation_summary(
    payload: dict[str, Any],
    confirmation: dict[str, Any],
) -> dict[str, Any]:
    baseline = payload["baseline"]
    optimized = payload["optimized"]
    confirmation_suite = {
        "timings": confirmation["records"],
        "trigger_assessment": None,
    }
    second_ratios = _optimization_ratio_summary(baseline, confirmation_suite)
    first_by_record = {item["record"]: item for item in payload["ratios"]["targets"]}
    second_by_record = {item["record"]: item for item in second_ratios["targets"]}
    required = {
        "smooth_exponential.romberg.divmax10.float64",
        "oscillatory_cosine.romberg.divmax10.float64",
        "expensive_identity.romberg.divmax10.float64",
    }
    vmap_improves_both = all(
        first_by_record[record]["modes"]["vmap_128"]["jaxstro_speedup"] > 1.0
        and second_by_record[record]["modes"]["vmap_128"]["jaxstro_speedup"] > 1.0
        for record in required
    )

    def timing_map(suite, timing_key="timings"):
        return {_record_identity(record): record for record in suite[timing_key]}

    baseline_timings = timing_map(baseline)
    first_timings = timing_map(optimized)
    second_timings = timing_map(confirmation, "records")
    reproducible_regressions = []
    for contract in baseline["records"]:
        if contract["family"] != "romberg":
            continue
        key = _record_identity(contract)
        for mode in ("scalar", "jvp"):
            warrant_key = (
                "primal_performance_interpretable"
                if mode == "scalar"
                else "jvp_performance_interpretable"
            )
            if not contract["warranted"][warrant_key]:
                continue
            before = baseline_timings[key][mode]["jaxstro"]
            first = first_timings[key][mode]["jaxstro"]
            second = second_timings[key][mode]["jaxstro"]
            if _timing_materially_slower(before, first) and _timing_materially_slower(
                before, second
            ):
                reproducible_regressions.append(
                    {"record": _trigger_identity(contract), "mode": mode}
                )
    source_revision_distinct = (
        confirmation["source_revision"] != optimized["source_revision"]
    )
    suite_run_ids_distinct = bool(
        optimized["timing_run_id"]
        and confirmation["run_id"]
        and optimized["timing_run_id"] != confirmation["run_id"]
    )
    expected_identities = [_record_identity(record) for record in optimized["timings"]]
    confirmation_identities = [
        _record_identity(record) for record in confirmation["records"]
    ]
    identity_set_exact = bool(
        len(confirmation_identities) == len(expected_identities)
        and len(set(confirmation_identities)) == len(confirmation_identities)
        and set(confirmation_identities) == set(expected_identities)
    )
    measurement_owner_equivalent = _measurement_owner_equivalent(
        optimized["source_revision"], confirmation["source_revision"]
    )
    controls_match = confirmation["controls"] == optimized["controls"]
    environment_match = confirmation["environment"] == optimized["timing_environment"]
    process_isolation = confirmation["process_isolation"]
    accepted = bool(
        vmap_improves_both
        and not reproducible_regressions
        and source_revision_distinct
        and suite_run_ids_distinct
        and identity_set_exact
        and measurement_owner_equivalent
        and controls_match
        and environment_match
        and process_isolation == "fresh_process_per_record"
    )
    return {
        "source_revision": confirmation["source_revision"],
        "run_id": confirmation["run_id"],
        "source_revision_distinct": source_revision_distinct,
        "suite_run_ids_distinct": suite_run_ids_distinct,
        "identity_set_exact": identity_set_exact,
        "measurement_owner_equivalent": measurement_owner_equivalent,
        "controls_match": controls_match,
        "environment_match": environment_match,
        "process_isolation": process_isolation,
        "targets": second_ratios["targets"],
        "vmap_128_improves_all_targets_in_both_suites": vmap_improves_both,
        "regression_scope": "all_contract_warranted_romberg_scalar_and_jvp_records",
        "reproducible_scalar_or_jvp_regressions": reproducible_regressions,
        "accepted": accepted,
    }


def _trigger_identity(record: dict[str, Any]) -> str:
    return ".".join(
        (
            record["case"],
            record["family"],
            record["pair_variant"],
            record["precision"],
        )
    )


def _ratio_regression(
    pair: dict[str, dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    ours = pair["jaxstro"]
    theirs = pair["quadax"]
    denominator = float(theirs["median_warm_seconds"])
    ratio = (
        float(ours["median_warm_seconds"]) / denominator
        if denominator > 0.0
        else math.inf
    )
    separated = float(ours["median_warm_seconds"]) - denominator > 2.0 * max(
        float(ours["mad_warm_seconds"]),
        float(theirs["mad_warm_seconds"]),
    )
    return {
        "ratio": ratio,
        "threshold": threshold,
        "two_mad_separated": separated,
        "trigger_case": bool(ratio > threshold and separated),
    }


def evaluate_optimization_triggers(
    baseline: dict[str, Any],
    *,
    compile_confirmation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the frozen primary-lane optimization policy mechanically."""
    timing_by_key = {_record_identity(record): record for record in baseline["timings"]}
    eligible = [
        record
        for record in baseline["records"]
        if record["precision"] == "float64"
        and record["lane"] == "family_matched"
        and record["case"] in REPRESENTATIVE_CASES
        and record["comparison_label"] in RATIO_ELIGIBLE_LABELS
        and record["warranted"]["primal_performance_interpretable"]
    ]
    warm: list[dict[str, Any]] = []
    compile_candidates: list[dict[str, Any]] = []
    work: list[dict[str, Any]] = []
    vmap = {str(batch): [] for batch in VMAP_BATCHES}
    ad = {"jvp": [], "reverse": []}
    for record in eligible:
        timing = timing_by_key[_record_identity(record)]
        identity = _trigger_identity(record)
        warm_result = _ratio_regression(timing["scalar"], WARM_RATIO_TRIGGER)
        if warm_result["trigger_case"]:
            warm.append({"record": identity, **warm_result})

        ours_compile = float(timing["scalar"]["jaxstro"]["compile_seconds"])
        theirs_compile = float(timing["scalar"]["quadax"]["compile_seconds"])
        compile_ratio = (
            ours_compile / theirs_compile if theirs_compile > 0.0 else math.inf
        )
        if compile_ratio > COMPILE_RATIO_TRIGGER:
            compile_candidates.append(
                {
                    "record": identity,
                    "ratio": compile_ratio,
                    "threshold": COMPILE_RATIO_TRIGGER,
                }
            )

        theirs_work = int(record["quadax"]["normalized_evaluations"])
        if theirs_work > 0:
            work_ratio = int(record["jaxstro"]["normalized_evaluations"]) / theirs_work
            if work_ratio > WORK_RATIO_TRIGGER:
                work.append(
                    {
                        "record": identity,
                        "ratio": work_ratio,
                        "threshold": WORK_RATIO_TRIGGER,
                    }
                )

        for batch in VMAP_BATCHES:
            result = _ratio_regression(timing["vmap"][str(batch)], VMAP_RATIO_TRIGGER)
            if result["trigger_case"]:
                vmap[str(batch)].append({"record": identity, **result})

        if record["warranted"]["jvp_performance_interpretable"]:
            result = _ratio_regression(timing["jvp"], AD_RATIO_TRIGGER)
            if result["trigger_case"]:
                ad["jvp"].append({"record": identity, **result})
        if record["warranted"]["reverse_performance_interpretable"]:
            result = _ratio_regression(timing["reverse"], AD_RATIO_TRIGGER)
            if result["trigger_case"]:
                ad["reverse"].append({"record": identity, **result})

    confirmed_compile: list[dict[str, Any]] = []
    confirmation_compatible = False
    if compile_confirmation is not None:
        confirmation_compatible = bool(
            compile_confirmation.get("source_revision")
            == baseline.get("source_revision")
            and compile_confirmation.get("process_isolation")
            == "fresh_process_per_record"
            and compile_confirmation.get("controls") == baseline.get("controls")
            and compile_confirmation.get("environment")
            == baseline.get("timing_environment")
        )
        if confirmation_compatible:
            confirmation_timings = {
                _record_identity(record): record
                for record in compile_confirmation["records"]
            }
            for candidate in compile_candidates:
                source = next(
                    record
                    for record in eligible
                    if _trigger_identity(record) == candidate["record"]
                )
                pair = confirmation_timings[_record_identity(source)]["scalar"]
                denominator = float(pair["quadax"]["compile_seconds"])
                ratio = (
                    float(pair["jaxstro"]["compile_seconds"]) / denominator
                    if denominator > 0.0
                    else math.inf
                )
                if ratio > COMPILE_RATIO_TRIGGER:
                    confirmed_compile.append(
                        {
                            **candidate,
                            "confirmation_ratio": ratio,
                        }
                    )

    fired = []
    if len(warm) >= MIN_WARM_CASES:
        fired.append("warm")
    if len(work) >= MIN_OTHER_CASES:
        fired.append("work")
    fired.extend(
        f"vmap_{batch}"
        for batch, cases in vmap.items()
        if len(cases) >= MIN_OTHER_CASES
    )
    fired.extend(
        f"ad_{mode}" for mode, cases in ad.items() if len(cases) >= MIN_OTHER_CASES
    )
    if len(confirmed_compile) >= MIN_COMPILE_CASES:
        fired.append("compile")
    compile_review_required = bool(
        len(compile_candidates) >= MIN_COMPILE_CASES
        and len(confirmed_compile) < MIN_COMPILE_CASES
    )
    status = "optimization_required" if fired else "no_optimization_required"
    if compile_review_required and not fired:
        status = "review_required"
    return _portable_tree(
        {
            "policy": "primary_float64_family_matched_representative",
            "eligible_records": len(eligible),
            "warm_regression_cases_over_25_percent": warm,
            "compile_or_memory_cases_over_2x": compile_candidates,
            "compile_confirmation_compatible": confirmation_compatible,
            "confirmed_compile_cases_over_2x": confirmed_compile,
            "work_inefficiency_cases": work,
            "vmap_regression_cases": vmap,
            "ad_regression_cases": ad,
            "fired_triggers": fired,
            "decision": status,
        }
    )


def _baseline_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("timings"):
        assessment = evaluate_optimization_triggers(payload)
    else:
        assessment = {
            "fired_triggers": [],
            "decision": "baseline_pending_timings",
        }
    return {
        "schema_version": 1,
        "controls": copy.deepcopy(payload["controls"]),
        "baseline": {**copy.deepcopy(payload), "trigger_assessment": assessment},
        "optimized": None,
        "ratios": None,
        "contract_parity": None,
        "optimization_decision": {
            "status": assessment["decision"],
            "triggered": bool(assessment["fired_triggers"]),
            "reason": (
                "The frozen primary-lane trigger assessment is recorded in the "
                "baseline payload. Profile before changing runtime code."
            ),
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


def build_optimized_artifact(
    payload: dict[str, Any],
    reviewed: EvidenceArtifact,
) -> EvidenceArtifact:
    """Attach a clean optimized run while retaining the reviewed baseline."""
    current = build_artifact(payload)
    reviewed_payload = _thaw(reviewed.method_payload)
    baseline = reviewed_payload["baseline"]
    optimized = {
        **copy.deepcopy(payload),
        "trigger_assessment": evaluate_optimization_triggers(payload),
    }
    contract_parity = deterministic_contracts_match(baseline, optimized)
    if not contract_parity:
        raise RuntimeError(
            "optimized quadrature evidence changes a reviewed scientific contract"
        )
    method_payload = merge_optimized(
        baseline=baseline,
        optimized=optimized,
        ratios=_optimization_ratio_summary(baseline, optimized),
        contract_parity=contract_parity,
    )
    return replace(current, method_payload=method_payload)


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


def _derivative_contract(derivatives: dict[str, Any]) -> dict[str, Any]:
    if not derivatives["available"]:
        return {
            "available": False,
            "reason": derivatives["reason"],
        }
    return {
        "available": True,
        "jaxstro_policy": derivatives["jaxstro_policy"],
        "quadax_policy": derivatives["quadax_policy"],
        "jvp": {
            library: {
                "passed": result["passed"],
                "truth": result["truth"],
                "threshold": result["threshold"],
            }
            for library, result in derivatives["jvp"].items()
        },
        "reverse": {
            library: {
                "supported": result["supported"],
                "passed": result.get("passed"),
                "reason": result.get("reason"),
                "truth": result.get("truth"),
                "threshold": result.get("threshold"),
            }
            for library, result in derivatives["reverse"].items()
        },
    }


def _record_contract(record: dict[str, Any]) -> dict[str, Any]:
    exact_result_keys = (
        "dtype",
        "converged",
        "raw_status",
        "semantic_status",
        "value_finite",
        "reported_evaluations",
        "normalized_evaluations",
        "refinements",
        "active_regions",
        "levels",
    )
    return {
        "identity": _record_identity(record),
        "comparison_label": record["comparison_label"],
        "truth": record["truth"],
        "truth_provenance": record["truth_provenance"],
        "results": {
            library: {key: record[library][key] for key in exact_result_keys}
            for library in ("jaxstro", "quadax")
        },
        "warrants": {
            library: {
                key: record["warranted"][library][key]
                for key in (
                    "passed",
                    "performance_interpretable",
                    "classification",
                    "criterion",
                    "threshold",
                )
                if key in record["warranted"][library]
            }
            for library in ("jaxstro", "quadax")
        }
        | {
            key: record["warranted"][key]
            for key in (
                "derivatives_passed",
                "primal_performance_interpretable",
                "jvp_performance_interpretable",
                "reverse_performance_interpretable",
            )
        },
        "derivatives": _derivative_contract(record["derivatives"]),
    }


def deterministic_contracts_match(
    baseline: dict[str, Any],
    optimized: dict[str, Any],
) -> bool:
    """Compare scientific contracts while allowing truth-gated roundoff drift."""
    baseline = _thaw(baseline)
    optimized = _thaw(optimized)
    if baseline.get("schema_version") != optimized.get("schema_version"):
        return False
    if baseline.get("controls") != optimized.get("controls"):
        return False
    baseline_records = baseline.get("records", [])
    optimized_records = optimized.get("records", [])
    if len(baseline_records) != len(optimized_records):
        return False
    return all(
        _record_contract(before) == _record_contract(after)
        for before, after in zip(
            baseline_records,
            optimized_records,
            strict=True,
        )
    )


def _record_identity(record: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        record["lane"],
        record["case"],
        record["family"],
        record["pair_variant"],
        record["precision"],
    )


def _timing_measurements(record: dict[str, Any]):
    for mode in ("scalar", "jvp"):
        for library, measurement in record[mode].items():
            yield mode, library, measurement
    for batch, pair in record["vmap"].items():
        for library, measurement in pair.items():
            yield f"vmap_{batch}", library, measurement
    for library, measurement in record["reverse"].items():
        if measurement.get("supported", False):
            yield "reverse", library, measurement


def _timing_ratio_cell(
    pair: dict[str, dict[str, Any]],
    *,
    warranted: bool,
    compile_time: bool = False,
) -> str:
    if not warranted:
        return "not warranted"
    ours = pair["jaxstro"]
    theirs = pair["quadax"]
    field = "compile_seconds" if compile_time else "median_warm_seconds"
    denominator = float(theirs[field])
    if denominator <= 0.0:
        return "not measurable"
    ratio = float(ours[field]) / denominator
    if compile_time:
        return f"{ratio:.2f}"
    separated = float(ours[field]) - denominator > 2.0 * max(
        float(ours["mad_warm_seconds"]),
        float(theirs["mad_warm_seconds"]),
    )
    suffix = "separated" if separated else "not separated"
    return f"{ratio:.2f} ({suffix})"


def _work_ratio_cell(record: dict[str, Any]) -> str:
    if not record["warranted"]["primal_performance_interpretable"]:
        return "not warranted"
    denominator = int(record["quadax"]["normalized_evaluations"])
    if denominator <= 0:
        return "not comparable"
    return f"{int(record['jaxstro']['normalized_evaluations']) / denominator:.2f}"


def render_report(artifact: EvidenceArtifact) -> str:
    """Render the authored researcher-facing MyST benchmark report."""
    payload = artifact.method_payload
    active = payload["optimized"] or payload["baseline"]
    records = active["records"]
    primal_warranted = sum(
        bool(record["warranted"]["primal_performance_interpretable"])
        for record in records
    )
    jvp_warranted = sum(
        bool(record["warranted"]["jvp_performance_interpretable"]) for record in records
    )
    reverse_warranted = sum(
        bool(record["warranted"]["reverse_performance_interpretable"])
        for record in records
    )
    derivative_failures = sum(
        record["derivatives"]["available"]
        and not record["warranted"]["derivatives_passed"]
        for record in records
    )
    timings = {_record_identity(record): record for record in active.get("timings", [])}
    timing_measurements = [
        measurement
        for timing in timings.values()
        for _, _, measurement in _timing_measurements(timing)
    ]
    stable_measurements = sum(
        measurement["median_warm_seconds"] > 0.0
        and measurement["mad_warm_seconds"] / measurement["median_warm_seconds"] <= 0.10
        for measurement in timing_measurements
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
            "| `best_method` | Predeclared practical choice using the frozen library-specific adapter settings. | "
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
            f"{primal_warranted} of {len(records)} records warrant primal timing interpretation. {jvp_warranted} warrant JVP timing, and {reverse_warranted} warrant a direct two-library reverse-mode comparison.",
            "",
            f"There are {derivative_failures} records with available derivative truth that fail at least one JVP gate. Records without declared derivative truth are explicitly ineligible for AD comparisons. Reported-error ratios are calibration diagnostics, not automatic bound claims.",
            "",
            "## Work",
            "",
            "Reported and normalized evaluations are retained separately. In particular, Quadax Clenshaw-Curtis interval work is converted to actual node evaluations before comparable-work analysis.",
            "",
            "## Compile, warm, VMAP, and AD timing",
            "",
            "Lowering, compilation, warm scalar execution, VMAP batches of 16 and 128, JVP, and supported reverse mode are measured separately with synchronized outputs and interleaved library order. Every method-case record is measured in a fresh Python process so internal compilation caches cannot leak between records.",
            "",
            f"Using a descriptive stability threshold of $\\operatorname{{MAD}}/\\operatorname{{median}} \\le 0.10$, {stable_measurements} of {len(timing_measurements)} supported timed library-mode measurements are stable. Automatic regression decisions use the stricter predeclared ratio, minimum-case, and two-MAD separation rules.",
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
        ]
    )
    eligible_labels = {
        ComparisonLabel.EXACT.value,
        ComparisonLabel.STRONG_MATCH.value,
        ComparisonLabel.NODE_MATCHED.value,
    }
    representative = {
        "smooth_exponential",
        "localized_gaussian",
        "breakpoint_kink",
        "endpoint_sqrt",
        "semi_infinite_exponential",
        "oscillatory_cosine",
        "expensive_identity",
    }
    primary = [
        record
        for record in records
        if record["precision"] == "float64"
        and record["lane"] == "family_matched"
        and record["case"] in representative
        and record["comparison_label"] in eligible_labels
    ]
    if timings:
        lines.extend(
            [
                "## Primary matched timing ratios",
                "",
                "Each timing ratio is $t_{\\mathrm{jaxstro}}/t_{\\mathrm{quadax}}$; each work ratio is $N_{\\mathrm{jaxstro}}/N_{\\mathrm{quadax}}$. Values above one therefore favor Quadax for that metric. The parenthetical timing label states whether the Jaxstro slowdown exceeds twice the larger MAD; it is not a winner declaration.",
                "",
                "| Case | Family | Compile | Warm | VMAP 128 | JVP | Work |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for record in primary:
            timing = timings[_record_identity(record)]
            primal = record["warranted"]["primal_performance_interpretable"]
            jvp = record["warranted"]["jvp_performance_interpretable"]
            lines.append(
                "| "
                + " | ".join(
                    (
                        f"`{record['case']}`",
                        f"`{record['family']}`",
                        _timing_ratio_cell(
                            timing["scalar"],
                            warranted=primal,
                            compile_time=True,
                        ),
                        _timing_ratio_cell(timing["scalar"], warranted=primal),
                        _timing_ratio_cell(timing["vmap"]["128"], warranted=primal),
                        _timing_ratio_cell(timing["jvp"], warranted=jvp),
                        _work_ratio_cell(record),
                    )
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "```{admonition} Scope of this table",
                ":class: note",
                "The table is restricted to the predeclared primary float64, family-matched, representative, ratio-eligible lane. Capability-only and practical-choice records remain in the machine-readable artifact but cannot drive matched-method superiority claims.",
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Environment",
            "",
            f"Source revision: `{artifact.source_revision}`",
            "",
        ]
    )
    lines.extend(f"- `{key}`: `{value}`" for key, value in artifact.environment.values)
    decision = payload["optimization_decision"]
    decision_lines = [
        "",
        "## Optimization decision",
        "",
        f"Status: `{decision['status']}`.",
        "",
        str(decision.get("reason", "See recorded trigger ratios.")),
        "",
    ]
    authorization = decision.get("authorization")
    active_triggers = active.get("trigger_assessment", {}).get("fired_triggers", [])
    if authorization:
        decision_lines.extend(
            [
                f"Reviewed baseline authorization: `{authorization['trigger']}`.",
                "",
                str(authorization["basis"]),
                "",
                "Post-optimization residual triggers: "
                + (", ".join(f"`{item}`" for item in active_triggers) or "none")
                + ". These are observations from the accepted optimized suite, "
                "not additional authorization for the implemented change.",
            ]
        )
    else:
        decision_lines.append(
            "Fired triggers: "
            + (", ".join(f"`{item}`" for item in active_triggers) or "none")
            + "."
        )
    lines.extend(decision_lines)
    confirmation = (payload.get("ratios") or {}).get("optimized_confirmation")
    if confirmation is not None:
        first_targets = {item["record"]: item for item in payload["ratios"]["targets"]}
        lines.extend(
            [
                "",
                "## Two-suite optimization acceptance",
                "",
                "The reviewed baseline is preserved unchanged. The two optimized suites have distinct generated run identifiers and distinct source revisions, with the exact same unique record set, unchanged runtime and measurement owners, matching controls and hardware, and per-record process isolation.",
                "",
                "| Romberg case | Suite 1 VMAP-128 speedup | Suite 2 VMAP-128 speedup |",
                "| --- | ---: | ---: |",
            ]
        )
        for item in confirmation["targets"]:
            record = item["record"]
            case = record.split(".", maxsplit=1)[0]
            first = first_targets[record]["modes"]["vmap_128"]["jaxstro_speedup"]
            second = item["modes"]["vmap_128"]["jaxstro_speedup"]
            lines.append(f"| `{case}` | {first:.2f} | {second:.2f} |")
        lines.extend(
            [
                "",
                "```{admonition} Acceptance rule",
                ":class: important",
                "All three VMAP-128 targets improve in both suites. No scalar or JVP slowdown above 25 percent and separated by more than twice the larger MAD reproduces in both suites.",
                "```",
            ]
        )
    lines.extend(["", "## Warranted limitations", ""])
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


def _confirm_optimized(path: Path) -> int:
    _require_clean_tree()
    if not OUTPUT.exists():
        raise RuntimeError("optimized quadrature evidence does not exist")
    artifact = artifact_from_dict(json.loads(OUTPUT.read_text(encoding="utf-8")))
    payload = _thaw(artifact.method_payload)
    if payload.get("optimized") is None or not payload.get("contract_parity"):
        raise RuntimeError("optimized quadrature evidence is not contract-warranted")
    confirmation = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    summary = _optimized_confirmation_summary(payload, confirmation)
    if not all(
        (
            summary["source_revision_distinct"],
            summary["suite_run_ids_distinct"],
            summary["identity_set_exact"],
            summary["measurement_owner_equivalent"],
            summary["controls_match"],
            summary["environment_match"],
            summary["process_isolation"] == "fresh_process_per_record",
            summary["accepted"],
        )
    ):
        raise RuntimeError("optimized quadrature confirmation failed acceptance")
    payload["ratios"]["optimized_confirmation"] = summary
    payload["optimization_decision"] = {
        "status": "optimized_accepted_two_suite",
        "triggered": True,
        "authorization": {
            "trigger": "vmap_128",
            "basis": (
                "The VMAP batch-128 trigger reproduced across smooth, "
                "oscillatory, and expensive-integrand Romberg cases in both "
                "clean baseline suites."
            ),
        },
        "reason": (
            "Two fresh isolated suites improve all three Romberg VMAP-128 "
            "targets without a reproducible scalar or JVP regression."
        ),
    }
    confirmed = replace(artifact, method_payload=payload)
    CONFIRMATION_OUTPUT.write_text(
        json.dumps(confirmation, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_artifact(confirmed)
    print(
        "accepted optimized quadrature evidence and wrote "
        f"{CONFIRMATION_OUTPUT.relative_to(REPO_ROOT)}"
    )
    return 0


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
        payload = _thaw(stored.method_payload)
        if payload.get("optimization_decision", {}).get("status") == (
            "optimized_accepted_two_suite"
        ):
            healthy = healthy and CONFIRMATION_OUTPUT.exists()
            if healthy:
                confirmation = json.loads(
                    CONFIRMATION_OUTPUT.read_text(encoding="utf-8")
                )
                healthy = payload["ratios"]["optimized_confirmation"] == (
                    _optimized_confirmation_summary(payload, confirmation)
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
    mode.add_argument("--confirm-optimized", type=Path, metavar="PATH")
    mode.add_argument(
        "--timing-record",
        nargs=2,
        metavar=("PRECISION", "INDEX"),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.timing_record is not None:
        precision, index = args.timing_record
        print(
            json.dumps(
                _portable_tree(_timing_record(precision, int(index))),
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    if args.check:
        return _check_mode()
    if args.confirm_optimized is not None:
        return _confirm_optimized(args.confirm_optimized)
    if args.timing_only is not None:
        return _timing_only(args.timing_only)
    _require_clean_tree()
    baseline = copy.deepcopy(run_deterministic_suite())
    timing = run_timing_suite()
    baseline["timings"] = timing["records"]
    baseline["timing_run_id"] = timing["run_id"]
    baseline["timing_started_utc"] = timing["started_utc"]
    baseline["timing_environment"] = timing["environment"]
    baseline["timing_process_isolation"] = timing["process_isolation"]
    reviewed = None
    if OUTPUT.exists():
        reviewed = artifact_from_dict(json.loads(OUTPUT.read_text(encoding="utf-8")))
    reviewed_payload = _thaw(reviewed.method_payload) if reviewed is not None else {}
    if (
        reviewed_payload.get("baseline")
        and reviewed_payload.get("optimization_decision", {}).get("triggered") is True
    ):
        artifact = build_optimized_artifact(baseline, reviewed)
    else:
        artifact = build_artifact(baseline)
    _write_artifact(artifact)
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)} and {REPORT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
