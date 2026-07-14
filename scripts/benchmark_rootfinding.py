#!/usr/bin/env python3
"""Measure bounded safeguarded-root and bisection evaluation costs."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

from jaxstro.jaxconfig import enable_high_precision

enable_high_precision()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from jaxstro import __version__  # noqa: E402
from jaxstro.evidence import (  # noqa: E402
    ComparisonRecord,
    ComparisonRelation,
    EnvironmentRecord,
    EvidenceArtifact,
    EvidenceStatus,
    MetricRecord,
    artifact_from_dict,
    check_artifact,
    emit_artifact,
)
from jaxstro.numerics.rootfinding import (  # noqa: E402
    bisect,
    safeguarded_bracketed_root,
)

WARM_REPEATS = 21
BISECTION_STEPS = 48
HYBRID_STEPS = 96
ATOL = 1.0e-12
RTOL = 1.0e-12
SAFEGUARD_FRACTION = 0.1
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "docs" / "validation" / "rootfinding-performance.json"
REPORT = (
    REPO_ROOT / "docs" / "60-validation" / "numerical" / "rootfinding-performance.md"
)


class Case(NamedTuple):
    name: str
    f: Callable[[jax.Array], jax.Array]
    lo: float
    hi: float


CASES = (
    Case("linear", lambda x: x - 2.0, 0.0, 4.0),
    Case("quadratic", lambda x: x**2 - 2.0, 0.0, 2.0),
    Case("flat_slope", lambda x: (x - 1.0) ** 3, 0.0, 3.0),
    Case(
        "monotone_kink",
        lambda x: jnp.where(x < 0.3, 2.0 * (x - 0.3), 0.5 * (x - 0.3)),
        0.0,
        1.0,
    ),
    Case("oscillatory_fixed_point_residual", lambda h: (0.7 - h) - h, 0.0, 1.0),
)


def _warm_seconds(call: Callable[[], Any], ready: Callable[[Any], Any]) -> float:
    value = call()
    jax.block_until_ready(ready(value))
    samples = []
    for _ in range(WARM_REPEATS):
        start = time.perf_counter()
        value = call()
        jax.block_until_ready(ready(value))
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _metric(value: int | float, unit: str) -> dict[str, int | float | str]:
    return {"value": value, "unit": unit}


def _relative_residual(case: Case, residual: float) -> float:
    scale = max(abs(float(case.f(case.lo))), abs(float(case.f(case.hi))))
    return abs(residual) / scale


def _measure_case(case: Case) -> dict[str, Any]:
    hybrid_call = jax.jit(
        lambda: safeguarded_bracketed_root(
            case.f,
            case.lo,
            case.hi,
            max_steps=HYBRID_STEPS,
            atol=ATOL,
            rtol=RTOL,
            safeguard_fraction=SAFEGUARD_FRACTION,
        )
    )
    bisection_call = jax.jit(
        lambda: bisect(case.f, case.lo, case.hi, max_steps=BISECTION_STEPS)
    )
    hybrid_wall = _warm_seconds(hybrid_call, lambda result: result.root)
    bisection_wall = _warm_seconds(bisection_call, lambda root: root)
    hybrid = hybrid_call()
    bisection_root = bisection_call()
    jax.block_until_ready((hybrid.root, bisection_root))
    if not bool(hybrid.converged):
        raise RuntimeError(f"hybrid benchmark case did not converge: {case.name}")

    hybrid_residual = abs(float(hybrid.residual))
    bisection_residual = abs(float(case.f(bisection_root)))
    hybrid_iterations = int(jnp.sum(hybrid.trace.executed))
    return {
        "name": case.name,
        "bracket": [case.lo, case.hi],
        "methods": {
            "bisection": {
                "status": "fixed_steps",
                "converged": None,
                "function_evaluations": _metric(BISECTION_STEPS + 2, "evaluations"),
                "executed_iterations": _metric(BISECTION_STEPS, "iterations"),
                "final_absolute_residual": _metric(
                    bisection_residual, "function units"
                ),
                "final_relative_residual": _metric(
                    _relative_residual(case, bisection_residual), "dimensionless"
                ),
                "warm_wall": _metric(bisection_wall, "s"),
            },
            "safeguarded_hybrid": {
                "status": int(hybrid.status),
                "converged": bool(hybrid.converged),
                "function_evaluations": _metric(
                    int(hybrid.n_evaluations), "evaluations"
                ),
                "executed_iterations": _metric(hybrid_iterations, "iterations"),
                "final_absolute_residual": _metric(hybrid_residual, "function units"),
                "final_relative_residual": _metric(
                    _relative_residual(case, hybrid_residual), "dimensionless"
                ),
                "warm_wall": _metric(hybrid_wall, "s"),
            },
        },
    }


def run_benchmark() -> dict[str, Any]:
    """Return deterministic algorithm metrics plus bounded warm timings."""
    git_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    working_tree_dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        ).strip()
    )
    return {
        "schema_version": 1,
        "precision": "float64",
        "warm_repeats": WARM_REPEATS,
        "environment": {
            "device": str(jax.devices()[0]),
            "git_revision": git_revision,
            "jax_backend": jax.default_backend(),
            "jax_version": jax.__version__,
            "measured_at_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "working_tree_dirty": working_tree_dirty,
        },
        "relative_residual_definition": ("abs(f(root)) / max(abs(f(lo)), abs(f(hi)))"),
        "controls": {
            "bisection_steps": BISECTION_STEPS,
            "hybrid_max_steps": HYBRID_STEPS,
            "atol": ATOL,
            "rtol": RTOL,
            "safeguard_fraction": SAFEGUARD_FRACTION,
            "matched_coordinate_tolerance": _metric(
                ATOL + RTOL * 4.0, "coordinate units"
            ),
        },
        "cases": [_measure_case(case) for case in CASES],
    }


def _validate(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1 or payload.get("precision") != "float64":
        raise ValueError("rootfinding benchmark schema or precision is invalid")
    if not payload.get("cases") or not payload.get("environment"):
        raise ValueError("rootfinding benchmark evidence is incomplete")
    substantial_reductions = 0
    for case in payload["cases"]:
        bisection = case["methods"]["bisection"]
        hybrid = case["methods"]["safeguarded_hybrid"]
        bisection_evals = bisection["function_evaluations"]["value"]
        hybrid_evals = hybrid["function_evaluations"]["value"]
        if not hybrid["converged"] or hybrid_evals > bisection_evals:
            raise ValueError(f"forward efficiency gate failed: {case['name']}")
        substantial_reductions += hybrid_evals <= 0.75 * bisection_evals
    if substantial_reductions < 3:
        raise ValueError("forward efficiency gate requires three 25% reductions")


def build_artifact(payload: dict[str, Any]) -> EvidenceArtifact:
    """Wrap unchanged measurements in the shared evidence envelope."""
    metrics = []
    comparisons = []
    symbols = {
        "function_evaluations": "N_eval",
        "executed_iterations": "N_iter",
        "final_absolute_residual": "abs(f(x_star))",
        "final_relative_residual": "R_root",
        "warm_wall": "t_wall,warm",
    }
    for case in payload["cases"]:
        methods = case["methods"]
        for method_name, method in methods.items():
            for metric_name, symbol in symbols.items():
                measured = method[metric_name]
                identity = f"{case['name']}.{method_name}.{metric_name}"
                metrics.append(
                    MetricRecord(
                        identity,
                        symbol,
                        measured["value"],
                        measured["unit"],
                        EvidenceStatus.INFO,
                    )
                )
        hybrid_id = f"{case['name']}.safeguarded_hybrid.function_evaluations"
        comparisons.append(
            ComparisonRecord(
                f"{case['name']}.hybrid-no-more-evaluations",
                hybrid_id,
                ComparisonRelation.LESS_EQUAL,
                methods["bisection"]["function_evaluations"]["value"],
                "evaluations",
                EvidenceStatus.PASS,
                note="Hybrid evaluation count must not exceed fixed-step bisection.",
            )
        )
    environment = payload["environment"]
    return EvidenceArtifact(
        schema_version="1",
        artifact_id="rootfinding.performance",
        artifact_version="1",
        package_version=__version__,
        source_revision=environment["git_revision"],
        generation_command="uv run --no-sync python scripts/benchmark_rootfinding.py --emit",
        precision=payload["precision"],
        deterministic_config=tuple(sorted(payload["controls"].items())),
        environment=EnvironmentRecord(
            "Recorded execution environment; wall metrics are informational.",
            tuple((key, str(value)) for key, value in sorted(environment.items())),
        ),
        metrics=tuple(metrics),
        comparisons=tuple(comparisons),
        limitations=("Warm wall time is hardware- and load-dependent.",),
        method_payload=payload,
    )


def _algorithmic_metrics_match(stored: dict[str, Any], current: dict[str, Any]) -> bool:
    stored_cases = {case["name"]: case for case in stored["cases"]}
    current_cases = {case["name"]: case for case in current["cases"]}
    if set(stored_cases) != set(current_cases):
        return False
    for name, current_case in current_cases.items():
        stored_methods = stored_cases[name]["methods"]
        for method, current_metrics in current_case["methods"].items():
            stored_metrics = stored_methods.get(method, {})
            for metric in (
                "function_evaluations",
                "executed_iterations",
                "final_absolute_residual",
                "final_relative_residual",
            ):
                stored_value = stored_metrics.get(metric, {}).get("value")
                current_value = current_metrics[metric]["value"]
                if not isinstance(stored_value, (int, float)) or not math.isclose(
                    stored_value,
                    current_value,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-15,
                ):
                    return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()

    current = run_benchmark()
    _validate(current)
    current_artifact = build_artifact(current)
    if args.emit:
        emit_artifact(OUTPUT, current_artifact)
        emit_artifact(REPORT, current_artifact)
        print(f"wrote {OUTPUT.relative_to(REPO_ROOT)}")
        return 0

    if not OUTPUT.exists():
        print("rootfinding benchmark manifest is missing")
        return 1
    stored_artifact = artifact_from_dict(json.loads(OUTPUT.read_text(encoding="utf-8")))
    stored = stored_artifact.method_payload
    _validate(stored)
    check_artifact(REPORT, stored_artifact)
    if not _algorithmic_metrics_match(stored, current):
        print("rootfinding benchmark algorithmic metrics are stale")
        return 1
    print("rootfinding benchmark healthy: algorithmic metrics match fresh run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
