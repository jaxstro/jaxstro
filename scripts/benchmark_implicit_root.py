#!/usr/bin/env python3
"""Emit and verify certified implicit-root derivative evidence."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
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
from jaxstro.numerics import (  # noqa: E402
    ImplicitRootAssumptions,
    implicit_bracketed_root,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "docs" / "validation" / "implicit-root-gradients.json"
REPORT = REPO_ROOT / "docs" / "validation" / "implicit-root-gradients.md"
FD_STEP = 1.0e-5
RESIDUAL_LIMIT = 1.0e-12
WIDTH_LIMIT = 1.0e-12
SLOPE_FLOOR = 1.0e-8


class Case(NamedTuple):
    name: str
    residual: Callable[[jax.Array, jax.Array], jax.Array]
    theta: float
    analytic_derivative: float


def _linear(x, theta):
    return x - theta


def _quadratic(x, theta):
    return x * x - theta


def _exponential(x, theta):
    return jnp.exp(x) - theta


CASES = (
    Case("linear", _linear, 2.0, 1.0),
    Case("quadratic", _quadratic, 2.0, 1.0 / (2.0 * math.sqrt(2.0))),
    Case("exponential", _exponential, 2.0, 0.5),
)


def _metric(value: int | float, unit: str) -> dict[str, int | float | str]:
    return {"value": value, "unit": unit}


def _solve(case: Case, theta):
    return implicit_bracketed_root(
        case.residual,
        jnp.asarray(theta, dtype=jnp.float64),
        0.0,
        4.0,
        assumptions=ImplicitRootAssumptions(True, True),
        max_steps=96,
        atol=1.0e-14,
        rtol=1.0e-14,
        safeguard_fraction=0.1,
        derivative_residual_atol=RESIDUAL_LIMIT,
        derivative_width_atol=WIDTH_LIMIT,
        derivative_slope_floor=SLOPE_FLOOR,
    )


def _measure_case(case: Case) -> dict[str, Any]:
    theta = jnp.asarray(case.theta, dtype=jnp.float64)
    result = _solve(case, theta)

    def root(parameter):
        return _solve(case, parameter).root

    ad = jax.grad(root)(theta)
    fd = (root(theta + FD_STEP) - root(theta - FD_STEP)) / (2.0 * FD_STEP)
    width = result.primal.final_bracket.hi - result.primal.final_bracket.lo
    relative_ad_fd_error = jnp.abs(ad - fd) / jnp.maximum(jnp.abs(fd), 1.0e-14)
    relative_ad_analytic_error = jnp.abs(ad - case.analytic_derivative) / jnp.maximum(
        jnp.abs(case.analytic_derivative), 1.0e-14
    )
    return {
        "name": case.name,
        "status": int(result.status),
        "certified": bool(result.certified),
        "root": _metric(float(result.root), "coordinate units"),
        "absolute_residual": _metric(abs(float(result.residual)), "function units"),
        "bracket_width": _metric(float(width), "coordinate units"),
        "slope_magnitude": _metric(
            abs(float(result.slope)), "function units per coordinate unit"
        ),
        "analytic_derivative": _metric(
            case.analytic_derivative, "coordinate units per parameter unit"
        ),
        "ad_derivative": _metric(float(ad), "coordinate units per parameter unit"),
        "fd_derivative": _metric(float(fd), "coordinate units per parameter unit"),
        "relative_ad_fd_error": _metric(float(relative_ad_fd_error), "dimensionless"),
        "relative_ad_analytic_error": _metric(
            float(relative_ad_analytic_error), "dimensionless"
        ),
    }


def run_benchmark() -> dict[str, Any]:
    """Recompute deterministic analytic, AD, and central-FD evidence."""
    git_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    working_tree_dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        ).strip()
    )
    return {
        "schema_version": 2,
        "precision": "float64",
        "provenance_policy": (
            "environment is an emission snapshot; --check gates deterministic "
            "controls, schema, units, and algorithmic metrics, not current revision"
        ),
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
        "controls": {
            "fd_step": _metric(FD_STEP, "parameter units"),
            "residual_limit": _metric(RESIDUAL_LIMIT, "function units"),
            "width_limit": _metric(WIDTH_LIMIT, "coordinate units"),
            "slope_floor": _metric(SLOPE_FLOOR, "function units per coordinate unit"),
        },
        "cases": [_measure_case(case) for case in CASES],
    }


def _validate(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 2 or payload.get("precision") != "float64":
        raise ValueError("implicit-root evidence schema or precision is invalid")
    if {case.get("name") for case in payload.get("cases", [])} != {
        "linear",
        "quadratic",
        "exponential",
    }:
        raise ValueError("implicit-root evidence cases are incomplete")
    expected_controls = {
        "fd_step": _metric(FD_STEP, "parameter units"),
        "residual_limit": _metric(RESIDUAL_LIMIT, "function units"),
        "width_limit": _metric(WIDTH_LIMIT, "coordinate units"),
        "slope_floor": _metric(SLOPE_FLOOR, "function units per coordinate unit"),
    }
    if payload.get("controls") != expected_controls:
        raise ValueError("implicit-root evidence controls or units are invalid")
    expected_environment = {
        "device",
        "git_revision",
        "jax_backend",
        "jax_version",
        "measured_at_utc",
        "platform",
        "python_version",
        "working_tree_dirty",
    }
    environment = payload.get("environment", {})
    if set(environment) != expected_environment:
        raise ValueError("implicit-root environment schema is invalid")
    if not all(
        isinstance(environment[key], str)
        for key in expected_environment - {"working_tree_dirty"}
    ) or not isinstance(environment["working_tree_dirty"], bool):
        raise ValueError("implicit-root environment field types are invalid")
    if payload.get("provenance_policy") != (
        "environment is an emission snapshot; --check gates deterministic "
        "controls, schema, units, and algorithmic metrics, not current revision"
    ):
        raise ValueError("implicit-root provenance policy is missing")
    expected_units = {
        "root": "coordinate units",
        "absolute_residual": "function units",
        "bracket_width": "coordinate units",
        "slope_magnitude": "function units per coordinate unit",
        "analytic_derivative": "coordinate units per parameter unit",
        "ad_derivative": "coordinate units per parameter unit",
        "fd_derivative": "coordinate units per parameter unit",
        "relative_ad_fd_error": "dimensionless",
        "relative_ad_analytic_error": "dimensionless",
    }
    for case in payload["cases"]:
        if not case.get("certified") or case.get("status") != 0:
            raise ValueError(f"implicit-root case is not certified: {case['name']}")
        if case["relative_ad_fd_error"]["value"] > 1.0e-6:
            raise ValueError(f"AD/FD evidence failed: {case['name']}")
        if case["relative_ad_analytic_error"]["value"] > 1.0e-9:
            raise ValueError(f"AD/analytic evidence failed: {case['name']}")
        for metric, unit in expected_units.items():
            if case.get(metric, {}).get("unit") != unit:
                raise ValueError(f"implicit-root metric unit is invalid: {metric}")


def build_artifact(payload: dict[str, Any]) -> EvidenceArtifact:
    """Wrap unchanged derivative measurements in the shared envelope."""
    metrics = []
    comparisons = []
    symbols = {
        "root": "x_star",
        "absolute_residual": "abs(f(x_star))",
        "bracket_width": "Delta_x",
        "slope_magnitude": "abs(df/dx)",
        "analytic_derivative": "dx_star/dtheta|analytic",
        "ad_derivative": "dx_star/dtheta|AD",
        "fd_derivative": "dx_star/dtheta|FD",
        "relative_ad_fd_error": "R_AD,FD",
        "relative_ad_analytic_error": "R_AD,analytic",
    }
    for case in payload["cases"]:
        for name, symbol in symbols.items():
            measured = case[name]
            metrics.append(
                MetricRecord(
                    f"{case['name']}.{name}",
                    symbol,
                    measured["value"],
                    measured["unit"],
                    EvidenceStatus.PASS,
                )
            )
        for name, reference, units in (
            ("absolute_residual", RESIDUAL_LIMIT, "function units"),
            ("bracket_width", WIDTH_LIMIT, "coordinate units"),
            ("relative_ad_fd_error", 1.0e-6, "dimensionless"),
            ("relative_ad_analytic_error", 1.0e-9, "dimensionless"),
        ):
            comparisons.append(
                ComparisonRecord(
                    f"{case['name']}.{name}.gate",
                    f"{case['name']}.{name}",
                    ComparisonRelation.LESS_EQUAL,
                    reference,
                    units,
                    EvidenceStatus.PASS,
                    note="Existing implicit-root validation threshold.",
                )
            )
    environment = payload["environment"]
    return EvidenceArtifact(
        schema_version="1",
        artifact_id="rootfinding.implicit-gradients",
        artifact_version="1",
        package_version=__version__,
        source_revision=environment["git_revision"],
        generation_command="uv run --no-sync python scripts/benchmark_implicit_root.py --emit",
        precision=payload["precision"],
        deterministic_config=tuple(sorted(payload["controls"].items())),
        environment=EnvironmentRecord(
            payload["provenance_policy"],
            tuple((key, str(value)) for key, value in sorted(environment.items())),
        ),
        metrics=tuple(metrics),
        comparisons=tuple(comparisons),
        limitations=(
            "Certification relies on caller assertions of uniqueness and smoothness.",
            "Flat-slope rejection is validated in executable tests, not represented as a certified case.",
        ),
        method_payload=payload,
    )


def algorithmic_metrics_match(stored: dict[str, Any], current: dict[str, Any]) -> bool:
    """Compare deterministic evidence while ignoring environment timestamps."""
    try:
        _validate(stored)
        _validate(current)
    except (KeyError, TypeError, ValueError):
        return False
    stored_cases = {case["name"]: case for case in stored["cases"]}
    current_cases = {case["name"]: case for case in current["cases"]}
    if set(stored_cases) != set(current_cases):
        return False
    if stored.get("controls") != current.get("controls"):
        return False
    metrics = (
        "root",
        "absolute_residual",
        "bracket_width",
        "slope_magnitude",
        "analytic_derivative",
        "ad_derivative",
        "fd_derivative",
        "relative_ad_fd_error",
        "relative_ad_analytic_error",
    )
    for name, current_case in current_cases.items():
        stored_case = stored_cases[name]
        if stored_case.get("status") != current_case["status"]:
            return False
        if stored_case.get("certified") != current_case["certified"]:
            return False
        for metric in metrics:
            if stored_case.get(metric, {}).get("unit") != current_case[metric]["unit"]:
                return False
            stored_value = stored_case.get(metric, {}).get("value")
            current_value = current_case[metric]["value"]
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
        print("implicit-root evidence manifest is missing")
        return 1
    stored_artifact = artifact_from_dict(json.loads(OUTPUT.read_text(encoding="utf-8")))
    stored = stored_artifact.method_payload
    check_artifact(REPORT, stored_artifact)
    if not algorithmic_metrics_match(stored, current):
        print("implicit-root evidence algorithmic metrics are stale")
        return 1
    print("implicit-root evidence healthy: algorithmic metrics match fresh run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
