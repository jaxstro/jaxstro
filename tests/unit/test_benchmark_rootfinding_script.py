"""Schema contract for safeguarded-root evaluation evidence."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "docs" / "validation" / "rootfinding-performance.json"
REPORT = (
    REPO_ROOT / "docs" / "60-validation" / "numerical" / "rootfinding-performance.md"
)


def test_rootfinding_benchmark_manifest_has_required_metrics_and_units() -> None:
    envelope = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert envelope["schema_version"] == "1"
    assert envelope["artifact_id"] == "rootfinding.performance"
    assert envelope["metrics"]
    assert envelope["comparisons"]
    assert {metric["status"] for metric in envelope["metrics"]} == {"info"}
    assert REPORT.is_file()
    payload = envelope["method_payload"]

    assert payload["schema_version"] == 1
    assert payload["precision"] == "float64"
    assert payload["warm_repeats"] >= 7
    assert set(payload["environment"]) == {
        "device",
        "git_revision",
        "jax_backend",
        "jax_version",
        "measured_at_utc",
        "platform",
        "python_version",
        "working_tree_dirty",
    }
    assert payload["relative_residual_definition"] == (
        "abs(f(root)) / max(abs(f(lo)), abs(f(hi)))"
    )
    assert (
        payload["controls"]["matched_coordinate_tolerance"]["unit"]
        == "coordinate units"
    )
    assert payload["cases"]
    for case in payload["cases"]:
        assert set(case["methods"]) == {"bisection", "safeguarded_hybrid"}
        assert case["methods"]["bisection"]["status"] == "fixed_steps"
        assert case["methods"]["bisection"]["converged"] is None
        assert isinstance(case["methods"]["safeguarded_hybrid"]["converged"], bool)
        for metrics in case["methods"].values():
            assert isinstance(metrics["status"], (int, str))
            assert metrics["function_evaluations"]["unit"] == "evaluations"
            assert metrics["executed_iterations"]["unit"] == "iterations"
            assert metrics["final_absolute_residual"]["unit"] == "function units"
            assert metrics["final_relative_residual"]["unit"] == "dimensionless"
            assert metrics["warm_wall"]["unit"] == "s"
            assert metrics["function_evaluations"]["value"] >= 2
            assert metrics["executed_iterations"]["value"] >= 0
            assert metrics["final_absolute_residual"]["value"] >= 0.0
            assert metrics["final_relative_residual"]["value"] >= 0.0
            assert metrics["warm_wall"]["value"] >= 0.0


def test_rootfinding_benchmark_check_recomputes_algorithmic_metrics() -> None:
    completed = subprocess.run(
        [sys.executable, REPO_ROOT / "scripts" / "benchmark_rootfinding.py", "--check"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "rootfinding benchmark healthy" in completed.stdout
