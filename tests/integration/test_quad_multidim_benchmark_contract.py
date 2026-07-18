from __future__ import annotations

import copy
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
RUNNER_PATH = ROOT / "scripts/benchmark_quad_multidim.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "benchmark_quad_multidim",
        RUNNER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_record(case_id: str, *, warm: float, memory: float, scaling: float):
    return {
        "case_id": case_id,
        "timings": {"scalar": {"warm_median_seconds": warm}},
        "memory_family_ratio": memory,
        "repeated_scaling_excess": scaling,
    }


def _synthetic_comparator(case_id: str, elapsed: float):
    return {
        "case_id": case_id,
        "library": "scipy",
        "label": "exact",
        "elapsed_seconds": elapsed,
        "controls": {"timing_relation": "matched"},
    }


def test_trigger_policy_has_both_no_change_and_addendum_paths():
    runner = _load_runner()
    no_change = runner._trigger_assessment(
        [_synthetic_record("case", warm=1.0, memory=1.0, scaling=0.0)],
        [_synthetic_comparator("case", elapsed=1.0)],
    )
    assert not no_change["trigger_fired"]
    assert no_change["decision"] == "no_runtime_change"

    fired = runner._trigger_assessment(
        [_synthetic_record("case", warm=1.5, memory=1.0, scaling=0.0)],
        [_synthetic_comparator("case", elapsed=1.0)],
    )
    assert fired["trigger_fired"]
    assert fired["decision"] == "optimization_addendum_required"


def test_payload_digest_rejects_mutation():
    runner = _load_runner()
    artifact = runner._with_digest({"artifact_id": "synthetic", "records": []})
    runner._validate_digest(artifact)
    mutated = copy.deepcopy(artifact)
    mutated["records"].append({"changed": True})
    with pytest.raises(ValueError, match="payload digest"):
        runner._validate_digest(mutated)


def test_baseline_manifest_is_frozen_before_measurement():
    runner = _load_runner()
    assert runner.DIMENSIONS == (2, 4, 8, 16)
    assert runner.VMAP_BATCHES == (16, 128)
    assert runner.WARM_REPEATS >= 2
    assert runner.TRIGGERS == {
        "warm_runtime_ratio": 1.50,
        "compiler_cost_ratio": 2.00,
        "memory_family_ratio": 2.00,
        "repeated_scaling_excess": 0.25,
    }


def test_evidence_emission_rejects_dirty_worktree():
    sentinel = ROOT / ".quad-benchmark-dirty-sentinel"
    sentinel.write_text("test-owned dirty state\n")
    try:
        completed = subprocess.run(
            (
                sys.executable,
                str(RUNNER_PATH),
                "--suite",
                "baseline",
                "--emit",
            ),
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        sentinel.unlink()
    assert completed.returncode != 0
    assert "evidence emission requires a clean worktree" in completed.stderr


def test_allow_dirty_is_for_exploration_only():
    completed = subprocess.run(
        (
            sys.executable,
            str(RUNNER_PATH),
            "--suite",
            "baseline",
            "--emit",
            "--allow-dirty",
        ),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "never valid for evidence emission" in completed.stderr
