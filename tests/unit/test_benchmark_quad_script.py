from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

from scripts import benchmark_quad

from jaxstro.evidence import artifact_from_dict, artifact_to_json

ROOT = Path(__file__).resolve().parents[2]


def test_deterministic_suite_contains_both_lanes_and_all_cases() -> None:
    payload = benchmark_quad.run_deterministic_suite()
    assert payload["schema_version"] == 1
    assert {record["precision"] for record in payload["records"]} == {
        "float32",
        "float64",
    }
    assert {record["lane"] for record in payload["records"]} == {
        "family_matched",
        "best_method",
    }
    assert {record["case"] for record in payload["records"]} == {
        case.name for case in benchmark_quad.CASES
    }
    assert {
        (
            record["lane"],
            record["case"],
            record["family"],
            record["pair_variant"],
            record["precision"],
        )
        for record in payload["records"]
    } == benchmark_quad.expected_record_keys()
    assert all(
        record[library]["dtype"] == record["precision"]
        for record in payload["records"]
        for library in ("jaxstro", "quadax")
    )


def test_correctness_precedes_performance_interpretation() -> None:
    payload = benchmark_quad.run_deterministic_suite()
    assert all(
        record["warranted"]["jaxstro"]["passed"] for record in payload["records"]
    )
    quadax_failures = {
        (
            record["case"],
            record["lane"],
            record["family"],
            record["precision"],
        )
        for record in payload["records"]
        if not record["warranted"]["quadax"]["passed"]
    }
    assert quadax_failures == {
        ("oscillatory_cosine", "family_matched", "romberg", "float32"),
        ("nonfinite_band", "family_matched", "gauss_kronrod", "float64"),
        ("nonfinite_band", "best_method", "gauss_kronrod", "float64"),
    }
    derivative_failures = {
        (
            record["case"],
            record["lane"],
            record["family"],
            record["precision"],
        )
        for record in payload["records"]
        if not record["warranted"]["derivatives_passed"]
    }
    structurally_expected = {
        ("breakpoint_kink", "family_matched", "clenshaw_curtis", "float32"),
        ("breakpoint_kink", "best_method", "clenshaw_curtis", "float32"),
        ("oscillatory_cosine", "family_matched", "romberg", "float32"),
        ("breakpoint_kink", "family_matched", "clenshaw_curtis", "float64"),
        ("breakpoint_kink", "best_method", "clenshaw_curtis", "float64"),
    }
    assert structurally_expected <= derivative_failures
    assert all(
        not record["warranted"]["jvp_performance_interpretable"]
        for record in payload["records"]
        if not record["warranted"]["derivatives_passed"]
    )
    assert any(
        record["warranted"]["primal_performance_interpretable"]
        and not record["warranted"]["jvp_performance_interpretable"]
        for record in payload["records"]
        if not record["warranted"]["derivatives_passed"]
    )
    assert all(
        not record["warranted"]["jvp_performance_interpretable"]
        and not record["warranted"]["reverse_performance_interpretable"]
        for record in payload["records"]
        if not record["derivatives"]["available"]
    )
    conservative = next(
        record
        for record in payload["records"]
        if record["precision"] == "float64"
        and record["case"] == "localized_gaussian"
        and record["family"] == "tanh_sinh"
        and record["pair_variant"] == "closest_work"
    )
    assert conservative["jaxstro"]["semantic_status"] == "roundoff_limited"
    assert conservative["warranted"]["jaxstro"]["passed"]
    assert not conservative["warranted"]["jaxstro"]["performance_interpretable"]


def test_timing_record_subprocess_is_fresh_and_complete() -> None:
    record = benchmark_quad._timing_record_subprocess("float32", 0)
    assert record["precision"] == "float32"
    assert record["process_isolation"] == "fresh_process_per_record"
    assert all(
        len(record[mode][library]["warm_seconds"]) == benchmark_quad.TIMING_REPEATS
        for mode in ("scalar", "jvp")
        for library in ("jaxstro", "quadax")
    )
    assert all(
        len(record["vmap"][str(batch)][library]["warm_seconds"])
        == benchmark_quad.TIMING_REPEATS
        for batch in benchmark_quad.VMAP_BATCHES
        for library in ("jaxstro", "quadax")
    )


def test_derivative_gate_rejects_a_wrong_finite_derivative() -> None:
    assert benchmark_quad.derivative_gate(
        measured=1.0,
        truth=1.0,
        atol=1.0e-10,
        rtol=1.0e-10,
    )["passed"]
    assert not benchmark_quad.derivative_gate(
        measured=1.1,
        truth=1.0,
        atol=1.0e-10,
        rtol=1.0e-10,
    )["passed"]


def test_standalone_entrypoint_enables_real_float64() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from scripts import benchmark_quad; "
                "print(benchmark_quad.lane_dtype('float64'))"
            ),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "float64" in completed.stdout


def test_nonfinite_failure_evidence_round_trips_without_json_nan() -> None:
    payload = benchmark_quad.run_deterministic_suite()
    artifact = benchmark_quad.build_artifact(payload)
    rendered = artifact_to_json(artifact)
    assert "NaN" not in rendered
    restored = artifact_from_dict(json.loads(rendered))
    failure = next(
        record
        for record in restored.method_payload["baseline"]["records"]
        if record["case"] == "nonfinite_band"
    )
    assert failure["jaxstro"]["value"] == {
        "finite": False,
        "classification": "nan",
    }


def test_report_exposes_mode_specific_warrants_and_honest_choice_label() -> None:
    artifact = benchmark_quad.build_artifact(benchmark_quad.run_deterministic_suite())
    report = benchmark_quad.render_report(artifact)
    assert "warrant primal timing interpretation" in report
    assert "warrant JVP timing" in report
    assert "Predeclared practical choice" in report
    assert "Independent practical choice" not in report


def test_optimized_merge_preserves_baseline_byte_for_byte() -> None:
    baseline = benchmark_quad.run_deterministic_suite()
    baseline_bytes = json.dumps(baseline, sort_keys=True, allow_nan=False)
    optimized = copy.deepcopy(baseline)
    optimized["source_revision"] = "optimized-revision"
    payload = benchmark_quad.merge_optimized(
        baseline=baseline,
        optimized=optimized,
        ratios={"warm": 0.8},
        contract_parity=True,
    )
    assert (
        json.dumps(payload["baseline"], sort_keys=True, allow_nan=False)
        == baseline_bytes
    )
    assert payload["optimized"]["source_revision"] == "optimized-revision"
    assert payload["ratios"] == {"warm": 0.8}
    assert payload["contract_parity"] is True


def test_freshness_ignores_timings_but_rejects_accuracy_drift() -> None:
    current = benchmark_quad.run_deterministic_suite()
    stored = copy.deepcopy(current)
    stored["timings"] = [{"median_warm_seconds": 99.0}]
    assert benchmark_quad.algorithmic_metrics_match(stored, current)
    stored["records"][0]["jaxstro"]["absolute_error"] += 1.0e-4
    assert not benchmark_quad.algorithmic_metrics_match(stored, current)


def test_check_mode_requires_the_emitted_artifact() -> None:
    completed = subprocess.run(
        [sys.executable, ROOT / "scripts" / "benchmark_quad.py", "--check"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode in {0, 1}
    assert "quadrature performance evidence" in completed.stdout
