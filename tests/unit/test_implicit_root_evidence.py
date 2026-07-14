"""Schema and scientific bounds for implicit-root evidence."""

import json
from pathlib import Path

from scripts import benchmark_implicit_root

EVIDENCE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "validation"
    / "implicit-root-gradients.json"
)
REPORT = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "60-validation"
    / "numerical"
    / "implicit-root-gradients.md"
)


def _payload() -> dict:
    envelope = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert envelope["schema_version"] == "1"
    assert envelope["artifact_id"] == "rootfinding.implicit-gradients"
    assert envelope["metrics"]
    assert envelope["comparisons"]
    assert REPORT.is_file()
    return envelope["method_payload"]


def test_implicit_envelope_distinguishes_observations_from_gates() -> None:
    envelope = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert {metric["status"] for metric in envelope["metrics"]} == {"info"}
    comparison_ids = {item["identity"] for item in envelope["comparisons"]}
    for case in ("linear", "quadratic", "exponential"):
        assert f"{case}.slope_magnitude.gate" in comparison_ids
        assert f"{case}.certificate.gate" in comparison_ids


def test_implicit_root_evidence_has_units_and_passes_claim_bounds() -> None:
    payload = _payload()

    assert payload["schema_version"] == 2
    assert payload["precision"] == "float64"
    assert {case["name"] for case in payload["cases"]} == {
        "linear",
        "quadratic",
        "exponential",
    }
    for case in payload["cases"]:
        assert case["status"] == 0
        assert case["certified"] is True
        for metric, evidence in case.items():
            if metric not in {"name", "status", "certified"}:
                assert set(evidence) == {"value", "unit"}
        assert case["absolute_residual"]["value"] <= 1.0e-12
        assert case["bracket_width"]["value"] <= 1.0e-12
        assert case["relative_ad_fd_error"]["value"] <= 1.0e-6


def test_implicit_root_evidence_matches_fresh_algorithmic_metrics() -> None:
    stored = _payload()
    current = benchmark_implicit_root.run_benchmark()

    assert benchmark_implicit_root.algorithmic_metrics_match(stored, current)


def test_implicit_root_evidence_records_environment_fields() -> None:
    payload = _payload()

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
    assert payload["provenance_policy"] == (
        "environment is an emission snapshot; --check gates deterministic "
        "controls, schema, units, and algorithmic metrics, not current revision"
    )
    assert payload["controls"] == {
        "fd_step": {"value": 1.0e-5, "unit": "parameter units"},
        "residual_limit": {"value": 1.0e-12, "unit": "function units"},
        "slope_floor": {
            "value": 1.0e-8,
            "unit": "function units per coordinate unit",
        },
        "width_limit": {"value": 1.0e-12, "unit": "coordinate units"},
    }


def test_implicit_root_evidence_matcher_rejects_control_and_unit_drift() -> None:
    stored = _payload()
    current = benchmark_implicit_root.run_benchmark()
    stored["controls"]["fd_step"]["value"] = 999.0
    assert not benchmark_implicit_root.algorithmic_metrics_match(stored, current)

    stored = _payload()
    stored["cases"][0]["root"]["unit"] = "wrong units"
    assert not benchmark_implicit_root.algorithmic_metrics_match(stored, current)
