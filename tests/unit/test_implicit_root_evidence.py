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


def test_implicit_root_evidence_has_units_and_passes_claim_bounds() -> None:
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))

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
    stored = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    current = benchmark_implicit_root.run_benchmark()

    assert benchmark_implicit_root.algorithmic_metrics_match(stored, current)


def test_implicit_root_evidence_records_environment_fields() -> None:
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))

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
