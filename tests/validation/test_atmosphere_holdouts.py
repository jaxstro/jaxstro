"""Schema and evidence contracts for measured atmosphere holdouts."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "docs" / "validation" / "atmosphere-interpolation.json"


def test_holdout_manifest_records_measured_policy_or_explicit_failure() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["primary_metric"] == "p95_relative_error"
    assert payload["selection_rule"] == "primary win plus no secondary regression"
    assert payload["products"]
    assert set(payload["representative_products"]) == set(payload["products"])
    for product_id, record in payload["products"].items():
        assert product_id == record["product_id"]
        assert record["status"] in {"accepted", "POLICY_NOT_VALIDATED"}
        assert set(record["metrics"]) == {"linear", "positive_log"}
        for metrics in record["metrics"].values():
            assert set(metrics) == {
                "median_relative_error",
                "p95_relative_error",
                "maximum_log_flux_error",
                "integrated_flux_relative_error",
            }
            assert all(value >= 0.0 for value in metrics.values())
        assert record["holdout_method"] == "leave-one-teff-slice-out"
        assert record["positive_support"]["bins"] >= 2
        assert (
            record["positive_support"]["wavelength_min_nm"]
            < record["positive_support"]["wavelength_max_nm"]
        )
