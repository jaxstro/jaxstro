"""Schema contract for the bounded spectra performance artifact."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "docs" / "validation" / "spectra-performance.json"
REPORT = REPO_ROOT / "docs" / "60-validation" / "data" / "spectra-performance.md"


def test_spectra_benchmark_manifest_has_separate_bounded_timings() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "1"
    assert payload["artifact_id"] == "spectra.performance"
    assert payload["metrics"]
    assert payload["comparisons"] == []
    assert REPORT.is_file()
    payload = payload["method_payload"]

    assert payload["schema_version"] == 1
    assert payload["case"]["product_id"] == "newera-v3-lowres"
    assert payload["case"]["spectral_bins"] == 64
    assert payload["case"]["batch_size"] == 8
    assert payload["memory_scope"] == "Python host allocations measured by tracemalloc"
    assert set(payload["timings_seconds"]) == {
        "host_preparation",
        "first_jit_evaluation",
        "cached_evaluation_median",
        "batched_evaluation",
    }
    assert all(value >= 0.0 for value in payload["timings_seconds"].values())
    assert payload["host_peak_memory_bytes"] > 0
    assert payload["output_shape"] == [64]
    assert payload["batched_output_shape"] == [8, 64]
