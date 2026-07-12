"""Schema and scientific bounds for implicit-root evidence."""

import json
from pathlib import Path

EVIDENCE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "validation"
    / "implicit-root-gradients.json"
)


def test_implicit_root_evidence_has_units_and_passes_claim_bounds() -> None:
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["precision"] == "float64"
    assert {case["name"] for case in payload["cases"]} == {
        "linear",
        "quadratic",
        "exponential",
    }
    for case in payload["cases"]:
        for metric, evidence in case.items():
            if metric != "name":
                assert set(evidence) == {"value", "unit"}
        assert case["absolute_residual"]["value"] <= 1.0e-12
        assert case["bracket_width"]["value"] <= 1.0e-12
        assert case["relative_ad_fd_error"]["value"] <= 1.0e-6
