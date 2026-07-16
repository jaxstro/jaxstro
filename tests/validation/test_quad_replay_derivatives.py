import json
from pathlib import Path

from jaxstro.evidence import EvidenceStatus, artifact_from_dict

ARTIFACT = Path("docs/validation/quad-replay-derivatives.json")


def test_replay_evidence_schema_and_required_case_families():
    artifact = artifact_from_dict(json.loads(ARTIFACT.read_text()))
    payload = artifact.method_payload
    assert artifact.artifact_id == "quad.replay-derivatives"
    assert artifact.schema_version == "1"
    assert (
        payload["claim"] == "replay-differentiable adaptive one-dimensional quadrature"
    )
    required = {
        "smooth_parameter",
        "vector_payload",
        "complex_payload",
        "moving_bounds",
        "reversed_bounds",
        "coincident_bounds",
        "semi_infinite_bound",
        "improper_tail",
        "endpoint_singularity",
        "weighted_density",
        "exhausted_finite",
        "quantity_rescaling",
        "invalid_input",
        "nonfinite_integrand",
    }
    assert required <= {case["family"] for case in payload["cases"]}
    semi_infinite = [
        case for case in payload["cases"] if case["family"] == "semi_infinite_bound"
    ]
    assert {(case["method"], case["variant"]) for case in semi_infinite} == {
        ("adaptive_tanh_sinh", "right"),
        ("adaptive_tanh_sinh", "left"),
        ("romberg_tanh_sinh", "right"),
        ("romberg_tanh_sinh", "left"),
    }
    assert all(
        comparison.status is not EvidenceStatus.FAIL
        for comparison in artifact.comparisons
    )
    for case in payload["cases"]:
        assert {
            "method",
            "family",
            "dtype",
            "primal_value",
            "analytic_value",
            "observed_primal_error",
            "reported_primal_error",
            "replay_ad_derivative",
            "analytic_derivative",
            "frozen_formula_fd",
            "adaptive_rerun_fd",
            "accepted_regions",
            "accepted_level",
            "parameter_unit",
            "integral_unit",
            "derivative_unit",
            "gates",
        } <= case.keys()


def test_replay_evidence_report_uses_progressive_disclosure():
    report = Path(
        "docs/60-validation/numerical/quadrature-replay-derivatives.md"
    ).read_text()
    assert "## What this evidence tests" in report
    assert "## Case map" in report
    assert ":::{dropdown} Complete gate records" in report
    assert ":::{dropdown} Complete machine-readable method payload" in report


def test_quantity_gate_units_are_physical():
    artifact = artifact_from_dict(json.loads(ARTIFACT.read_text()))
    metrics = {metric.identity: metric for metric in artifact.metrics}
    assert (
        metrics[
            "gauss_kronrod.quantity_rescaling.physical_value_absolute_error_cm2"
        ].units
        == "cm^2"
    )
    assert (
        metrics[
            "gauss_kronrod.quantity_rescaling."
            "physical_derivative_absolute_error_cm2_per_cm"
        ].units
        == "cm^2/cm"
    )


def test_every_method_has_scalar_vector_complex_and_stability_ladders():
    artifact = artifact_from_dict(json.loads(ARTIFACT.read_text()))
    payload = artifact.method_payload
    methods = {
        "gauss_kronrod",
        "adaptive_clenshaw_curtis",
        "adaptive_tanh_sinh",
        "romberg",
        "romberg_tanh_sinh",
    }
    for family in ("smooth_parameter", "vector_payload", "complex_payload"):
        assert {
            case["method"] for case in payload["cases"] if case["family"] == family
        } == methods
    for family in ("invalid_input", "nonfinite_integrand"):
        assert {
            case["method"] for case in payload["cases"] if case["family"] == family
        } == methods
    assert {item["method"] for item in payload["stability_ladders"]} == methods
    for ladder in payload["stability_ladders"]:
        assert len(ladder["tolerances"]) >= 3
        assert len(ladder["capacities"]) >= 2
        assert all(rung["passed"] for rung in ladder["tolerances"][-2:])
        assert all(rung["passed"] for rung in ladder["capacities"])
