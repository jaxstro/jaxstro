# tests/test_provenance.py
"""Tests for numerical-method provenance and trust reports."""

import json

from jaxstro.testing import (
    EvidenceAnchor,
    MethodEvidence,
    NumericalTrustReport,
    default_numerics_trust_report,
    trust_report_to_dict,
    trust_report_to_json,
    trust_report_to_markdown,
)


def test_trust_report_json_is_deterministic_and_sorted():
    report = NumericalTrustReport(
        title="Example report",
        methods=(
            MethodEvidence(
                method="zeta",
                summary="Later method.",
                status="validated",
                anchors=(
                    EvidenceAnchor(
                        property="Z property",
                        measured="Synthetic value",
                        tolerance="exact",
                        anchors=("tests/unit/test_z.py",),
                    ),
                ),
            ),
            MethodEvidence(
                method="alpha",
                summary="Earlier method.",
                status="validated",
                anchors=(
                    EvidenceAnchor(
                        property="A property",
                        measured="Synthetic value",
                        tolerance="exact",
                        anchors=("tests/unit/test_a.py",),
                    ),
                ),
            ),
        ),
    )
    first = trust_report_to_json(report)
    second = trust_report_to_json(report)
    assert first == second
    payload = json.loads(first)
    assert [method["method"] for method in payload["methods"]] == ["alpha", "zeta"]


def test_trust_report_markdown_contains_anchor_table():
    report = NumericalTrustReport(
        title="Example report",
        methods=(
            MethodEvidence(
                method="rootfinding",
                summary="Fixed-iteration solvers.",
                status="validated",
                anchors=(
                    EvidenceAnchor(
                        property="Newton gradients match FD",
                        measured="FD-vs-AD",
                        tolerance="1e-5 relative",
                        anchors=("tests/validation/test_grad_checks.py",),
                    ),
                ),
                known_limits=("Bisection is value-first for function parameters.",),
            ),
        ),
    )
    markdown = trust_report_to_markdown(report)
    assert markdown.startswith("# Example report")
    assert "## rootfinding" in markdown
    assert "| Property | Measured | Tolerance | Anchors |" in markdown
    assert "Bisection is value-first" in markdown


def test_default_numerics_trust_report_covers_current_roadmap_batches():
    report = default_numerics_trust_report()
    data = trust_report_to_dict(report)
    methods = {method["method"] for method in data["methods"]}
    assert {
        "splines",
        "interpolation",
        "regular_grid",
        "rootfinding",
        "linear_algebra",
        "special",
        "grids_sampling",
    }.issubset(methods)
    assert all(method["anchors"] for method in data["methods"])
