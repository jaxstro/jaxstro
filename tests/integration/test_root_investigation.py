"""Executable curriculum contract for root values and sensitivities."""

import math

from examples.investigations.root_values_and_sensitivities import run


def test_root_investigation_predicts_computes_and_audits() -> None:
    result = run()
    metrics = {item.identity: item for item in result.metrics}
    assert result.unit_id == "root-values-and-sensitivities"
    assert "value-first" in result.prediction
    assert "implicit" in result.prediction
    assert metrics["root.absolute_residual"].value <= 1.0e-12
    assert metrics["root.bracket_width"].value <= 1.0e-12
    assert metrics["root.derivative_absolute_error"].value <= 1.0e-9
    assert math.isfinite(metrics["root.implicit_derivative"].value)
    assert all(check.passed for check in result.audit_checks)
    analytic_check = next(
        check
        for check in result.audit_checks
        if check.identity == "root.positive-branch-unique-smooth"
    )
    assert "strictly increasing" in analytic_check.evidence
    assert "does not make the value-first branch history an IFT derivative" in (
        result.warranted_claim
    )
