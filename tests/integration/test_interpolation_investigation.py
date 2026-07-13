"""Executable curriculum contract for interpolation boundary policies."""

from examples.investigations.interpolation_boundary_policies import run


def test_interpolation_investigation_separates_interior_and_boundary_claims() -> None:
    result = run()
    metrics = {item.identity: item for item in result.metrics}
    assert result.unit_id == "interpolation-boundary-policies"
    assert metrics["interpolation.interior_value_error"].value <= 1.0e-12
    assert metrics["interpolation.interior_derivative_error"].value <= 1.0e-12
    assert metrics["interpolation.regular_grid_affine_error"].value <= 1.0e-12
    assert metrics["interpolation.reject_raised"].value == 1
    assert all(check.passed for check in result.audit_checks)
    assert "not at knots or policy boundaries" in result.warranted_claim
