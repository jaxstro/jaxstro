"""Contracts for smooth, boundary-sensitive, and validation-only exemplars."""

from jaxstro.contracts import ADSemantics, get_callable_contract


def test_powerlaw_names_removable_limit_and_support() -> None:
    record = get_callable_contract("jaxstro.numerics.powerlaw_cdf")
    assert record.ad_semantics is ADSemantics.SMOOTH_PATHWISE
    assert "alpha=-1" in record.domain.replace(" ", "")
    assert any("support" in item.summary.lower() for item in record.boundaries)


def test_regular_grid_keeps_boundary_policies_distinct() -> None:
    record = get_callable_contract("jaxstro.numerics.regular_grid_interp")
    assert any("clamp" in item.summary for item in record.boundaries)
    assert any("reject" in item.summary for item in record.boundaries)


def test_gradient_audit_is_validation_only() -> None:
    record = get_callable_contract("jaxstro.testing.compare_gradients")
    assert record.ad_semantics is ADSemantics.VALIDATION_ONLY
