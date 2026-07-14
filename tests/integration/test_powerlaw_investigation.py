"""Executable research workflow for the finite power-law removable limit."""

from examples.investigations.powerlaw_removable_limit import run


def test_powerlaw_investigation_audits_the_alpha_minus_one_limit() -> None:
    result = run()
    metrics = {item.identity: item for item in result.metrics}
    assert result.unit_id == "powerlaw-removable-limit"
    assert "alpha = -1" in result.prediction
    assert metrics["powerlaw.normalization_error"].value <= 1.0e-6
    assert metrics["powerlaw.cdf_ppf_roundtrip_error"].value <= 1.0e-12
    assert metrics["powerlaw.ad_fd_error"].value <= 1.0e-8
    assert metrics["powerlaw.ad_analytic_error"].value <= 1.0e-10
    assert metrics["powerlaw.cdf_at_lower_support"].value == 0.0
    assert metrics["powerlaw.cdf_at_upper_support"].value == 1.0
    assert all(check.passed for check in result.audit_checks)
