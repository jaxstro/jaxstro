"""Finite power-law removable-singularity investigation."""

import jax
import jax.numpy as jnp

from jaxstro.jaxconfig import enable_high_precision
from jaxstro.numerics import powerlaw_cdf, powerlaw_logpdf, powerlaw_ppf

from ._common import (
    AuditCheck,
    InvestigationMetric,
    InvestigationResult,
    metric_table,
    validate_result,
)

XMIN = 0.1
XMAX = 10.0
X = 1.5
U = 0.37
FD_STEP = 1.0e-5


def _cdf(alpha):
    return powerlaw_cdf(X, alpha=alpha, xmin=XMIN, xmax=XMAX)


def _analytic_cdf_derivative_at_minus_one() -> float:
    y = jnp.log(X / XMIN)
    width = jnp.log(XMAX / XMIN)
    return float(y * (y - width) / (2.0 * width))


def run() -> InvestigationResult:
    """Audit values and alpha derivatives through the logarithmic limit."""
    enable_high_precision()
    alpha = jnp.asarray(-1.0, dtype=jnp.float64)
    grid = jnp.geomspace(XMIN, XMAX, 4097)
    density = jnp.exp(powerlaw_logpdf(grid, alpha=alpha, xmin=XMIN, xmax=XMAX))
    normalization = jnp.trapezoid(density, grid)
    quantile = powerlaw_ppf(jnp.asarray(U), alpha=alpha, xmin=XMIN, xmax=XMAX)
    roundtrip = powerlaw_cdf(quantile, alpha=alpha, xmin=XMIN, xmax=XMAX)
    ad = jax.grad(_cdf)(alpha)
    fd = (_cdf(alpha + FD_STEP) - _cdf(alpha - FD_STEP)) / (2.0 * FD_STEP)
    analytic = _analytic_cdf_derivative_at_minus_one()
    metrics = (
        InvestigationMetric(
            "powerlaw.normalization_error",
            "abs(integral(p)-1)",
            abs(float(normalization - 1.0)),
            "dimensionless",
        ),
        InvestigationMetric(
            "powerlaw.cdf_ppf_roundtrip_error",
            "abs(CDF(PPF(u))-u)",
            abs(float(roundtrip - U)),
            "dimensionless",
        ),
        InvestigationMetric(
            "powerlaw.alpha_derivative_ad",
            "dCDF/dalpha|AD",
            float(ad),
            "dimensionless per exponent unit",
        ),
        InvestigationMetric(
            "powerlaw.alpha_derivative_analytic",
            "dCDF/dalpha|analytic",
            analytic,
            "dimensionless per exponent unit",
        ),
        InvestigationMetric(
            "powerlaw.ad_fd_error",
            "abs(d_AD-d_FD)",
            abs(float(ad - fd)),
            "dimensionless per exponent unit",
        ),
        InvestigationMetric(
            "powerlaw.ad_analytic_error",
            "abs(d_AD-d_analytic)",
            abs(float(ad - analytic)),
            "dimensionless per exponent unit",
        ),
        InvestigationMetric(
            "powerlaw.cdf_at_lower_support",
            "CDF(xmin)",
            float(powerlaw_cdf(jnp.asarray(XMIN), alpha=alpha, xmin=XMIN, xmax=XMAX)),
            "dimensionless",
        ),
        InvestigationMetric(
            "powerlaw.cdf_at_upper_support",
            "CDF(xmax)",
            float(powerlaw_cdf(jnp.asarray(XMAX), alpha=alpha, xmin=XMIN, xmax=XMAX)),
            "dimensionless",
        ),
    )
    checks = (
        AuditCheck(
            "powerlaw.normalized",
            metrics[0].value <= 1.0e-6,
            "independent trapezoidal quadrature on a logarithmic grid",
        ),
        AuditCheck(
            "powerlaw.roundtrip",
            metrics[1].value <= 1.0e-12,
            "CDF and inverse-CDF composition",
        ),
        AuditCheck(
            "powerlaw.derivative-fd",
            metrics[4].value <= 1.0e-8,
            "central finite difference across alpha = -1",
        ),
        AuditCheck(
            "powerlaw.derivative-limit",
            metrics[5].value <= 1.0e-10,
            "independently derived series coefficient",
        ),
        AuditCheck(
            "powerlaw.support",
            metrics[6].value == 0.0 and metrics[7].value == 1.0,
            "closed-support CDF boundaries",
        ),
    )
    result = InvestigationResult(
        "powerlaw-removable-limit",
        "At alpha = -1 the finite power law should approach a logarithmic CDF with a smooth parameter derivative from both sides.",
        metrics,
        checks,
        "The public finite-power-law kernels preserve normalization, inversion, support, and the independently derived local alpha derivative through the tested removable limit.",
    )
    validate_result(result)
    return result


if __name__ == "__main__":
    print(metric_table(run()), end="")
