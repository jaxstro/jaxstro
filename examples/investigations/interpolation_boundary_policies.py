"""Interpolation interior-derivative and boundary-policy investigation."""

import jax
import jax.numpy as jnp

from jaxstro.jaxconfig import enable_high_precision
from jaxstro.numerics import regular_grid_interp
from jaxstro.numerics.interpolation import interp1d

from ._common import (
    AuditCheck,
    InvestigationMetric,
    InvestigationResult,
    calibrated_claim,
    investigation_report,
    validate_result,
)


def run() -> InvestigationResult:
    """Audit affine recovery separately from boundary-policy behavior."""
    enable_high_precision()
    x = jnp.asarray([0.0, 1.0, 2.0])
    y = 2.0 * x + 1.0
    query = jnp.asarray(0.7)
    value = interp1d(x, y, query)
    derivative = jax.grad(lambda q: interp1d(x, y, q))(query)

    axis_x = jnp.asarray([0.0, 1.0, 2.0])
    axis_y = jnp.asarray([-1.0, 0.0, 1.0])
    xx, yy = jnp.meshgrid(axis_x, axis_y, indexing="ij")
    values = 2.0 * xx - 3.0 * yy + 1.0
    xi = jnp.asarray([0.4, 0.2])
    grid_value = regular_grid_interp((axis_x, axis_y), values, xi)
    analytic_grid = 2.0 * xi[0] - 3.0 * xi[1] + 1.0
    outside = jnp.asarray([3.0, 0.2])
    clamped = regular_grid_interp((axis_x, axis_y), values, outside, boundary="clamp")
    expected_clamped = 2.0 * axis_x[-1] - 3.0 * outside[1] + 1.0
    filled = regular_grid_interp(
        (axis_x, axis_y), values, outside, boundary="fill", fill_value=-99.0
    )
    reject_raised = 0
    try:
        regular_grid_interp((axis_x, axis_y), values, outside, boundary="reject")
    except ValueError:
        reject_raised = 1

    metrics = (
        InvestigationMetric(
            "interpolation.interior_value_error",
            "abs(I(x)-f(x))",
            abs(float(value - (2.0 * query + 1.0))),
            "function units",
        ),
        InvestigationMetric(
            "interpolation.interior_derivative_error",
            "abs(dI/dx-2)",
            abs(float(derivative - 2.0)),
            "function units per coordinate unit",
        ),
        InvestigationMetric(
            "interpolation.regular_grid_affine_error",
            "abs(I(x,y)-f(x,y))",
            abs(float(grid_value - analytic_grid)),
            "function units",
        ),
        InvestigationMetric(
            "interpolation.clamp_value",
            "I_clamp",
            float(clamped),
            "function units",
        ),
        InvestigationMetric(
            "interpolation.clamp_error",
            "abs(I_clamp-f(x_clipped))",
            abs(float(clamped - expected_clamped)),
            "function units",
        ),
        InvestigationMetric(
            "interpolation.fill_value",
            "I_fill",
            float(filled),
            "function units",
        ),
        InvestigationMetric(
            "interpolation.reject_raised",
            "I_reject",
            reject_raised,
            "events",
        ),
    )
    checks = (
        AuditCheck(
            "interpolation.affine-1d",
            metrics[0].value <= 1.0e-12,
            "analytic affine identity",
        ),
        AuditCheck(
            "interpolation.derivative-interior",
            metrics[1].value <= 1.0e-12,
            "analytic interior derivative",
        ),
        AuditCheck(
            "interpolation.affine-grid",
            metrics[2].value <= 1.0e-12,
            "analytic bilinear affine identity",
        ),
        AuditCheck(
            "interpolation.boundary-policies",
            metrics[4].value <= 1.0e-12
            and metrics[5].value == -99.0
            and reject_raised == 1,
            "explicit clamp, fill, and reject policy outcomes",
        ),
    )
    claim = calibrated_claim(
        checks,
        "The tested affine values and derivatives are exact on cell interiors, not at knots or policy boundaries; outside-grid behavior follows the explicitly selected policy.",
    )
    result = InvestigationResult(
        "interpolation-boundary-policies",
        "Affine data should be exact on branch-stable interiors; clamp, fill, and reject should remain visibly different outside the grid.",
        metrics,
        checks,
        claim,
    )
    validate_result(result)
    return result


if __name__ == "__main__":
    output = run()
    print(investigation_report(output), end="")
    raise SystemExit(not all(check.passed for check in output.audit_checks))
