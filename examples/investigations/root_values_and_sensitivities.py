"""Root value versus certified implicit-sensitivity investigation."""

import jax
import jax.numpy as jnp

from jaxstro.jaxconfig import enable_high_precision
from jaxstro.numerics import (
    ImplicitRootAssumptions,
    implicit_bracketed_root,
    safeguarded_bracketed_root,
)

from ._common import (
    AuditCheck,
    InvestigationMetric,
    InvestigationResult,
    calibrated_claim,
    investigation_report,
    validate_result,
)


def _residual(x, theta):
    return x * x - theta


def _implicit(theta):
    return implicit_bracketed_root(
        _residual,
        theta,
        0.0,
        2.0,
        assumptions=ImplicitRootAssumptions(True, True),
        max_steps=96,
        atol=1.0e-14,
        rtol=1.0e-14,
        derivative_residual_atol=1.0e-12,
        derivative_width_atol=1.0e-12,
        derivative_slope_floor=1.0e-8,
    )


def run() -> InvestigationResult:
    """Run a deterministic analytic-root and IFT sensitivity audit."""
    enable_high_precision()
    theta = jnp.asarray(2.0, dtype=jnp.float64)
    value = safeguarded_bracketed_root(
        lambda x: _residual(x, theta),
        0.0,
        2.0,
        max_steps=96,
        atol=1.0e-14,
        rtol=1.0e-14,
    )
    implicit = _implicit(theta)
    derivative = jax.grad(lambda parameter: _implicit(parameter).root)(theta)
    analytic_root = jnp.sqrt(theta)
    analytic_derivative = 1.0 / (2.0 * analytic_root)
    width = implicit.primal.final_bracket.hi - implicit.primal.final_bracket.lo
    metrics = (
        InvestigationMetric(
            "root.value", "x_star", float(value.root), "coordinate units"
        ),
        InvestigationMetric(
            "root.absolute_residual",
            "abs(f(x_star))",
            abs(float(value.residual)),
            "function units",
        ),
        InvestigationMetric(
            "root.bracket_width", "Delta_x", float(width), "coordinate units"
        ),
        InvestigationMetric(
            "root.analytic_value_error",
            "abs(x_star - sqrt(theta))",
            abs(float(value.root - analytic_root)),
            "coordinate units",
        ),
        InvestigationMetric(
            "root.implicit_derivative",
            "dx_star/dtheta",
            float(derivative),
            "coordinate units per parameter unit",
        ),
        InvestigationMetric(
            "root.derivative_absolute_error",
            "abs(d_AD - d_analytic)",
            abs(float(derivative - analytic_derivative)),
            "coordinate units per parameter unit",
        ),
        InvestigationMetric(
            "root.function_evaluations",
            "N_eval",
            int(value.n_evaluations),
            "evaluations",
        ),
    )
    checks = (
        AuditCheck(
            "root.value-converged", bool(value.converged), "typed terminal status"
        ),
        AuditCheck(
            "root.analytic-identity",
            metrics[3].value <= 1.0e-12,
            "independent square-root identity",
        ),
        AuditCheck(
            "root.positive-branch-unique-smooth",
            True,
            "analytic fixture: x^2-theta is smooth and strictly increasing for x > 0 because df/dx = 2x > 0",
        ),
        AuditCheck(
            "root.implicit-certified",
            bool(implicit.certified),
            "caller uniqueness/smoothness assertions plus checked convergence, finite state, residual, width, and slope gates",
        ),
        AuditCheck(
            "root.derivative-identity",
            metrics[5].value <= 1.0e-9,
            "analytic implicit derivative",
        ),
    )
    claim = calibrated_claim(
        checks,
        "The tested unique smooth quadratic root has a certified implicit sensitivity; this does not make the value-first branch history an IFT derivative.",
    )
    result = InvestigationResult(
        "root-values-and-sensitivities",
        "The value-first solve should locate sqrt(theta); only the separately certified implicit API should expose the ideal-root sensitivity.",
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
