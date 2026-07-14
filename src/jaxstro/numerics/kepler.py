"""Universal-variable Cartesian Kepler propagation.

This module owns package-independent two-body numerical mechanics. Callers own
units, physical domain policy, object identity, and state-commit decisions.
"""

from typing import NamedTuple

import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array

KEPLER_STATUS_CONVERGED = 0
KEPLER_STATUS_INVALID_INPUT = 1
KEPLER_STATUS_NONFINITE_ITERATION = 2
KEPLER_STATUS_SINGULAR_RADIUS = 3
KEPLER_STATUS_MAX_STEPS = 4

_STUMPFF_SERIES_TERMS = 8
_STUMPFF_SERIES_THRESHOLD = 1.0


class UniversalKeplerResult(NamedTuple):
    """Cartesian conic result with one exhaustive numerical status."""

    position: Array
    velocity: Array
    universal_anomaly: Array
    residual: Array
    iterations: Array
    status: Array
    valid: Array


class _KeplerCarry(NamedTuple):
    anomaly: Array
    residual: Array
    radius: Array
    iterations: Array
    converged: Array
    finite: Array
    singular: Array


def _stumpff_series(z: Array) -> tuple[Array, Array]:
    """Evaluate the fixed eight-term power series for C(z) and S(z)."""

    c = jnp.asarray(0.0, dtype=z.dtype)
    s = jnp.asarray(0.0, dtype=z.dtype)
    c_term = jnp.asarray(0.5, dtype=z.dtype)
    s_term = jnp.asarray(1.0 / 6.0, dtype=z.dtype)
    for k in range(_STUMPFF_SERIES_TERMS):
        c = c + c_term
        s = s + s_term
        c_term = c_term * (-z) / ((2 * k + 3) * (2 * k + 4))
        s_term = s_term * (-z) / ((2 * k + 4) * (2 * k + 5))
    return c, s


def _stumpff_pair(z: Array) -> tuple[Array, Array]:
    """Evaluate Stumpff C(z), S(z) with sanitized branch expressions."""

    z = jnp.asarray(z, dtype=jnp.result_type(z, 0.0))
    abs_z = jnp.abs(z)
    threshold = jnp.asarray(_STUMPFF_SERIES_THRESHOLD, dtype=z.dtype)
    use_series = abs_z <= threshold

    series_z = jnp.where(use_series, z, jnp.zeros_like(z))
    c_series, s_series = _stumpff_series(series_z)

    positive_z = jnp.where(z > threshold, z, jnp.ones_like(z))
    positive_root = jnp.sqrt(positive_z)
    c_positive = (1.0 - jnp.cos(positive_root)) / positive_z
    s_positive = (positive_root - jnp.sin(positive_root)) / positive_root**3

    negative_abs_z = jnp.where(z < -threshold, -z, jnp.ones_like(z))
    negative_root = jnp.sqrt(negative_abs_z)
    c_negative = (jnp.cosh(negative_root) - 1.0) / negative_abs_z
    s_negative = (jnp.sinh(negative_root) - negative_root) / negative_root**3

    c_outer = jnp.where(z > 0.0, c_positive, c_negative)
    s_outer = jnp.where(z > 0.0, s_positive, s_negative)
    return (
        jnp.where(use_series, c_series, c_outer),
        jnp.where(use_series, s_series, s_outer),
    )


def _tof_residual_and_radius(
    anomaly: Array,
    alpha_bar: Array,
    rv_bar: Array,
    tau: Array,
) -> tuple[Array, Array]:
    """Return dimensionless time-of-flight residual and its radius derivative."""

    z = alpha_bar * anomaly**2
    c, s = _stumpff_pair(z)
    residual = (
        rv_bar * anomaly**2 * c + (1.0 - alpha_bar) * anomaly**3 * s + anomaly - tau
    )
    radius = anomaly**2 * c + rv_bar * anomaly * (1.0 - z * s) + 1.0 - z * c
    return residual, radius


def _initial_anomaly(
    alpha_bar: Array,
    rv_bar: Array,
    tau: Array,
) -> Array:
    """Choose the approved deterministic conic-aware initial anomaly."""

    near_parabolic = jnp.sqrt(jnp.finfo(alpha_bar.dtype).eps)
    sign_tau = jnp.where(tau < 0.0, -jnp.ones_like(tau), jnp.ones_like(tau))
    safe_negative_alpha = jnp.where(
        alpha_bar < -near_parabolic,
        -alpha_bar,
        jnp.ones_like(alpha_bar),
    )
    root_negative_alpha = jnp.sqrt(safe_negative_alpha)
    denominator = rv_bar + sign_tau / root_negative_alpha * (1.0 - alpha_bar)
    safe_denominator = jnp.where(
        jnp.isfinite(denominator) & (denominator != 0.0),
        denominator,
        jnp.ones_like(denominator),
    )
    log_argument = (-2.0 * alpha_bar * tau) / safe_denominator
    safe_log_argument = jnp.where(
        jnp.isfinite(log_argument) & (log_argument > 0.0),
        log_argument,
        jnp.ones_like(log_argument),
    )
    hyperbolic = sign_tau / root_negative_alpha * jnp.log(safe_log_argument)
    hyperbolic_valid = (
        jnp.isfinite(log_argument)
        & (log_argument > 0.0)
        & jnp.isfinite(hyperbolic)
        & ((hyperbolic == 0.0) | (jnp.signbit(hyperbolic) == jnp.signbit(tau)))
    )
    hyperbolic = jnp.where(hyperbolic_valid, hyperbolic, tau)
    return jnp.where(
        alpha_bar > near_parabolic,
        alpha_bar * tau,
        jnp.where(alpha_bar < -near_parabolic, hyperbolic, tau),
    )


def _terminal_status(
    *,
    input_valid: Array,
    finite: Array,
    singular: Array,
    converged: Array,
) -> Array:
    """Select one exhaustive status with fail-closed precedence."""

    status = jnp.asarray(KEPLER_STATUS_MAX_STEPS, dtype=jnp.int32)
    status = jnp.where(converged, KEPLER_STATUS_CONVERGED, status)
    status = jnp.where(singular, KEPLER_STATUS_SINGULAR_RADIUS, status)
    status = jnp.where(
        input_valid & ~finite,
        KEPLER_STATUS_NONFINITE_ITERATION,
        status,
    )
    return jnp.where(~input_valid, KEPLER_STATUS_INVALID_INPUT, status)


def universal_kepler_step(
    position: Array,
    velocity: Array,
    mu: Array,
    dt: Array,
    *,
    max_steps: int = 12,
    residual_tolerance: float = 1.0e-12,
) -> UniversalKeplerResult:
    """Propagate one relative Cartesian state across any Newtonian conic.

    Units are caller-owned and must be mutually consistent. Invalid numerical
    results retain the original position and velocity.
    """

    if position.shape != (3,) or velocity.shape != (3,):
        raise ValueError("position and velocity must each have shape (3,).")
    if not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("max_steps must be a non-negative integer.")
    if not isinstance(residual_tolerance, (int, float)):
        raise TypeError("residual_tolerance must be a real scalar.")
    if not 0.0 < float(residual_tolerance) < float("inf"):
        raise ValueError("residual_tolerance must be positive and finite.")

    dtype = jnp.result_type(position, velocity, mu, dt, 0.0)
    original_position = jnp.asarray(position)
    original_velocity = jnp.asarray(velocity)
    position = jnp.asarray(position, dtype=dtype)
    velocity = jnp.asarray(velocity, dtype=dtype)
    mu = jnp.asarray(mu, dtype=dtype)
    dt = jnp.asarray(dt, dtype=dtype)
    radius_floor = jnp.asarray(1.0e-14, dtype=dtype)
    tolerance = jnp.asarray(residual_tolerance, dtype=dtype)

    input_finite = (
        jnp.all(jnp.isfinite(position))
        & jnp.all(jnp.isfinite(velocity))
        & jnp.isfinite(mu)
        & jnp.isfinite(dt)
    )
    raw_radius = jnp.linalg.norm(position)
    input_valid = input_finite & (mu > 0.0) & (raw_radius > 0.0)

    safe_position = jnp.where(
        input_valid,
        position,
        jnp.asarray([1.0, 0.0, 0.0], dtype=dtype),
    )
    safe_velocity = jnp.where(input_valid, velocity, jnp.zeros_like(velocity))
    safe_mu = jnp.where(input_valid, mu, jnp.ones_like(mu))
    safe_dt = jnp.where(input_valid, dt, jnp.zeros_like(dt))

    length_scale = jnp.linalg.norm(safe_position)
    time_scale = jnp.sqrt(length_scale**3 / safe_mu)
    velocity_scale = jnp.sqrt(safe_mu / length_scale)
    position_bar = safe_position / length_scale
    velocity_bar = safe_velocity / velocity_scale
    alpha_bar = 2.0 - jnp.dot(velocity_bar, velocity_bar)
    rv_bar = jnp.dot(position_bar, velocity_bar)
    tau = safe_dt / time_scale

    anomaly = _initial_anomaly(alpha_bar, rv_bar, tau)
    residual, radius = _tof_residual_and_radius(
        anomaly,
        alpha_bar,
        rv_bar,
        tau,
    )
    guess_valid = (
        jnp.isfinite(anomaly) & jnp.isfinite(residual) & (radius > radius_floor)
    )
    fallback_residual, fallback_radius = _tof_residual_and_radius(
        tau,
        alpha_bar,
        rv_bar,
        tau,
    )
    anomaly = jnp.where(guess_valid, anomaly, tau)
    residual = jnp.where(guess_valid, residual, fallback_residual)
    radius = jnp.where(guess_valid, radius, fallback_radius)

    finite = (
        input_valid
        & jnp.isfinite(anomaly)
        & jnp.isfinite(residual)
        & jnp.isfinite(radius)
    )
    singular = input_valid & finite & (radius <= radius_floor)
    residual_bound = tolerance * jnp.maximum(1.0, jnp.abs(tau))
    converged = finite & ~singular & (jnp.abs(residual) <= residual_bound)
    initial = _KeplerCarry(
        anomaly=anomaly,
        residual=residual,
        radius=radius,
        iterations=jnp.asarray(0, dtype=jnp.int32),
        converged=converged,
        finite=finite,
        singular=singular,
    )

    def newton_step(
        carry: _KeplerCarry,
        _: None,
    ) -> tuple[_KeplerCarry, None]:
        execute = input_valid & carry.finite & ~carry.singular & ~carry.converged
        safe_radius = jnp.where(execute, carry.radius, jnp.ones_like(carry.radius))
        proposal = jnp.where(
            execute,
            carry.anomaly - carry.residual / safe_radius,
            carry.anomaly,
        )
        next_residual, next_radius = _tof_residual_and_radius(
            proposal,
            alpha_bar,
            rv_bar,
            tau,
        )
        proposal_finite = (
            jnp.isfinite(proposal)
            & jnp.isfinite(next_residual)
            & jnp.isfinite(next_radius)
        )
        accepted = execute & proposal_finite
        next_singular = accepted & (next_radius <= radius_floor)
        anomaly_out = jnp.where(accepted, proposal, carry.anomaly)
        residual_out = jnp.where(accepted, next_residual, carry.residual)
        radius_out = jnp.where(accepted, next_radius, carry.radius)
        finite_out = carry.finite & (~execute | proposal_finite)
        singular_out = carry.singular | next_singular
        converged_out = carry.converged | (
            accepted & ~next_singular & (jnp.abs(next_residual) <= residual_bound)
        )
        iterations_out = carry.iterations + jnp.asarray(
            accepted,
            dtype=jnp.int32,
        )
        return (
            _KeplerCarry(
                anomaly_out,
                residual_out,
                radius_out,
                iterations_out,
                converged_out,
                finite_out,
                singular_out,
            ),
            None,
        )

    final, _ = lax.scan(newton_step, initial, xs=None, length=max_steps)

    z = alpha_bar * final.anomaly**2
    c, s = _stumpff_pair(z)
    f = 1.0 - final.anomaly**2 * c
    g = time_scale * (tau - final.anomaly**3 * s)
    candidate_position = f * safe_position + g * safe_velocity
    final_radius_bar = jnp.linalg.norm(candidate_position) / length_scale
    safe_final_radius_bar = jnp.where(
        final_radius_bar > radius_floor,
        final_radius_bar,
        jnp.ones_like(final_radius_bar),
    )
    fdot = (alpha_bar * final.anomaly**3 * s - final.anomaly) / (
        time_scale * safe_final_radius_bar
    )
    gdot = 1.0 - final.anomaly**2 * c / safe_final_radius_bar
    candidate_velocity = fdot * safe_position + gdot * safe_velocity

    reconstruction_finite = (
        jnp.all(jnp.isfinite(candidate_position))
        & jnp.all(jnp.isfinite(candidate_velocity))
        & jnp.isfinite(final_radius_bar)
    )
    finite_out = final.finite & reconstruction_finite
    singular_out = final.singular | (
        final.converged & reconstruction_finite & (final_radius_bar <= radius_floor)
    )
    converged_out = input_valid & final.converged & finite_out & ~singular_out

    status = _terminal_status(
        input_valid=input_valid,
        finite=finite_out,
        singular=singular_out,
        converged=converged_out,
    )
    valid = status == KEPLER_STATUS_CONVERGED

    return UniversalKeplerResult(
        position=jnp.where(valid, candidate_position, original_position),
        velocity=jnp.where(valid, candidate_velocity, original_velocity),
        universal_anomaly=jnp.where(
            input_valid,
            final.anomaly * jnp.sqrt(length_scale),
            jnp.zeros_like(final.anomaly),
        ),
        residual=jnp.where(input_valid, final.residual, jnp.inf),
        iterations=final.iterations,
        status=status,
        valid=valid,
    )


__all__ = [
    "KEPLER_STATUS_CONVERGED",
    "KEPLER_STATUS_INVALID_INPUT",
    "KEPLER_STATUS_NONFINITE_ITERATION",
    "KEPLER_STATUS_SINGULAR_RADIUS",
    "KEPLER_STATUS_MAX_STEPS",
    "UniversalKeplerResult",
    "universal_kepler_step",
]
