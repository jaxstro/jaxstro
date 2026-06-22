"""Small optimization helpers for differentiable scientific objectives.

This module intentionally stops short of being an optimizer stack. It provides
loss kernels, scalar diagnostics, and a fixed-iteration Armijo line-search helper
that compose with JAX transforms.
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


class LineSearchResult(NamedTuple):
    """Result from a fixed-iteration Armijo backtracking search."""

    step: Float[Array, ""]
    value: Float[Array, ""]
    accepted: Bool[Array, ""]
    iterations: Array


def squared_loss(residual: Float[Array, "..."]) -> Float[Array, "..."]:
    """Return ``0.5 * residual**2`` elementwise."""
    residual = jnp.asarray(residual)
    return 0.5 * residual**2


def huber_loss(
    residual: Float[Array, "..."],
    *,
    delta: float = 1.0,
) -> Float[Array, "..."]:
    """Elementwise Huber loss with quadratic core and linear tails."""
    residual = jnp.asarray(residual)
    abs_r = jnp.abs(residual)
    quadratic = 0.5 * residual**2
    linear = delta * (abs_r - 0.5 * delta)
    return jnp.where(abs_r <= delta, quadratic, linear)


def pseudo_huber_loss(
    residual: Float[Array, "..."],
    *,
    delta: float = 1.0,
) -> Float[Array, "..."]:
    """Smooth approximation to the Huber loss."""
    residual = jnp.asarray(residual)
    scaled = residual / delta
    return delta**2 * (jnp.sqrt(1.0 + scaled**2) - 1.0)


def objective_summary(
    residuals: Float[Array, "..."],
    weights: Float[Array, "..."] | None = None,
) -> dict[str, Array]:
    """Summarize a residual vector with squared-loss diagnostics."""
    residuals = jnp.asarray(residuals)
    if weights is None:
        weighted_sq = residuals**2
        normalizer = jnp.asarray(residuals.size, dtype=residuals.dtype)
    else:
        weights = jnp.asarray(weights)
        weighted_sq = weights * residuals**2
        normalizer = jnp.sum(weights)

    normalizer_safe = jnp.where(normalizer > 0.0, normalizer, 1.0)
    sum_sq = jnp.sum(weighted_sq)
    loss = 0.5 * sum_sq
    mean_loss = loss / normalizer_safe
    rmse = jnp.sqrt(sum_sq / normalizer_safe)
    return {
        "loss": loss,
        "mean_loss": mean_loss,
        "rmse": rmse,
        "max_abs_residual": jnp.max(jnp.abs(residuals)),
        "n": jnp.asarray(residuals.size),
    }


def armijo_backtracking(
    f: Callable[[Float[Array, "..."]], Float[Array, ""]],
    x: Float[Array, "..."],
    direction: Float[Array, "..."],
    grad: Float[Array, "..."],
    *,
    initial_step: float = 1.0,
    contraction: float = 0.5,
    c1: float = 1e-4,
    max_steps: int = 20,
) -> LineSearchResult:
    """Run fixed-iteration Armijo backtracking along a descent direction.

    ``f`` and ``max_steps`` are static under ``jax.jit``. The scan always runs
    ``max_steps`` iterations and records the first accepted candidate.
    """
    x = jnp.asarray(x)
    direction = jnp.asarray(direction)
    grad = jnp.asarray(grad)
    f0 = f(x)
    directional_derivative = jnp.vdot(grad, direction)
    initial_step_arr = jnp.asarray(initial_step, dtype=x.dtype)

    def body(carry, i):
        best_step, best_value, accepted, iterations = carry
        step = initial_step_arr * contraction**i
        candidate = x + step * direction
        value = f(candidate)
        armijo_rhs = f0 + c1 * step * directional_derivative
        accepts = value <= armijo_rhs
        take = jnp.logical_and(~accepted, accepts)
        best_step = jnp.where(take, step, best_step)
        best_value = jnp.where(take, value, best_value)
        iterations = jnp.where(take, i + 1, iterations)
        accepted = jnp.logical_or(accepted, accepts)
        return (best_step, best_value, accepted, iterations), None

    last_step = initial_step_arr * contraction ** (max_steps - 1)
    init = (
        last_step,
        f(x + last_step * direction),
        jnp.asarray(False),
        jnp.asarray(max_steps),
    )
    (step, value, accepted, iterations), _ = jax.lax.scan(
        body,
        init,
        jnp.arange(max_steps),
    )
    return LineSearchResult(
        step=step, value=value, accepted=accepted, iterations=iterations
    )


def gradient_inf_norm(grad: Float[Array, "..."]) -> Float[Array, ""]:
    """Return the infinity norm of a gradient-like array."""
    return jnp.max(jnp.abs(jnp.asarray(grad)))


def relative_step_norm(
    x_new: Float[Array, "..."],
    x_old: Float[Array, "..."],
    *,
    scale_floor: float = 1e-12,
) -> Float[Array, ""]:
    """Return ``||x_new - x_old|| / max(||x_old||, scale_floor)``."""
    x_new = jnp.asarray(x_new)
    x_old = jnp.asarray(x_old)
    denominator = jnp.maximum(jnp.linalg.norm(x_old), scale_floor)
    return jnp.linalg.norm(x_new - x_old) / denominator


def convergence_summary(
    *,
    x_new: Float[Array, "..."],
    x_old: Float[Array, "..."],
    grad: Float[Array, "..."],
    loss_new: Float[Array, ""],
    loss_old: Float[Array, ""],
    step_tol: float = 1e-8,
    grad_tol: float = 1e-8,
    loss_tol: float = 1e-10,
) -> dict[str, Array]:
    """Return optimizer-agnostic convergence diagnostics."""
    step_norm = relative_step_norm(x_new, x_old)
    grad_norm = gradient_inf_norm(grad)
    loss_scale = jnp.maximum(jnp.abs(loss_old), 1.0)
    loss_change = jnp.abs(loss_new - loss_old) / loss_scale
    step_converged = step_norm <= step_tol
    grad_converged = grad_norm <= grad_tol
    loss_converged = loss_change <= loss_tol
    converged = jnp.logical_and(
        jnp.logical_and(step_converged, grad_converged),
        loss_converged,
    )
    return {
        "converged": converged,
        "step_converged": step_converged,
        "grad_converged": grad_converged,
        "loss_converged": loss_converged,
        "step_norm": step_norm,
        "grad_norm": grad_norm,
        "loss_change": loss_change,
    }


__all__ = [
    "LineSearchResult",
    "squared_loss",
    "huber_loss",
    "pseudo_huber_loss",
    "objective_summary",
    "armijo_backtracking",
    "gradient_inf_norm",
    "relative_step_norm",
    "convergence_summary",
]
