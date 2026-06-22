"""Finite-difference diagnostics for AD validation tests.

These helpers are intentionally testing utilities, not model primitives. They
compare numerical finite differences against JAX autodiff paths and return
structured reports with the tolerances used for the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

DiagnosticKind = Literal["gradient", "jacobian", "directional_derivative"]


@dataclass(frozen=True)
class DifferenceReport:
    """Structured comparison between autodiff and finite differences."""

    kind: DiagnosticKind
    passed: bool
    ad: Float[Array, "..."]
    fd: Float[Array, "..."]
    abs_error: Float[Array, "..."]
    rel_error: Float[Array, "..."]
    max_abs_error: float
    max_rel_error: float
    atol: float
    rtol: float
    eps: float


ScalarFn = Callable[[jax.Array], jax.Array]


def finite_difference_grad(
    f: ScalarFn,
    x: Float[Array, "..."],
    *,
    eps: float = 1e-6,
) -> Float[Array, "..."]:
    """Central finite-difference gradient of scalar-output ``f`` at ``x``."""
    x = jnp.asarray(x, dtype=jnp.float64)
    flat = x.ravel()
    grad = []
    for idx in range(flat.shape[0]):
        plus = flat.at[idx].add(eps).reshape(x.shape)
        minus = flat.at[idx].add(-eps).reshape(x.shape)
        grad.append((f(plus) - f(minus)) / (2.0 * eps))
    return jnp.reshape(jnp.asarray(grad, dtype=x.dtype), x.shape)


def finite_difference_jacobian(
    f: Callable[[jax.Array], jax.Array],
    x: Float[Array, "..."],
    *,
    eps: float = 1e-6,
) -> Float[Array, "..."]:
    """Central finite-difference Jacobian with shape ``f(x).shape + x.shape``."""
    x = jnp.asarray(x, dtype=jnp.float64)
    y0 = jnp.asarray(f(x), dtype=jnp.float64)
    flat = x.ravel()
    columns = []
    for idx in range(flat.shape[0]):
        plus = flat.at[idx].add(eps).reshape(x.shape)
        minus = flat.at[idx].add(-eps).reshape(x.shape)
        columns.append((jnp.asarray(f(plus)) - jnp.asarray(f(minus))) / (2.0 * eps))
    jac_flat = jnp.stack(columns, axis=-1)
    return jnp.reshape(jac_flat, y0.shape + x.shape)


def directional_derivative(
    f: ScalarFn,
    x: Float[Array, "..."],
    direction: Float[Array, "..."],
    *,
    eps: float = 1e-6,
) -> Float[Array, ""]:
    """Central finite-difference directional derivative along ``direction``."""
    x = jnp.asarray(x, dtype=jnp.float64)
    direction = jnp.asarray(direction, dtype=jnp.float64)
    if direction.shape != x.shape:
        raise ValueError("directional_derivative direction shape must match x shape")
    return (f(x + eps * direction) - f(x - eps * direction)) / (2.0 * eps)


def compare_gradients(
    f: ScalarFn,
    x: Float[Array, "..."],
    *,
    eps: float = 1e-6,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> DifferenceReport:
    """Compare ``jax.grad(f)(x)`` with a central finite-difference gradient."""
    x = jnp.asarray(x, dtype=jnp.float64)
    ad = jax.grad(f)(x)
    fd = finite_difference_grad(f, x, eps=eps)
    return _difference_report(
        kind="gradient",
        ad=ad,
        fd=fd,
        eps=eps,
        atol=atol,
        rtol=rtol,
    )


def compare_jacobians(
    f: Callable[[jax.Array], jax.Array],
    x: Float[Array, "..."],
    *,
    eps: float = 1e-6,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> DifferenceReport:
    """Compare ``jax.jacrev(f)(x)`` with a central finite-difference Jacobian."""
    x = jnp.asarray(x, dtype=jnp.float64)
    ad = jax.jacrev(f)(x)
    fd = finite_difference_jacobian(f, x, eps=eps)
    return _difference_report(
        kind="jacobian",
        ad=ad,
        fd=fd,
        eps=eps,
        atol=atol,
        rtol=rtol,
    )


def check_directional_derivative(
    f: ScalarFn,
    x: Float[Array, "..."],
    direction: Float[Array, "..."],
    *,
    eps: float = 1e-6,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> DifferenceReport:
    """Compare AD and FD directional derivatives along ``direction``."""
    x = jnp.asarray(x, dtype=jnp.float64)
    direction = jnp.asarray(direction, dtype=jnp.float64)
    if direction.shape != x.shape:
        raise ValueError(
            "check_directional_derivative direction shape must match x shape"
        )
    ad = jnp.vdot(jax.grad(f)(x), direction)
    fd = directional_derivative(f, x, direction, eps=eps)
    return _difference_report(
        kind="directional_derivative",
        ad=ad,
        fd=fd,
        eps=eps,
        atol=atol,
        rtol=rtol,
    )


def _difference_report(
    *,
    kind: DiagnosticKind,
    ad: Float[Array, "..."],
    fd: Float[Array, "..."],
    eps: float,
    atol: float,
    rtol: float,
) -> DifferenceReport:
    ad = jnp.asarray(ad, dtype=jnp.float64)
    fd = jnp.asarray(fd, dtype=jnp.float64)
    abs_error = jnp.abs(ad - fd)
    rel_error = abs_error / jnp.maximum(
        jnp.abs(fd), jnp.asarray(atol, dtype=jnp.float64)
    )
    threshold = atol + rtol * jnp.abs(fd)
    passed = bool(jnp.all(abs_error <= threshold))
    return DifferenceReport(
        kind=kind,
        passed=passed,
        ad=ad,
        fd=fd,
        abs_error=abs_error,
        rel_error=rel_error,
        max_abs_error=float(jnp.max(abs_error)),
        max_rel_error=float(jnp.max(rel_error)),
        atol=atol,
        rtol=rtol,
        eps=eps,
    )


__all__ = [
    "DifferenceReport",
    "check_directional_derivative",
    "compare_gradients",
    "compare_jacobians",
    "directional_derivative",
    "finite_difference_grad",
    "finite_difference_jacobian",
]
