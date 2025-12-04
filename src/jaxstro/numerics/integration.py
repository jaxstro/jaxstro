# src/jaxstro/numerics/integration.py
"""
Lightweight numerical integration helpers for JAX.

These are small utilities for 1D quadrature over gridded data.
For ODE solving and more complex integrators, prefer diffrax.
"""

from typing import Optional

import jax
import jax.numpy as jnp

from .types import Array


@jax.jit
def trapz(
    y: Array,
    x: Optional[Array] = None,
    *,
    axis: int = -1,
) -> Array:
    """
    Trapezoidal rule integration along a given axis.
    """
    y = jnp.asarray(y)
    if x is None:
        dx = 1.0
        slice1 = jnp.take(y, indices=range(y.shape[axis] - 1), axis=axis)
        slice2 = jnp.take(y, indices=range(1, y.shape[axis]), axis=axis)
        return 0.5 * dx * jnp.sum(slice1 + slice2, axis=axis)
    else:
        x = jnp.asarray(x)
        if x.ndim != 1:
            raise ValueError("x must be 1D if provided")
        if x.shape[0] != y.shape[axis]:
            raise ValueError("x and y must have matching lengths along the integration axis")

        dx = jnp.diff(x)
        idx_left = jnp.arange(0, x.shape[0] - 1)
        idx_right = jnp.arange(1, x.shape[0])

        y_left = jnp.take(y, idx_left, axis=axis)
        y_right = jnp.take(y, idx_right, axis=axis)

        integrand = 0.5 * (y_left + y_right) * dx
        return jnp.sum(integrand, axis=axis)


@jax.jit
def cumulative_trapz(
    y: Array,
    x: Optional[Array] = None,
    *,
    axis: int = -1,
) -> Array:
    """
    Cumulative trapezoidal integration along a given axis.

    The first element along the integration axis is zero.
    """
    y = jnp.asarray(y)
    if x is None:
        dx = 1.0
        idx_left = jnp.arange(0, y.shape[axis] - 1)
        idx_right = jnp.arange(1, y.shape[axis])

        y_left = jnp.take(y, idx_left, axis=axis)
        y_right = jnp.take(y, idx_right, axis=axis)
        contrib = 0.5 * dx * (y_left + y_right)
    else:
        x = jnp.asarray(x)
        if x.ndim != 1:
            raise ValueError("x must be 1D if provided")
        if x.shape[0] != y.shape[axis]:
            raise ValueError("x and y must have matching lengths along the integration axis")
        dx = jnp.diff(x)

        idx_left = jnp.arange(0, x.shape[0] - 1)
        idx_right = jnp.arange(1, x.shape[0])

        y_left = jnp.take(y, idx_left, axis=axis)
        y_right = jnp.take(y, idx_right, axis=axis)
        contrib = 0.5 * (y_left + y_right) * dx

    cumsum = jnp.cumsum(contrib, axis=axis)

    pad_shape = list(cumsum.shape)
    pad_shape[axis] = 1
    zeros = jnp.zeros(pad_shape, dtype=cumsum.dtype)

    result = jnp.concatenate([zeros, cumsum], axis=axis)
    return result


@jax.jit
def simpson(
    y: Array,
    x: Optional[Array] = None,
    *,
    axis: int = -1,
) -> Array:
    """
    Simpson's rule integration along a given axis.

    Requires an odd number of samples along the integration axis.
    """
    y = jnp.asarray(y)
    n = y.shape[axis]
    if n < 3 or (n % 2) == 0:
        raise ValueError("simpson requires an odd number of points >= 3")

    if x is None:
        dx = 1.0
    else:
        x = jnp.asarray(x)
        if x.ndim != 1:
            raise ValueError("x must be 1D if provided")
        if x.shape[0] != n:
            raise ValueError("x and y must have matching lengths along the integration axis")
        dx = (x[-1] - x[0]) / (n - 1)

    idx = jnp.arange(n)
    y0 = jnp.take(y, idx[0:-2:2], axis=axis)
    y1 = jnp.take(y, idx[1:-1:2], axis=axis)
    y2 = jnp.take(y, idx[2::2], axis=axis)

    return (dx / 3.0) * jnp.sum(y0 + 4.0 * y1 + y2, axis=axis)


__all__ = ["trapz", "cumulative_trapz", "simpson"]
