# src/jaxstro/numerics/integration.py
"""
Lightweight numerical integration helpers for JAX.

These are small utilities for 1D quadrature over gridded data.
For ODE solving and more complex integrators, prefer diffrax.
"""

from functools import partial
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
        idx_left = jnp.arange(0, y.shape[axis] - 1)
        idx_right = jnp.arange(1, y.shape[axis])
        slice1 = jnp.take(y, indices=idx_left, axis=axis)
        slice2 = jnp.take(y, indices=idx_right, axis=axis)
        return 0.5 * jnp.sum(slice1 + slice2, axis=axis)
    else:
        x = jnp.asarray(x)
        if x.ndim != 1:
            raise ValueError("x must be 1D if provided")
        if x.shape[0] != y.shape[axis]:
            raise ValueError(
                "x and y must have matching lengths along the integration axis"
            )

        dx_arr = jnp.diff(x)
        idx_left = jnp.arange(0, x.shape[0] - 1)
        idx_right = jnp.arange(1, x.shape[0])

        y_left = jnp.take(y, idx_left, axis=axis)
        y_right = jnp.take(y, idx_right, axis=axis)

        integrand = 0.5 * (y_left + y_right) * dx_arr
        return jnp.sum(integrand, axis=axis)


@partial(jax.jit, static_argnames="axis")
def cumulative_trapz(
    y: Array,
    x: Optional[Array] = None,
    *,
    dx: float = 1.0,
    axis: int = -1,
) -> Array:
    """
    Cumulative trapezoidal integration along a given axis.

    The first element along the integration axis is zero, so the result has the
    same length as ``y`` along ``axis``.

    Two spacing modes:

    - **Uniform** (``x is None``): constant spacing ``dx`` (default ``1.0``). The
      uniform path is **dx-outside**: it computes ``cumsum(0.5 * (y_left + y_right))``
      first and multiplies by the scalar ``dx`` exactly once at the end. This is the
      single-source-of-truth ordering shared with progenax's ``cumulative_trapezoid``
      and is byte-for-byte identical to it on shared inputs.

      Note on the ~1-ulp relationship to the former *dx-inside* ordering
      (``cumsum(0.5 * dx * (y_left + y_right))``, i.e. multiply each increment by
      ``dx`` before the cumsum): the two forms are mathematically equal but the
      floating-point rounding differs by at most ~1 ulp because the dx factor is
      applied at a different point in the summation. Standardizing on dx-outside is
      the intended reconciliation across the ecosystem.

    - **Non-uniform** (``x`` provided): per-interval spacing ``diff(x)``; ``dx`` is
      ignored. Each trapezoid increment carries its own ``dx_arr`` factor inside the
      cumsum (there is no single scalar to factor out).
    """
    y = jnp.asarray(y)
    if x is None:
        idx_left = jnp.arange(0, y.shape[axis] - 1)
        idx_right = jnp.arange(1, y.shape[axis])

        y_left = jnp.take(y, idx_left, axis=axis)
        y_right = jnp.take(y, idx_right, axis=axis)
        # dx-OUTSIDE: cumsum the raw trapezoid increments, then scale by dx once.
        cumsum = jnp.cumsum(0.5 * (y_left + y_right), axis=axis) * dx
    else:
        x = jnp.asarray(x)
        if x.ndim != 1:
            raise ValueError("x must be 1D if provided")
        if x.shape[0] != y.shape[axis]:
            raise ValueError(
                "x and y must have matching lengths along the integration axis"
            )
        dx_arr = jnp.diff(x)

        idx_left = jnp.arange(0, x.shape[0] - 1)
        idx_right = jnp.arange(1, x.shape[0])

        y_left = jnp.take(y, idx_left, axis=axis)
        y_right = jnp.take(y, idx_right, axis=axis)
        contrib = 0.5 * (y_left + y_right) * dx_arr
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
        dx: Array = jnp.asarray(1.0)
    else:
        x = jnp.asarray(x)
        if x.ndim != 1:
            raise ValueError("x must be 1D if provided")
        if x.shape[0] != n:
            raise ValueError(
                "x and y must have matching lengths along the integration axis"
            )
        dx = (x[-1] - x[0]) / (n - 1)

    idx = jnp.arange(n)
    y0 = jnp.take(y, idx[0:-2:2], axis=axis)
    y1 = jnp.take(y, idx[1:-1:2], axis=axis)
    y2 = jnp.take(y, idx[2::2], axis=axis)

    return (dx / 3.0) * jnp.sum(y0 + 4.0 * y1 + y2, axis=axis)


__all__ = ["trapz", "cumulative_trapz", "simpson"]
