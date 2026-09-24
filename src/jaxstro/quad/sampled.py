"""Canonical sampled-data integration methods."""

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def _along(values: Array, axis: int, ndim: int) -> Array:
    """Reshape a 1D per-sample vector to broadcast along ``axis`` of an array."""
    shape = [1] * ndim
    shape[axis] = values.shape[0]
    return values.reshape(shape)


@partial(jax.jit, static_argnames="axis")
def trapezoid(
    y: Float[Array, "..."],
    x: Optional[Float[Array, " n"]] = None,
    *,
    dx: float = 1.0,
    axis: int = -1,
) -> Float[Array, "..."]:
    """Integrate sampled values with the composite trapezoidal rule."""
    y = jnp.asarray(y)
    idx_left = jnp.arange(0, y.shape[axis] - 1)
    idx_right = jnp.arange(1, y.shape[axis])
    y_left = jnp.take(y, indices=idx_left, axis=axis)
    y_right = jnp.take(y, indices=idx_right, axis=axis)
    if x is None:
        return 0.5 * jnp.sum(y_left + y_right, axis=axis) * dx
    x = jnp.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be 1D if provided")
    if x.shape[0] != y.shape[axis]:
        raise ValueError(
            "x and y must have matching lengths along the integration axis"
        )
    widths = _along(jnp.diff(x), axis, y.ndim)
    return jnp.sum(0.5 * (y_left + y_right) * widths, axis=axis)


@partial(jax.jit, static_argnames="axis")
def cumulative_trapezoid(
    y: Float[Array, "..."],
    x: Optional[Float[Array, " n"]] = None,
    *,
    dx: float = 1.0,
    axis: int = -1,
) -> Float[Array, "..."]:
    """Return cumulative trapezoidal sums with an initial zero."""
    y = jnp.asarray(y)
    idx_left = jnp.arange(0, y.shape[axis] - 1)
    idx_right = jnp.arange(1, y.shape[axis])
    y_left = jnp.take(y, idx_left, axis=axis)
    y_right = jnp.take(y, idx_right, axis=axis)
    if x is None:
        cumsum = jnp.cumsum(0.5 * (y_left + y_right), axis=axis) * dx
    else:
        x = jnp.asarray(x)
        if x.ndim != 1:
            raise ValueError("x must be 1D if provided")
        if x.shape[0] != y.shape[axis]:
            raise ValueError(
                "x and y must have matching lengths along the integration axis"
            )
        widths = _along(jnp.diff(x), axis, y.ndim)
        cumsum = jnp.cumsum(0.5 * (y_left + y_right) * widths, axis=axis)
    pad_shape = list(cumsum.shape)
    pad_shape[axis] = 1
    zeros = jnp.zeros(pad_shape, dtype=cumsum.dtype)
    return jnp.concatenate([zeros, cumsum], axis=axis)


def simpson(
    y: Float[Array, "..."],
    x: Optional[Float[Array, " n"]] = None,
    *,
    dx: float = 1.0,
    axis: int = -1,
) -> Float[Array, "..."]:
    """Integrate sampled values with the composite Simpson rule.

    With ``x``, each two-interval panel uses the nonuniform Simpson weights,
    exact for quadratics on any grid; on a uniform grid they reduce to
    ``h / 3 * (y0 + 4 y1 + y2)``. Without ``x``, the scalar spacing ``dx`` is
    used. The number of samples along ``axis`` must be odd and at least 3.
    """
    return _simpson_core(y, x, dx=dx, axis=axis)


def cumulative_simpson(
    y: Float[Array, "..."],
    x: Optional[Float[Array, " n"]] = None,
    *,
    dx: float = 1.0,
    axis: int = -1,
) -> Float[Array, "..."]:
    """Return cumulative Simpson sums at panel endpoints.

    The panel weights are those of :func:`simpson`, so nonuniform ``x`` is
    supported on any axis.
    """
    return _cumulative_simpson_core(y, x, dx=dx, axis=axis)


def _simpson_panels(y: Array, x: Optional[Array], dx: float, axis: int) -> Array:
    """Integral of each two-interval panel along ``axis``."""
    n = y.shape[axis]
    if n < 3 or (n % 2) == 0:
        raise ValueError("simpson requires an odd number of points >= 3")
    idx = jnp.arange(n)
    y0 = jnp.take(y, idx[0:-2:2], axis=axis)
    y1 = jnp.take(y, idx[1:-1:2], axis=axis)
    y2 = jnp.take(y, idx[2::2], axis=axis)
    if x is None:
        return (jnp.asarray(dx) / 3.0) * (y0 + 4.0 * y1 + y2)
    x = jnp.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be 1D if provided")
    if x.shape[0] != n:
        raise ValueError(
            "x and y must have matching lengths along the integration axis"
        )
    widths = jnp.diff(x)
    h0 = _along(widths[0::2], axis, y.ndim)
    h1 = _along(widths[1::2], axis, y.ndim)
    # A repeated abscissa leaves no quadratic through three distinct points;
    # that panel is the trapezoid over its nonzero sub-interval, exact for
    # linear data. Sanitize the widths before the select so the unused Simpson
    # branch cannot put NaN into a derivative.
    degenerate = (h0 == 0.0) | (h1 == 0.0)
    safe_h0 = jnp.where(degenerate, 1.0, h0)
    safe_h1 = jnp.where(degenerate, 1.0, h1)
    total = safe_h0 + safe_h1
    # Composite Simpson on a nonuniform grid (the quadratic through the three
    # samples, integrated exactly over [x0, x2]).
    simpson = (total / 6.0) * (
        (2.0 - safe_h1 / safe_h0) * y0
        + (total * total / (safe_h0 * safe_h1)) * y1
        + (2.0 - safe_h0 / safe_h1) * y2
    )
    trapezoid = 0.5 * (y0 + y1) * h0 + 0.5 * (y1 + y2) * h1
    return jnp.where(degenerate, trapezoid, simpson)


@partial(jax.jit, static_argnames="axis")
def _cumulative_simpson_core(
    y: Float[Array, "..."],
    x: Optional[Float[Array, " n"]] = None,
    *,
    dx: float = 1.0,
    axis: int = -1,
) -> Float[Array, "..."]:
    y = jnp.asarray(y)
    cumsum = jnp.cumsum(_simpson_panels(y, x, dx, axis), axis=axis)
    pad_shape = list(cumsum.shape)
    pad_shape[axis] = 1
    zeros = jnp.zeros(pad_shape, dtype=cumsum.dtype)
    return jnp.concatenate([zeros, cumsum], axis=axis)


@partial(jax.jit, static_argnames="axis")
def _simpson_core(
    y: Float[Array, "..."],
    x: Optional[Float[Array, " n"]] = None,
    *,
    dx: float = 1.0,
    axis: int = -1,
) -> Float[Array, "..."]:
    y = jnp.asarray(y)
    return jnp.sum(_simpson_panels(y, x, dx, axis), axis=axis)


trapz = trapezoid
cumulative_trapz = cumulative_trapezoid

__all__ = [
    "cumulative_simpson",
    "cumulative_trapezoid",
    "cumulative_trapz",
    "simpson",
    "trapezoid",
    "trapz",
]
