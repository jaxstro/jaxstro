"""Canonical sampled-data integration methods."""

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxstro.numerics.checks import try_concrete_bool


@jax.jit
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
    return jnp.sum(0.5 * (y_left + y_right) * jnp.diff(x), axis=axis)


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
        cumsum = jnp.cumsum(0.5 * (y_left + y_right) * jnp.diff(x), axis=axis)
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
    """Integrate uniformly sampled values with the composite Simpson rule."""
    if x is not None:
        x = jnp.asarray(x)
        n = jnp.asarray(y).shape[axis]
        if x.ndim == 1 and x.shape[0] == n:
            step = (x[-1] - x[0]) / (n - 1)
            is_uniform = try_concrete_bool(jnp.allclose(jnp.diff(x), step))
            if is_uniform is False:
                raise ValueError(
                    "simpson assumes uniform spacing in x; got a non-uniform "
                    "grid. Resample to a uniform grid or use trapezoid for "
                    "arbitrary spacing."
                )
    return _simpson_core(y, x, dx=dx, axis=axis)


def cumulative_simpson(
    y: Float[Array, "..."],
    x: Optional[Float[Array, " n"]] = None,
    *,
    dx: float = 1.0,
    axis: int = -1,
) -> Float[Array, "..."]:
    """Return cumulative Simpson sums at panel endpoints."""
    y = jnp.asarray(y)
    n = y.shape[axis]
    if n < 3 or (n % 2) == 0:
        raise ValueError("cumulative_simpson requires an odd number of points >= 3")
    if x is not None:
        x = jnp.asarray(x)
        if x.ndim != 1:
            raise ValueError("x must be 1D if provided")
        if x.shape[0] != n:
            raise ValueError(
                "x and y must have matching lengths along the integration axis"
            )
        step = (x[-1] - x[0]) / (n - 1)
        is_uniform = try_concrete_bool(jnp.allclose(jnp.diff(x), step))
        if is_uniform is False:
            raise ValueError(
                "cumulative_simpson assumes uniform spacing in x; got a "
                "non-uniform grid."
            )
    return _cumulative_simpson_core(y, x, dx=dx, axis=axis)


@partial(jax.jit, static_argnames="axis")
def _cumulative_simpson_core(
    y: Float[Array, "..."],
    x: Optional[Float[Array, " n"]] = None,
    *,
    dx: float = 1.0,
    axis: int = -1,
) -> Float[Array, "..."]:
    y = jnp.asarray(y)
    n = y.shape[axis]
    if n < 3 or (n % 2) == 0:
        raise ValueError("cumulative_simpson requires an odd number of points >= 3")
    if x is None:
        step = jnp.asarray(dx)
    else:
        x = jnp.asarray(x)
        if x.ndim != 1:
            raise ValueError("x must be 1D if provided")
        if x.shape[0] != n:
            raise ValueError(
                "x and y must have matching lengths along the integration axis"
            )
        step = (x[-1] - x[0]) / (n - 1)
    idx = jnp.arange(n)
    y0 = jnp.take(y, idx[0:-2:2], axis=axis)
    y1 = jnp.take(y, idx[1:-1:2], axis=axis)
    y2 = jnp.take(y, idx[2::2], axis=axis)
    panels = (step / 3.0) * (y0 + 4.0 * y1 + y2)
    cumsum = jnp.cumsum(panels, axis=axis)
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
    n = y.shape[axis]
    if n < 3 or (n % 2) == 0:
        raise ValueError("simpson requires an odd number of points >= 3")
    if x is None:
        step: Float[Array, ""] = jnp.asarray(dx)
    else:
        x = jnp.asarray(x)
        if x.ndim != 1:
            raise ValueError("x must be 1D if provided")
        if x.shape[0] != n:
            raise ValueError(
                "x and y must have matching lengths along the integration axis"
            )
        step = (x[-1] - x[0]) / (n - 1)
    idx = jnp.arange(n)
    y0 = jnp.take(y, idx[0:-2:2], axis=axis)
    y1 = jnp.take(y, idx[1:-1:2], axis=axis)
    y2 = jnp.take(y, idx[2::2], axis=axis)
    return (step / 3.0) * jnp.sum(y0 + 4.0 * y1 + y2, axis=axis)


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
