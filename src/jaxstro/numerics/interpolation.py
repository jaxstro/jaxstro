# src/jaxstro/numerics/interpolation.py
"""
1D interpolation utilities for JAX.

This module provides a simple, differentiable linear interpolator and a
small TabulatedFunction1D wrapper for table-based models (e.g. fits,
opacities, cooling curves).
"""

from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .checks import try_concrete_bool


def _move_axis_to_last(x: Float[Array, "..."], axis: int) -> Float[Array, "..."]:
    if axis < 0:
        axis = x.ndim + axis
    return jnp.moveaxis(x, axis, -1)


def _normalize_axis(axis: int, ndim: int) -> int:
    if ndim == 0:
        raise ValueError("interpolation values must have a sample axis")
    if axis < 0:
        axis = ndim + axis
    if axis < 0 or axis >= ndim:
        raise ValueError("interpolation axis is out of bounds")
    return axis


def _validate_x_y(x: Float[Array, " m"], y: Float[Array, "..."], axis: int) -> int:
    if x.ndim != 1:
        raise ValueError("interpolation x grid must be a 1D array")
    if x.shape[0] < 2:
        raise ValueError("interpolation requires at least two grid points")
    axis = _normalize_axis(axis, y.ndim)
    if y.shape[axis] != x.shape[0]:
        raise ValueError(
            "interpolation value axis length must equal "
            f"len(x)={x.shape[0]}; got axis length {y.shape[axis]}"
        )

    is_increasing = try_concrete_bool(jnp.all(jnp.diff(x) > 0))
    if is_increasing is False:
        raise ValueError(
            "interpolation requires x to be strictly increasing (x[i+1] > x[i])."
        )
    return axis


def _safe_div_zero(
    numerator: Float[Array, "..."],
    denominator: Float[Array, "..."],
) -> Float[Array, "..."]:
    denominator_safe = jnp.where(denominator == 0.0, 1.0, denominator)
    ratio = numerator / denominator_safe
    return jnp.where(denominator == 0.0, 0.0, ratio)


def interp1d(
    x: Float[Array, " m"],
    y: Float[Array, "..."],
    x_new: Float[Array, "..."],
    *,
    axis: int = -1,
    left: Optional[float] = None,
    right: Optional[float] = None,
    extrapolate: bool = False,
) -> Float[Array, "..."]:
    """
    Linear interpolation along one axis.

    Assumes that ``x`` is strictly increasing. This is validated by an
    *eager/debug* check in this thin Python wrapper: when ``x`` is a concrete
    (non-traced) array, a non-strictly-increasing grid raises ``ValueError``
    before the jitted core runs. Under ``jax.jit`` the grid is a tracer whose
    values are unknown at trace time, so the wrapper's check is skipped (a
    value-dependent ``raise`` cannot fire on a tracer); callers inside ``jit``
    are responsible for passing a valid monotonic grid. The numerics live in
    the jitted ``_interp1d_core``.
    """
    x = jnp.asarray(x)
    # Eager/debug guard: reject non-strictly-increasing grids when x is concrete.
    # Skipped under tracing (try_concrete_bool returns None inside a JAX trace).
    is_increasing = try_concrete_bool(jnp.all(jnp.diff(x) > 0))
    if is_increasing is False:
        raise ValueError(
            "interp1d requires x to be strictly increasing (x[i+1] > x[i])."
        )
    return _interp1d_core(
        x, y, x_new, axis=axis, left=left, right=right, extrapolate=extrapolate
    )


def cubic_hermite_interp(
    x: Float[Array, " m"],
    y: Float[Array, "..."],
    dydx: Float[Array, "..."],
    x_new: Float[Array, "..."],
    *,
    axis: int = -1,
    extrapolate: bool = False,
) -> Float[Array, "..."]:
    """Evaluate cubic Hermite interpolation with supplied node derivatives."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    dydx = jnp.asarray(dydx)
    axis = _validate_x_y(x, y, axis)
    if dydx.shape != y.shape:
        raise ValueError("cubic_hermite_interp derivatives must match y shape")
    return _cubic_hermite_interp_core(
        x,
        y,
        dydx,
        jnp.asarray(x_new),
        axis=axis,
        extrapolate=extrapolate,
    )


@partial(jax.jit, static_argnames=("axis", "extrapolate"))
def _cubic_hermite_interp_core(
    x: Float[Array, " m"],
    y: Float[Array, "..."],
    dydx: Float[Array, "..."],
    x_new: Float[Array, "..."],
    *,
    axis: int,
    extrapolate: bool,
) -> Float[Array, "..."]:
    y_moved = _move_axis_to_last(y, axis)
    dydx_moved = _move_axis_to_last(dydx, axis)

    idx = jnp.searchsorted(x, x_new, side="right") - 1
    idx = jnp.clip(idx, 0, x.shape[0] - 2)

    x0 = x[idx]
    x1 = x[idx + 1]
    h = x1 - x0
    t = (x_new - x0) / h

    y0 = jnp.take(y_moved, idx, axis=-1)
    y1 = jnp.take(y_moved, idx + 1, axis=-1)
    m0 = jnp.take(dydx_moved, idx, axis=-1)
    m1 = jnp.take(dydx_moved, idx + 1, axis=-1)

    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2

    y_interp = h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1

    if not extrapolate:
        below = x_new < x[0]
        above = x_new > x[-1]
        left_vals = jnp.take(y_moved, 0, axis=-1)
        right_vals = jnp.take(y_moved, x.shape[0] - 1, axis=-1)
        left_vals = jnp.reshape(left_vals, left_vals.shape + (1,) * x_new.ndim)
        right_vals = jnp.reshape(right_vals, right_vals.shape + (1,) * x_new.ndim)
        y_interp = jnp.where(below, left_vals, y_interp)
        y_interp = jnp.where(above, right_vals, y_interp)

    return y_interp


def pchip_slopes(
    x: Float[Array, " m"],
    y: Float[Array, "..."],
    *,
    axis: int = -1,
) -> Float[Array, "..."]:
    """Return monotone piecewise-cubic slopes using the PCHIP limiter."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    axis = _validate_x_y(x, y, axis)
    return _pchip_slopes_core(x, y, axis=axis)


@partial(jax.jit, static_argnames=("axis",))
def _pchip_slopes_core(
    x: Float[Array, " m"],
    y: Float[Array, "..."],
    *,
    axis: int,
) -> Float[Array, "..."]:
    y_moved = _move_axis_to_last(y, axis)
    h = jnp.diff(x)
    delta = jnp.diff(y_moved, axis=-1) / h

    if x.shape[0] == 2:
        slopes_moved = jnp.concatenate([delta, delta], axis=-1)
        return jnp.moveaxis(slopes_moved, -1, axis)

    d_prev = delta[..., :-1]
    d_next = delta[..., 1:]
    h_prev = h[:-1]
    h_next = h[1:]

    w1 = 2.0 * h_next + h_prev
    w2 = h_next + 2.0 * h_prev
    harmonic = _safe_div_zero(
        w1 + w2,
        _safe_div_zero(w1, d_prev) + _safe_div_zero(w2, d_next),
    )
    interior = jnp.where(d_prev * d_next > 0.0, harmonic, 0.0)

    left = _pchip_endpoint_slope(delta[..., 0], delta[..., 1], h[0], h[1])
    right = _pchip_endpoint_slope(delta[..., -1], delta[..., -2], h[-1], h[-2])

    slopes_moved = jnp.concatenate(
        [left[..., None], interior, right[..., None]],
        axis=-1,
    )
    return jnp.moveaxis(slopes_moved, -1, axis)


def _pchip_endpoint_slope(
    d0: Float[Array, "..."],
    d1: Float[Array, "..."],
    h0: Float[Array, ""],
    h1: Float[Array, ""],
) -> Float[Array, "..."]:
    raw = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    same_direction = jnp.sign(raw) == jnp.sign(d0)
    limited = jnp.where(same_direction, raw, 0.0)
    overshoots = (jnp.sign(d0) != jnp.sign(d1)) & (jnp.abs(limited) > 3.0 * jnp.abs(d0))
    return jnp.where(overshoots, 3.0 * d0, limited)


def monotone_cubic_interp(
    x: Float[Array, " m"],
    y: Float[Array, "..."],
    x_new: Float[Array, "..."],
    *,
    axis: int = -1,
    extrapolate: bool = False,
) -> Float[Array, "..."]:
    """Evaluate PCHIP-style monotone cubic interpolation."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    axis = _validate_x_y(x, y, axis)
    slopes = _pchip_slopes_core(x, y, axis=axis)
    return _cubic_hermite_interp_core(
        x,
        y,
        slopes,
        jnp.asarray(x_new),
        axis=axis,
        extrapolate=extrapolate,
    )


def natural_cubic_spline_coeffs(
    x: Float[Array, " m"],
    y: Float[Array, " m"],
) -> Tuple[
    Float[Array, " m_minus_1"],
    Float[Array, " m_minus_1"],
    Float[Array, " m_minus_1"],
    Float[Array, " m_minus_1"],
]:
    """Return natural cubic spline coefficients about each left knot."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    _validate_x_y(x, y, axis=0)
    return _natural_cubic_spline_coeffs_core(x, y)


@jax.jit
def _natural_cubic_spline_coeffs_core(
    x: Float[Array, " m"],
    y: Float[Array, " m"],
) -> Tuple[
    Float[Array, " m_minus_1"],
    Float[Array, " m_minus_1"],
    Float[Array, " m_minus_1"],
    Float[Array, " m_minus_1"],
]:
    h = jnp.diff(x)
    slopes = jnp.diff(y) / h
    n = x.shape[0]

    if n == 2:
        m = jnp.zeros_like(x)
    else:
        lower = jnp.concatenate([h[:-1], jnp.zeros(1, dtype=x.dtype)])
        diag = jnp.concatenate(
            [
                jnp.ones(1, dtype=x.dtype),
                2.0 * (h[:-1] + h[1:]),
                jnp.ones(1, dtype=x.dtype),
            ]
        )
        upper = jnp.concatenate([jnp.zeros(1, dtype=x.dtype), h[1:]])
        rhs = jnp.concatenate(
            [
                jnp.zeros(1, dtype=x.dtype),
                6.0 * (slopes[1:] - slopes[:-1]),
                jnp.zeros(1, dtype=x.dtype),
            ]
        )
        system = jnp.diag(diag) + jnp.diag(lower, k=-1) + jnp.diag(upper, k=1)
        m = jnp.linalg.solve(system, rhs)

    a = y[:-1]
    b = slopes - h * (2.0 * m[:-1] + m[1:]) / 6.0
    c = 0.5 * m[:-1]
    d = (m[1:] - m[:-1]) / (6.0 * h)
    return a, b, c, d


def eval_cubic_spline(
    x_knots: Float[Array, " m"],
    coeffs: Tuple[
        Float[Array, " m_minus_1"],
        Float[Array, " m_minus_1"],
        Float[Array, " m_minus_1"],
        Float[Array, " m_minus_1"],
    ],
    x_query: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Evaluate per-interval cubic spline coefficients at query points."""
    x_knots = jnp.asarray(x_knots)
    is_increasing = try_concrete_bool(jnp.all(jnp.diff(x_knots) > 0))
    if is_increasing is False:
        raise ValueError("eval_cubic_spline requires x_knots to be strictly increasing")
    return _eval_cubic_spline_core(x_knots, coeffs, jnp.asarray(x_query))


@jax.jit
def _eval_cubic_spline_core(
    x_knots: Float[Array, " m"],
    coeffs: Tuple[
        Float[Array, " m_minus_1"],
        Float[Array, " m_minus_1"],
        Float[Array, " m_minus_1"],
        Float[Array, " m_minus_1"],
    ],
    x_query: Float[Array, "..."],
) -> Float[Array, "..."]:
    a, b, c, d = coeffs
    x_eval = jnp.clip(x_query, x_knots[0], x_knots[-1])
    idx = jnp.searchsorted(x_knots, x_eval, side="right") - 1
    idx = jnp.clip(idx, 0, x_knots.shape[0] - 2)
    dx = x_eval - x_knots[idx]
    return a[idx] + dx * (b[idx] + dx * (c[idx] + dx * d[idx]))


@partial(jax.jit, static_argnames=("axis", "left", "right", "extrapolate"))
def _interp1d_core(
    x: Float[Array, " m"],
    y: Float[Array, "..."],
    x_new: Float[Array, "..."],
    *,
    axis: int = -1,
    left: Optional[float] = None,
    right: Optional[float] = None,
    extrapolate: bool = False,
) -> Float[Array, "..."]:
    """Jitted core for :func:`interp1d` (no input validation)."""
    y_moved = _move_axis_to_last(y, axis)
    x = jnp.asarray(x)
    x_new = jnp.asarray(x_new)

    M = x.shape[0]
    if M < 2:
        raise ValueError("interp1d requires at least two grid points")

    idx = jnp.searchsorted(x, x_new, side="right") - 1
    idx = jnp.clip(idx, 0, M - 2)

    x0 = x[idx]
    x1 = x[idx + 1]

    y0 = jnp.take(y_moved, idx, axis=-1)
    y1 = jnp.take(y_moved, idx + 1, axis=-1)

    denom = x1 - x0
    denom = jnp.where(denom == 0.0, 1.0, denom)
    t = (x_new - x0) / denom

    y_interp = (1.0 - t) * y0 + t * y1

    if not extrapolate:
        x_min = x[0]
        x_max = x[-1]
        below = x_new < x_min
        above = x_new > x_max

        if left is None:
            left_vals = jnp.take(y_moved, 0, axis=-1)
            y_interp = jnp.where(below, left_vals, y_interp)
        else:
            y_interp = jnp.where(below, left, y_interp)

        if right is None:
            right_vals = jnp.take(y_moved, M - 1, axis=-1)
            y_interp = jnp.where(above, right_vals, y_interp)
        else:
            y_interp = jnp.where(above, right, y_interp)

    return y_interp


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class TabulatedFunction1D:
    """
    Simple 1D tabulated function with linear interpolation.

    This wrapper is JAX-pytree compatible, so it can be captured in
    jitted functions or vmapped over if needed.
    """

    x: Float[Array, " m"]
    y: Float[Array, "..."]

    def __call__(self, x_new: Float[Array, "..."]) -> Float[Array, "..."]:
        return interp1d(self.x, self.y, x_new)

    def tree_flatten(
        self,
    ) -> Tuple[Tuple[Float[Array, " m"], Float[Array, "..."]], None]:
        return (self.x, self.y), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        x, y = children
        return cls(x=x, y=y)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MonotoneTabulatedFunction1D:
    """Shape-preserving 1D tabulated function using PCHIP-style slopes."""

    x: Float[Array, " m"]
    y: Float[Array, "..."]
    axis: int = -1

    def __call__(self, x_new: Float[Array, "..."]) -> Float[Array, "..."]:
        return monotone_cubic_interp(self.x, self.y, x_new, axis=self.axis)

    def slopes(self) -> Float[Array, "..."]:
        return pchip_slopes(self.x, self.y, axis=self.axis)

    def tree_flatten(
        self,
    ) -> Tuple[Tuple[Float[Array, " m"], Float[Array, "..."]], int]:
        return (self.x, self.y), self.axis

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        x, y = children
        return cls(x=x, y=y, axis=aux_data)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NaturalCubicSpline1D:
    """Natural cubic spline wrapper for 1D tabulated data."""

    x: Float[Array, " m"]
    y: Float[Array, " m"]

    def __call__(self, x_new: Float[Array, "..."]) -> Float[Array, "..."]:
        return eval_cubic_spline(self.x, self.coeffs(), x_new)

    def coeffs(
        self,
    ) -> Tuple[
        Float[Array, " m_minus_1"],
        Float[Array, " m_minus_1"],
        Float[Array, " m_minus_1"],
        Float[Array, " m_minus_1"],
    ]:
        return natural_cubic_spline_coeffs(self.x, self.y)

    def tree_flatten(
        self,
    ) -> Tuple[Tuple[Float[Array, " m"], Float[Array, " m"]], None]:
        return (self.x, self.y), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        x, y = children
        return cls(x=x, y=y)


__all__ = [
    "interp1d",
    "cubic_hermite_interp",
    "pchip_slopes",
    "monotone_cubic_interp",
    "natural_cubic_spline_coeffs",
    "eval_cubic_spline",
    "TabulatedFunction1D",
    "MonotoneTabulatedFunction1D",
    "NaturalCubicSpline1D",
]
