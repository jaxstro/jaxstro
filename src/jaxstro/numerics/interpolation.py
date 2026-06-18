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


__all__ = ["interp1d", "TabulatedFunction1D"]
