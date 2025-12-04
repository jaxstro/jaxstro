# src/jaxstro/numerics/interpolation.py
"""
1D interpolation utilities for JAX.

This module provides a simple, differentiable linear interpolator and a
small TabulatedFunction1D wrapper for table-based models (e.g. fits,
opacities, cooling curves).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from .types import Array


def _move_axis_to_last(x: Array, axis: int) -> Array:
    if axis < 0:
        axis = x.ndim + axis
    return jnp.moveaxis(x, axis, -1)


@jax.jit
def interp1d(
    x: Array,
    y: Array,
    x_new: Array,
    *,
    axis: int = -1,
    left: Optional[float] = None,
    right: Optional[float] = None,
    extrapolate: bool = False,
) -> Array:
    """
    Linear interpolation along one axis.

    Assumes that x is strictly increasing.
    """
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

    x: Array
    y: Array

    def __call__(self, x_new: Array) -> Array:
        return interp1d(self.x, self.y, x_new)

    def tree_flatten(self) -> Tuple[Tuple[Array, Array], None]:
        return (self.x, self.y), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        x, y = children
        return cls(x=x, y=y)


__all__ = ["interp1d", "TabulatedFunction1D"]
