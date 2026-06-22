# src/jaxstro/numerics/regular_grid.py
"""JAX-native regular-grid interpolation utilities."""

from __future__ import annotations

from functools import partial
from itertools import product

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .checks import try_concrete_bool

BoundaryPolicy = str


def regular_grid_interp(
    points: tuple[Float[Array, " n"], ...],
    values: Float[Array, "..."],
    xi: Float[Array, "... ndim"],
    *,
    boundary: BoundaryPolicy = "clamp",
    fill_value: float = jnp.nan,
) -> Float[Array, "..."]:
    """Interpolate values on a static-rank regular grid.

    Grid axes are the leading dimensions of ``values``. Any trailing dimensions
    are treated as vector-valued payload axes and are preserved in the result.
    Query coordinates live on the final axis of ``xi``.
    """
    axes = tuple(jnp.asarray(axis) for axis in points)
    values = jnp.asarray(values)
    xi = jnp.asarray(xi)
    _validate_regular_grid_inputs(axes, values, xi, boundary)
    _validate_reject_boundary(axes, xi, boundary)
    return _regular_grid_interp_core(
        axes,
        values,
        xi,
        boundary=boundary,
        fill_value=float(fill_value),
        ndim=len(axes),
    )


@partial(jax.jit, static_argnames=("boundary", "fill_value", "ndim"))
def _regular_grid_interp_core(
    axes: tuple[Float[Array, " n"], ...],
    values: Float[Array, "..."],
    xi: Float[Array, "... ndim"],
    *,
    boundary: BoundaryPolicy,
    fill_value: float,
    ndim: int,
) -> Float[Array, "..."]:
    query_shape = xi.shape[:-1]
    payload_shape = values.shape[ndim:]
    result = jnp.zeros(query_shape + payload_shape, dtype=values.dtype)
    outside = jnp.zeros(query_shape, dtype=bool)

    lower_indices: list[Float[Array, "..."]] = []
    fractions: list[Float[Array, "..."]] = []
    for dim in range(ndim):
        axis = axes[dim]
        coord = xi[..., dim]
        outside = outside | (coord < axis[0]) | (coord > axis[-1])
        coord_eval = jnp.clip(coord, axis[0], axis[-1])
        idx = jnp.searchsorted(axis, coord_eval, side="right") - 1
        idx = jnp.clip(idx, 0, axis.shape[0] - 2)
        x0 = axis[idx]
        x1 = axis[idx + 1]
        frac = (coord_eval - x0) / (x1 - x0)
        lower_indices.append(idx)
        fractions.append(frac)

    for corner in product((0, 1), repeat=ndim):
        weight = jnp.ones(query_shape, dtype=values.dtype)
        indices = []
        for dim, bit in enumerate(corner):
            frac = fractions[dim]
            weight = weight * jnp.where(bit == 1, frac, 1.0 - frac)
            indices.append(lower_indices[dim] + bit)
        corner_values = values[tuple(indices)]
        weight = jnp.reshape(weight, query_shape + (1,) * len(payload_shape))
        result = result + weight * corner_values

    if boundary == "fill":
        fill = jnp.asarray(fill_value, dtype=result.dtype)
        fill = jnp.broadcast_to(fill, payload_shape)
        fill = jnp.broadcast_to(fill, result.shape)
        result = jnp.where(
            outside[..., None] if payload_shape else outside, fill, result
        )

    return result


def bilinear_interp(
    x: Float[Array, " nx"],
    y: Float[Array, " ny"],
    values: Float[Array, "..."],
    x_new: Float[Array, "..."],
    y_new: Float[Array, "..."],
    *,
    boundary: BoundaryPolicy = "clamp",
    fill_value: float = jnp.nan,
) -> Float[Array, "..."]:
    """Bilinear interpolation convenience wrapper."""
    x_new, y_new = jnp.broadcast_arrays(jnp.asarray(x_new), jnp.asarray(y_new))
    xi = jnp.stack([x_new, y_new], axis=-1)
    return regular_grid_interp(
        (x, y),
        values,
        xi,
        boundary=boundary,
        fill_value=fill_value,
    )


def trilinear_interp(
    x: Float[Array, " nx"],
    y: Float[Array, " ny"],
    z: Float[Array, " nz"],
    values: Float[Array, "..."],
    x_new: Float[Array, "..."],
    y_new: Float[Array, "..."],
    z_new: Float[Array, "..."],
    *,
    boundary: BoundaryPolicy = "clamp",
    fill_value: float = jnp.nan,
) -> Float[Array, "..."]:
    """Trilinear interpolation convenience wrapper."""
    x_new, y_new, z_new = jnp.broadcast_arrays(
        jnp.asarray(x_new),
        jnp.asarray(y_new),
        jnp.asarray(z_new),
    )
    xi = jnp.stack([x_new, y_new, z_new], axis=-1)
    return regular_grid_interp(
        (x, y, z),
        values,
        xi,
        boundary=boundary,
        fill_value=fill_value,
    )


def _validate_regular_grid_inputs(
    axes: tuple[Float[Array, " n"], ...],
    values: Float[Array, "..."],
    xi: Float[Array, "... ndim"],
    boundary: BoundaryPolicy,
) -> None:
    if boundary not in {"clamp", "fill", "reject"}:
        raise ValueError(
            "regular_grid_interp boundary must be 'clamp', 'fill', or 'reject'"
        )
    if not axes:
        raise ValueError("regular_grid_interp requires at least one grid axis")
    if xi.ndim == 0 or xi.shape[-1] != len(axes):
        raise ValueError("regular_grid_interp xi final axis must match grid rank")
    if values.ndim < len(axes):
        raise ValueError(
            "regular_grid_interp values leading grid shape must match axes"
        )

    leading_shape = values.shape[: len(axes)]
    expected_shape = tuple(axis.shape[0] for axis in axes)
    if leading_shape != expected_shape:
        raise ValueError(
            "regular_grid_interp values leading grid shape must match axes; "
            f"expected {expected_shape}, got {leading_shape}"
        )

    for axis in axes:
        if axis.ndim != 1:
            raise ValueError("regular_grid_interp axes must be 1D arrays")
        if axis.shape[0] < 2:
            raise ValueError("regular_grid_interp axes require at least two points")
        is_increasing = try_concrete_bool(jnp.all(jnp.diff(axis) > 0.0))
        if is_increasing is False:
            raise ValueError("regular_grid_interp axes must be strictly increasing")


def _validate_reject_boundary(
    axes: tuple[Float[Array, " n"], ...],
    xi: Float[Array, "... ndim"],
    boundary: BoundaryPolicy,
) -> None:
    if boundary != "reject":
        return
    outside = jnp.zeros(xi.shape[:-1], dtype=bool)
    for dim, axis in enumerate(axes):
        outside = outside | (xi[..., dim] < axis[0]) | (xi[..., dim] > axis[-1])
    has_outside = try_concrete_bool(jnp.any(outside))
    if has_outside is True:
        raise ValueError(
            "regular_grid_interp reject boundary received outside query points"
        )


__all__ = [
    "bilinear_interp",
    "regular_grid_interp",
    "trilinear_interp",
]
