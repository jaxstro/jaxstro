# src/jaxstro/numerics/splines.py
"""JAX-native 1D B-spline basis and evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .checks import try_concrete_bool


def open_uniform_knots(
    x_min: float,
    x_max: float,
    n_basis: int,
    *,
    degree: int = 3,
) -> Float[Array, " n_knots"]:
    """Return a clamped open-uniform knot vector.

    The returned vector has ``n_basis + degree + 1`` knots, with the first and
    last knot repeated ``degree + 1`` times. ``n_basis`` must be at least
    ``degree + 1`` for degree > 0 so the clamped spline has a valid active
    interval.
    """
    _validate_degree(degree)
    if n_basis < 1:
        raise ValueError("open_uniform_knots requires n_basis >= 1")
    if n_basis < degree + 1:
        raise ValueError("open_uniform_knots requires n_basis >= degree + 1")
    if x_max <= x_min:
        raise ValueError("open_uniform_knots requires x_max > x_min")

    boundary = degree + 1
    n_interior = n_basis - degree - 1
    left = jnp.full((boundary,), float(x_min), dtype=jnp.float64)
    right = jnp.full((boundary,), float(x_max), dtype=jnp.float64)
    if n_interior == 0:
        return jnp.concatenate([left, right])
    interior = jnp.linspace(float(x_min), float(x_max), n_interior + 2)[1:-1]
    return jnp.concatenate([left, interior, right])


def bspline_basis(
    knots: Float[Array, " n_knots"],
    x: Float[Array, "..."],
    *,
    degree: int = 3,
) -> Float[Array, "... n_basis"]:
    """Evaluate all 1D B-spline basis functions at ``x``.

    Inputs outside the active knot domain are clamped to the endpoint basis
    values. This mirrors the existing fail-closed interpolation posture, but it
    also means gradients with respect to ``x`` saturate outside the domain.
    """
    knots = jnp.asarray(knots)
    _validate_knots(knots, degree)
    return _bspline_basis_core(knots, jnp.asarray(x), degree=degree)


def bspline_design_matrix(
    knots: Float[Array, " n_knots"],
    x: Float[Array, " n_samples"],
    *,
    degree: int = 3,
) -> Float[Array, " n_samples n_basis"]:
    """Return the B-spline design matrix for 1D sample coordinates."""
    x = jnp.asarray(x)
    if x.ndim != 1:
        raise ValueError("bspline_design_matrix requires 1D sample coordinates")
    return bspline_basis(knots, x, degree=degree)


@partial(jax.jit, static_argnames=("degree",))
def _bspline_basis_core(
    knots: Float[Array, " n_knots"],
    x: Float[Array, "..."],
    *,
    degree: int,
) -> Float[Array, "... n_basis"]:
    domain_left = knots[degree]
    domain_right = knots[-degree - 1]
    x_clamped = jnp.clip(x, domain_left, domain_right)
    x_expanded = x_clamped[..., None]

    left = knots[:-1]
    right = knots[1:]
    basis = jnp.where(
        (x_expanded >= left) & (x_expanded < right),
        1.0,
        0.0,
    )

    for order in range(1, degree + 1):
        count = knots.shape[0] - order - 1
        left_num = x_expanded - knots[:count]
        left_den = knots[order : order + count] - knots[:count]
        right_num = knots[order + 1 : order + 1 + count] - x_expanded
        right_den = knots[order + 1 : order + 1 + count] - knots[1 : count + 1]

        left_weight = _safe_ratio_or_zero(left_num, left_den)
        right_weight = _safe_ratio_or_zero(right_num, right_den)
        basis = (
            left_weight * basis[..., :count] + right_weight * basis[..., 1 : count + 1]
        )

    endpoint = x_clamped == domain_right
    endpoint_basis = jnp.zeros_like(basis).at[..., -1].set(1.0)
    return jnp.where(endpoint[..., None], endpoint_basis, basis)


def _safe_ratio_or_zero(
    numerator: Float[Array, "..."],
    denominator: Float[Array, "..."],
) -> Float[Array, "..."]:
    denominator_safe = jnp.where(denominator == 0.0, 1.0, denominator)
    ratio = numerator / denominator_safe
    return jnp.where(denominator == 0.0, 0.0, ratio)


def bspline_eval(
    knots: Float[Array, " n_knots"],
    coeffs: Float[Array, "..."],
    x: Float[Array, "..."],
    *,
    degree: int = 3,
    axis: int = -1,
) -> Float[Array, "..."]:
    """Evaluate a 1D B-spline with supplied knots and coefficients."""
    knots = jnp.asarray(knots)
    coeffs = jnp.asarray(coeffs)
    _validate_knots(knots, degree)
    axis = _normalize_axis(axis, coeffs.ndim)
    n_basis = knots.shape[0] - degree - 1
    if coeffs.shape[axis] != n_basis:
        raise ValueError(
            "bspline_eval coefficient axis length must equal "
            f"n_basis={n_basis}; got coefficient axis length {coeffs.shape[axis]}"
        )
    return _bspline_eval_core(
        knots,
        coeffs,
        jnp.asarray(x),
        degree=degree,
        axis=axis,
    )


@partial(jax.jit, static_argnames=("degree", "axis"))
def _bspline_eval_core(
    knots: Float[Array, " n_knots"],
    coeffs: Float[Array, "..."],
    x: Float[Array, "..."],
    *,
    degree: int,
    axis: int,
) -> Float[Array, "..."]:
    coeffs_moved = jnp.moveaxis(coeffs, axis, -1)
    basis = _bspline_basis_core(knots, x, degree=degree)
    return jnp.tensordot(coeffs_moved, basis, axes=([-1], [-1]))


def bspline_eval_deboor(
    knots: Float[Array, " n_knots"],
    coeffs: Float[Array, "..."],
    x: Float[Array, "..."],
    *,
    degree: int = 3,
    axis: int = -1,
) -> Float[Array, "..."]:
    """Evaluate a 1D B-spline using the de Boor recursion."""
    knots = jnp.asarray(knots)
    coeffs = jnp.asarray(coeffs)
    _validate_knots(knots, degree)
    axis = _normalize_axis(axis, coeffs.ndim)
    n_basis = knots.shape[0] - degree - 1
    if coeffs.shape[axis] != n_basis:
        raise ValueError(
            "bspline_eval_deboor coefficient axis length must equal "
            f"n_basis={n_basis}; got coefficient axis length {coeffs.shape[axis]}"
        )
    return _bspline_eval_deboor_core(
        knots,
        coeffs,
        jnp.asarray(x),
        degree=degree,
        axis=axis,
    )


@partial(jax.jit, static_argnames=("degree", "axis"))
def _bspline_eval_deboor_core(
    knots: Float[Array, " n_knots"],
    coeffs: Float[Array, "..."],
    x: Float[Array, "..."],
    *,
    degree: int,
    axis: int,
) -> Float[Array, "..."]:
    coeffs_moved = jnp.moveaxis(coeffs, axis, -1)
    payload_shape = coeffs_moved.shape[:-1]
    n_basis = coeffs_moved.shape[-1]
    domain_left = knots[degree]
    domain_right = knots[-degree - 1]
    x_clamped = jnp.clip(x, domain_left, domain_right)
    x_flat = jnp.ravel(x_clamped)

    def eval_one(xi):
        raw_span = jnp.searchsorted(knots, xi, side="right") - 1
        span = jnp.clip(raw_span, degree, n_basis - 1)
        local_idx = span - degree + jnp.arange(degree + 1)
        d = jnp.take(coeffs_moved, local_idx, axis=-1)

        for r in range(1, degree + 1):
            for j in range(degree, r - 1, -1):
                i = span - degree + j
                den = knots[i + degree - r + 1] - knots[i]
                alpha = _safe_ratio_or_zero(xi - knots[i], den)
                updated = (1.0 - alpha) * d[..., j - 1] + alpha * d[..., j]
                d = d.at[..., j].set(updated)
        return d[..., degree]

    flat_values = jax.vmap(eval_one)(x_flat)
    values = jnp.reshape(flat_values, x.shape + payload_shape)
    if x.ndim == 0:
        return values
    source_axes = tuple(range(x.ndim))
    target_axes = tuple(range(len(payload_shape), len(payload_shape) + x.ndim))
    return jnp.moveaxis(values, source_axes, target_axes)


def bspline_derivative(
    knots: Float[Array, " n_knots"],
    coeffs: Float[Array, "..."],
    x: Float[Array, "..."],
    *,
    degree: int = 3,
    axis: int = -1,
) -> Float[Array, "..."]:
    """Evaluate ``dS/dx`` for a 1D B-spline with fixed knots and coefficients.

    Outside the active knot domain this returns zero, matching the derivative of
    the public clamped evaluator with respect to ``x``.
    """
    if degree <= 0:
        raise ValueError("bspline_derivative requires a positive degree")
    knots = jnp.asarray(knots)
    coeffs = jnp.asarray(coeffs)
    _validate_knots(knots, degree)
    axis = _normalize_axis(axis, coeffs.ndim)
    n_basis = knots.shape[0] - degree - 1
    if coeffs.shape[axis] != n_basis:
        raise ValueError(
            "bspline_derivative coefficient axis length must equal "
            f"n_basis={n_basis}; got coefficient axis length {coeffs.shape[axis]}"
        )
    return _bspline_derivative_core(
        knots,
        coeffs,
        jnp.asarray(x),
        degree=degree,
        axis=axis,
    )


@partial(jax.jit, static_argnames=("degree", "axis"))
def _bspline_derivative_core(
    knots: Float[Array, " n_knots"],
    coeffs: Float[Array, "..."],
    x: Float[Array, "..."],
    *,
    degree: int,
    axis: int,
) -> Float[Array, "..."]:
    coeffs_moved = jnp.moveaxis(coeffs, axis, -1)
    denom = (
        knots[degree + 1 : degree + coeffs_moved.shape[-1]]
        - knots[1 : coeffs_moved.shape[-1]]
    )
    diff = jnp.diff(coeffs_moved, axis=-1)
    derivative_coeffs = float(degree) * _safe_ratio_or_zero(diff, denom)
    derivative_values = _bspline_eval_core(
        knots[1:-1],
        derivative_coeffs,
        x,
        degree=degree - 1,
        axis=-1,
    )

    domain_left = knots[degree]
    domain_right = knots[-degree - 1]
    outside = (x < domain_left) | (x > domain_right)
    return jnp.where(outside, 0.0, derivative_values)


def bspline_antiderivative(
    knots: Float[Array, " n_knots"],
    coeffs: Float[Array, "..."],
    *,
    degree: int = 3,
    axis: int = -1,
    constant: float = 0.0,
) -> tuple[Float[Array, " n_knots_plus_2"], Float[Array, "..."]]:
    """Return knots and coefficients for the first antiderivative spline."""
    knots = jnp.asarray(knots)
    coeffs = jnp.asarray(coeffs)
    _validate_knots(knots, degree)
    axis = _normalize_axis(axis, coeffs.ndim)
    n_basis = knots.shape[0] - degree - 1
    if coeffs.shape[axis] != n_basis:
        raise ValueError(
            "bspline_antiderivative coefficient axis length must equal "
            f"n_basis={n_basis}; got coefficient axis length {coeffs.shape[axis]}"
        )
    new_knots = jnp.concatenate([knots[:1], knots, knots[-1:]])
    new_coeffs = _bspline_antiderivative_coeffs_core(
        knots,
        coeffs,
        degree=degree,
        axis=axis,
        constant=constant,
    )
    return new_knots, new_coeffs


@partial(jax.jit, static_argnames=("degree", "axis", "constant"))
def _bspline_antiderivative_coeffs_core(
    knots: Float[Array, " n_knots"],
    coeffs: Float[Array, "..."],
    *,
    degree: int,
    axis: int,
    constant: float,
) -> Float[Array, "..."]:
    coeffs_moved = jnp.moveaxis(coeffs, axis, -1)
    widths = knots[degree + 1 :] - knots[: coeffs_moved.shape[-1]]
    increments = coeffs_moved * widths / float(degree + 1)
    initial = jnp.full((*coeffs_moved.shape[:-1], 1), constant, dtype=coeffs.dtype)
    cumulative = jnp.cumsum(increments, axis=-1) + initial
    anti = jnp.concatenate([initial, cumulative], axis=-1)
    return jnp.moveaxis(anti, -1, axis)


def bspline_integral(
    knots: Float[Array, " n_knots"],
    coeffs: Float[Array, "..."],
    a: float | Float[Array, ""],
    b: float | Float[Array, ""],
    *,
    degree: int = 3,
    axis: int = -1,
) -> Float[Array, "..."]:
    """Evaluate the definite integral of a 1D B-spline from ``a`` to ``b``."""
    anti_knots, anti_coeffs = bspline_antiderivative(
        knots, coeffs, degree=degree, axis=axis
    )
    return bspline_eval(
        anti_knots,
        anti_coeffs,
        jnp.asarray(b),
        degree=degree + 1,
        axis=axis,
    ) - bspline_eval(
        anti_knots,
        anti_coeffs,
        jnp.asarray(a),
        degree=degree + 1,
        axis=axis,
    )


def _derivative_coeffs_once(
    knots: Float[Array, " n_knots"],
    coeffs: Float[Array, "..."],
    *,
    degree: int,
    axis: int,
) -> tuple[Float[Array, " n_knots_minus_2"], Float[Array, "..."]]:
    coeffs_moved = jnp.moveaxis(coeffs, axis, -1)
    denom = (
        knots[degree + 1 : degree + coeffs_moved.shape[-1]]
        - knots[1 : coeffs_moved.shape[-1]]
    )
    diff = jnp.diff(coeffs_moved, axis=-1)
    derivative_coeffs = float(degree) * _safe_ratio_or_zero(diff, denom)
    return knots[1:-1], jnp.moveaxis(derivative_coeffs, -1, axis)


def bspline_roughness_penalty(
    knots: Float[Array, " n_knots"],
    coeffs: Float[Array, "..."],
    *,
    degree: int = 3,
    axis: int = -1,
    derivative_order: int = 2,
    n_samples: int = 129,
) -> Float[Array, "..."]:
    """Approximate the integrated squared derivative penalty on the active domain."""
    if derivative_order < 0:
        raise ValueError("derivative_order must be nonnegative")
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    knots = jnp.asarray(knots)
    coeffs = jnp.asarray(coeffs)
    _validate_knots(knots, degree)
    axis = _normalize_axis(axis, coeffs.ndim)
    current_knots = knots
    current_coeffs = coeffs
    current_degree = degree
    for _ in range(derivative_order):
        if current_degree == 0:
            return jnp.zeros(coeffs.shape[:axis] + coeffs.shape[axis + 1 :])
        current_knots, current_coeffs = _derivative_coeffs_once(
            current_knots,
            current_coeffs,
            degree=current_degree,
            axis=axis,
        )
        current_degree -= 1
    x = jnp.linspace(
        current_knots[current_degree],
        current_knots[-current_degree - 1],
        n_samples,
    )
    values = bspline_eval(
        current_knots,
        current_coeffs,
        x,
        degree=current_degree,
        axis=axis,
    )
    return jnp.trapezoid(values**2, x=x, axis=-1)


def fit_bspline_lstsq(
    knots: Float[Array, " n_knots"],
    x: Float[Array, " n_samples"],
    y: Float[Array, "..."],
    *,
    degree: int = 3,
    sample_axis: int = 0,
    rcond: float | None = None,
) -> Float[Array, "..."]:
    """Fit fixed-knot B-spline coefficients by linear least squares.

    The sample axis of ``y`` is replaced by the coefficient axis in the result.
    This is a policy-light solver: callers choose knots and provide sample
    values; no smoothing, adaptive knot placement, or extrapolation is applied.
    """
    knots = jnp.asarray(knots)
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    _validate_knots(knots, degree)
    if x.ndim != 1:
        raise ValueError("fit_bspline_lstsq requires 1D sample coordinates")
    sample_axis = _normalize_axis(sample_axis, y.ndim)
    if y.shape[sample_axis] != x.shape[0]:
        raise ValueError(
            "fit_bspline_lstsq sample axis length must equal "
            f"len(x)={x.shape[0]}; got sample axis length {y.shape[sample_axis]}"
        )
    return _fit_bspline_lstsq_core(
        knots,
        x,
        y,
        degree=degree,
        sample_axis=sample_axis,
        rcond=rcond,
    )


@partial(jax.jit, static_argnames=("degree", "sample_axis", "rcond"))
def _fit_bspline_lstsq_core(
    knots: Float[Array, " n_knots"],
    x: Float[Array, " n_samples"],
    y: Float[Array, "..."],
    *,
    degree: int,
    sample_axis: int,
    rcond: float | None,
) -> Float[Array, "..."]:
    design = _bspline_basis_core(knots, x, degree=degree)
    y_moved = jnp.moveaxis(y, sample_axis, 0)
    y_flat = jnp.reshape(y_moved, (x.shape[0], -1))
    coeffs_flat = jnp.linalg.lstsq(design, y_flat, rcond=rcond)[0]
    coeffs = jnp.reshape(coeffs_flat, (design.shape[-1], *y_moved.shape[1:]))
    return jnp.moveaxis(coeffs, 0, sample_axis)


def adaptive_open_uniform_knots(
    x: Float[Array, " n_samples"],
    n_basis: int,
    *,
    degree: int = 3,
) -> Float[Array, " n_knots"]:
    """Return clamped knots whose interior locations follow sample quantiles."""
    _validate_degree(degree)
    if n_basis < degree + 1:
        raise ValueError("adaptive_open_uniform_knots requires n_basis >= degree + 1")
    x = jnp.asarray(x)
    if x.ndim != 1:
        raise ValueError("adaptive_open_uniform_knots requires 1D sample coordinates")
    x_min = jnp.min(x)
    x_max = jnp.max(x)
    has_width = try_concrete_bool(x_min < x_max)
    if has_width is False:
        raise ValueError("adaptive_open_uniform_knots requires positive sample width")
    boundary = degree + 1
    n_interior = n_basis - degree - 1
    left = jnp.full((boundary,), x_min, dtype=x.dtype)
    right = jnp.full((boundary,), x_max, dtype=x.dtype)
    if n_interior == 0:
        return jnp.concatenate([left, right])
    quantiles = jnp.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    interior = jnp.quantile(x, quantiles)
    return jnp.concatenate([left, interior, right])


def tensor_product_design_matrix(
    *basis_matrices: Float[Array, " n_samples n_basis"],
) -> Float[Array, " n_samples n_tensor_basis"]:
    """Return a row-wise tensor-product design matrix from 1D basis matrices."""
    if len(basis_matrices) < 1:
        raise ValueError("tensor_product_design_matrix requires at least one basis")
    result = jnp.asarray(basis_matrices[0])
    if result.ndim != 2:
        raise ValueError("tensor_product_design_matrix inputs must be 2D")
    n_samples = result.shape[0]
    for basis in basis_matrices[1:]:
        basis = jnp.asarray(basis)
        if basis.ndim != 2:
            raise ValueError("tensor_product_design_matrix inputs must be 2D")
        if basis.shape[0] != n_samples:
            raise ValueError("tensor_product_design_matrix sample counts must match")
        result = (result[:, :, None] * basis[:, None, :]).reshape(
            n_samples, result.shape[1] * basis.shape[1]
        )
    return result


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BSpline1D:
    """PyTree wrapper for a 1D B-spline with fixed knots and coefficients."""

    knots: Float[Array, " n_knots"]
    coeffs: Float[Array, "..."]
    degree: int = 3
    axis: int = -1

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return bspline_eval(
            self.knots,
            self.coeffs,
            x,
            degree=self.degree,
            axis=self.axis,
        )

    def basis(self, x: Float[Array, "..."]) -> Float[Array, "... n_basis"]:
        return bspline_basis(self.knots, x, degree=self.degree)

    def derivative(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return bspline_derivative(
            self.knots,
            self.coeffs,
            x,
            degree=self.degree,
            axis=self.axis,
        )

    def tree_flatten(
        self,
    ) -> tuple[tuple[Float[Array, " n_knots"], Float[Array, "..."]], dict[str, int]]:
        return (self.knots, self.coeffs), {"degree": self.degree, "axis": self.axis}

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: dict[str, int],
        children: tuple[Float[Array, " n_knots"], Float[Array, "..."]],
    ) -> "BSpline1D":
        knots, coeffs = children
        return cls(
            knots=knots,
            coeffs=coeffs,
            degree=aux_data["degree"],
            axis=aux_data["axis"],
        )


def _validate_degree(degree: int) -> None:
    if not isinstance(degree, int) or isinstance(degree, bool) or degree < 0:
        raise ValueError("B-spline degree must be a nonnegative integer")


def _validate_knots(knots: Float[Array, " n_knots"], degree: int) -> None:
    _validate_degree(degree)
    if knots.ndim != 1:
        raise ValueError("B-spline knots must be a 1D array")
    if knots.shape[0] < degree + 2:
        raise ValueError("B-spline knots must contain at least degree + 2 entries")
    n_basis = knots.shape[0] - degree - 1
    if n_basis < 1:
        raise ValueError("B-spline knot vector implies no basis functions")

    is_nondecreasing = try_concrete_bool(jnp.all(jnp.diff(knots) >= 0.0))
    if is_nondecreasing is False:
        raise ValueError("B-spline knots must be nondecreasing")

    has_domain_width = try_concrete_bool(knots[degree] < knots[-degree - 1])
    if has_domain_width is False:
        raise ValueError("B-spline active knot domain must have positive width")


def _normalize_axis(axis: int, ndim: int) -> int:
    if ndim == 0:
        raise ValueError("bspline_eval coefficients must have a coefficient axis")
    if axis < 0:
        axis = ndim + axis
    if axis < 0 or axis >= ndim:
        raise ValueError("bspline_eval coefficient axis is out of bounds")
    return axis


__all__ = [
    "BSpline1D",
    "adaptive_open_uniform_knots",
    "bspline_antiderivative",
    "bspline_basis",
    "bspline_derivative",
    "bspline_design_matrix",
    "bspline_eval",
    "bspline_eval_deboor",
    "bspline_integral",
    "bspline_roughness_penalty",
    "fit_bspline_lstsq",
    "open_uniform_knots",
    "tensor_product_design_matrix",
]
