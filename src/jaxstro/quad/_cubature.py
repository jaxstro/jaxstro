"""Local Genz-Malik rule construction and estimation."""

import itertools
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ._tensor import validate_b1_dimension
from .tolerance import ErrorNorm, MaxNorm
from .tolerance import error_norm as reduce_error_norm


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GenzMalikData:
    """Points, aligned weights, and deterministic orbit metadata."""

    points: Array
    high_weights: Array
    low_weights: Array
    lambda2_axis_indices: Array
    lambda4_axis_indices: Array
    dimension: int
    point_count: int
    center_slice: slice
    lambda2_axis_slice: slice
    lambda4_axis_slice: slice
    lambda4_pair_slice: slice
    lambda5_corner_slice: slice

    def tree_flatten(self):
        children = (
            self.points,
            self.high_weights,
            self.low_weights,
            self.lambda2_axis_indices,
            self.lambda4_axis_indices,
        )
        metadata = (
            self.dimension,
            self.point_count,
            self.center_slice.start,
            self.center_slice.stop,
            self.lambda2_axis_slice.start,
            self.lambda2_axis_slice.stop,
            self.lambda4_axis_slice.start,
            self.lambda4_axis_slice.stop,
            self.lambda4_pair_slice.start,
            self.lambda4_pair_slice.stop,
            self.lambda5_corner_slice.start,
            self.lambda5_corner_slice.stop,
        )
        return children, metadata

    @classmethod
    def tree_unflatten(cls, metadata, children):
        (
            dimension,
            point_count,
            center_start,
            center_stop,
            lambda2_start,
            lambda2_stop,
            lambda4_axis_start,
            lambda4_axis_stop,
            lambda4_pair_start,
            lambda4_pair_stop,
            lambda5_start,
            lambda5_stop,
        ) = metadata
        (
            points,
            high_weights,
            low_weights,
            lambda2_axis_indices,
            lambda4_axis_indices,
        ) = children
        return cls(
            points=points,
            high_weights=high_weights,
            low_weights=low_weights,
            lambda2_axis_indices=lambda2_axis_indices,
            lambda4_axis_indices=lambda4_axis_indices,
            dimension=dimension,
            point_count=point_count,
            center_slice=slice(center_start, center_stop),
            lambda2_axis_slice=slice(lambda2_start, lambda2_stop),
            lambda4_axis_slice=slice(lambda4_axis_start, lambda4_axis_stop),
            lambda4_pair_slice=slice(lambda4_pair_start, lambda4_pair_stop),
            lambda5_corner_slice=slice(lambda5_start, lambda5_stop),
        )


class LocalCubatureEstimate(NamedTuple):
    """Higher rule, embedded error, split evidence, and finiteness."""

    value: Array
    error: Array
    axis_difference: Array
    nonfinite: Array


def _real_floating_dtype(dtype) -> np.dtype:
    normalized = np.dtype(jnp.dtype(dtype).name)
    if not np.issubdtype(normalized, np.floating):
        raise TypeError("Genz-Malik rule construction requires a real floating dtype")
    return normalized


def _target_scalar(value: float | int, dtype: np.dtype) -> float:
    return cast(float, dtype.type(value))


def _target_radius(
    numerator: int,
    denominator: int,
    dtype: np.dtype,
) -> float:
    ratio = _target_scalar(numerator, dtype) / _target_scalar(denominator, dtype)
    return cast(float, np.sqrt(ratio, dtype=dtype))


def _map_reference_coordinate(
    sign: int,
    radius: float,
    dtype: np.dtype,
) -> float:
    center = _target_scalar(0.5, dtype)
    half = _target_scalar(0.5, dtype)
    return cast(
        float,
        center + half * _target_scalar(sign, dtype) * radius,
    )


def _axis_orbit(
    dimension: int,
    radius: float,
    dtype: np.dtype,
) -> list[np.ndarray]:
    center = _target_scalar(0.5, dtype)
    orbit = []
    for axis in range(dimension):
        for sign in (-1, 1):
            point = np.full(dimension, center, dtype=dtype)
            point[axis] = _map_reference_coordinate(sign, radius, dtype)
            orbit.append(point)
    return orbit


def _pair_orbit(
    dimension: int,
    radius: float,
    dtype: np.dtype,
) -> list[np.ndarray]:
    center = _target_scalar(0.5, dtype)
    orbit = []
    for first, second in itertools.combinations(range(dimension), 2):
        for first_sign, second_sign in itertools.product((-1, 1), repeat=2):
            point = np.full(dimension, center, dtype=dtype)
            point[first] = _map_reference_coordinate(first_sign, radius, dtype)
            point[second] = _map_reference_coordinate(second_sign, radius, dtype)
            orbit.append(point)
    return orbit


def _corner_orbit(
    dimension: int,
    radius: float,
    dtype: np.dtype,
) -> list[np.ndarray]:
    center = _target_scalar(0.5, dtype)
    half = _target_scalar(0.5, dtype)
    return [
        np.asarray(
            center + half * radius * np.asarray(signs, dtype=dtype),
            dtype=dtype,
        )
        for signs in itertools.product((-1, 1), repeat=dimension)
    ]


def _repeat_weight(
    weight: float,
    count: int,
    dtype: np.dtype,
) -> np.ndarray:
    return np.full(count, weight, dtype=dtype)


@lru_cache(maxsize=None)
def _genz_malik_data_cached(dimension: int, dtype_name: str) -> GenzMalikData:
    dtype = np.dtype(dtype_name)

    def scalar(value: float | int) -> float:
        return _target_scalar(value, dtype)

    dimension_value = scalar(dimension)

    lambda2 = _target_radius(9, 70, dtype)
    lambda4 = _target_radius(9, 10, dtype)
    lambda5 = _target_radius(9, 19, dtype)

    center_points = [np.full(dimension, scalar(0.5), dtype=dtype)]
    lambda2_axis_points = _axis_orbit(dimension, lambda2, dtype)
    lambda4_axis_points = _axis_orbit(dimension, lambda4, dtype)
    lambda4_pair_points = _pair_orbit(dimension, lambda4, dtype)
    lambda5_corner_points = _corner_orbit(dimension, lambda5, dtype)

    axis_count = 2 * dimension
    pair_count = 2 * dimension * (dimension - 1)
    corner_count = 2**dimension
    point_count = 1 + 2 * axis_count + pair_count + corner_count

    center_slice = slice(0, 1)
    lambda2_axis_slice = slice(center_slice.stop, center_slice.stop + axis_count)
    lambda4_axis_slice = slice(
        lambda2_axis_slice.stop,
        lambda2_axis_slice.stop + axis_count,
    )
    lambda4_pair_slice = slice(
        lambda4_axis_slice.stop,
        lambda4_axis_slice.stop + pair_count,
    )
    lambda5_corner_slice = slice(lambda4_pair_slice.stop, point_count)

    points = np.stack(
        (
            *center_points,
            *lambda2_axis_points,
            *lambda4_axis_points,
            *lambda4_pair_points,
            *lambda5_corner_points,
        ),
        axis=0,
    )

    high_center = (
        scalar(12824)
        - scalar(9120) * dimension_value
        + scalar(400) * dimension_value ** scalar(2)
    ) / scalar(19683)
    high_lambda2_axis = scalar(980) / scalar(6561)
    high_lambda4_axis = (scalar(1820) - scalar(400) * dimension_value) / scalar(19683)
    high_lambda4_pair = scalar(200) / scalar(19683)
    high_lambda5_corner = scalar(6859) / (scalar(19683) * scalar(2) ** dimension_value)
    high_weights = np.concatenate(
        (
            _repeat_weight(high_center, 1, dtype),
            _repeat_weight(high_lambda2_axis, axis_count, dtype),
            _repeat_weight(high_lambda4_axis, axis_count, dtype),
            _repeat_weight(high_lambda4_pair, pair_count, dtype),
            _repeat_weight(high_lambda5_corner, corner_count, dtype),
        )
    )

    low_center = (
        scalar(729)
        - scalar(950) * dimension_value
        + scalar(50) * dimension_value ** scalar(2)
    ) / scalar(729)
    low_lambda2_axis = scalar(245) / scalar(486)
    low_lambda4_axis = (scalar(265) - scalar(100) * dimension_value) / scalar(1458)
    low_lambda4_pair = scalar(25) / scalar(729)
    low_weights = np.concatenate(
        (
            _repeat_weight(low_center, 1, dtype),
            _repeat_weight(low_lambda2_axis, axis_count, dtype),
            _repeat_weight(low_lambda4_axis, axis_count, dtype),
            _repeat_weight(low_lambda4_pair, pair_count, dtype),
            np.zeros(corner_count, dtype=dtype),
        )
    )

    return GenzMalikData(
        points=jnp.asarray(points),
        high_weights=jnp.asarray(high_weights),
        low_weights=jnp.asarray(low_weights),
        lambda2_axis_indices=jnp.arange(
            lambda2_axis_slice.start,
            lambda2_axis_slice.stop,
            dtype=jnp.int32,
        ).reshape(dimension, 2),
        lambda4_axis_indices=jnp.arange(
            lambda4_axis_slice.start,
            lambda4_axis_slice.stop,
            dtype=jnp.int32,
        ).reshape(dimension, 2),
        dimension=dimension,
        point_count=point_count,
        center_slice=center_slice,
        lambda2_axis_slice=lambda2_axis_slice,
        lambda4_axis_slice=lambda4_axis_slice,
        lambda4_pair_slice=lambda4_pair_slice,
        lambda5_corner_slice=lambda5_corner_slice,
    )


def genz_malik_data(dimension: int, dtype) -> GenzMalikData:
    """Construct the normalized degree-7 and embedded degree-5 rule data."""
    validate_b1_dimension(dimension)
    target_dtype = _real_floating_dtype(dtype)
    return _genz_malik_data_cached(dimension, target_dtype.name)


def _weighted_payload_sum(values: Array, weights: Array) -> Array:
    if values.ndim < 1 or values.shape[0] != weights.shape[0]:
        raise ValueError("cubature values must have one leading point axis")
    broadcast_shape = (weights.shape[0],) + (1,) * (values.ndim - 1)
    return jnp.sum(values * weights.reshape(broadcast_shape), axis=0)


def _axis_fourth_differences(
    values: Array,
    data: GenzMalikData,
    error_norm: ErrorNorm,
) -> Array:
    center = values[data.center_slice.start]
    inner = values[data.lambda2_axis_indices]
    outer = values[data.lambda4_axis_indices]
    inner_second = jnp.sum(inner, axis=1) - 2 * center
    outer_second = jnp.sum(outer, axis=1) - 2 * center
    radius_squared_ratio = jnp.asarray(7.0, dtype=data.points.dtype)
    fourth_difference = outer_second - radius_squared_ratio * inner_second
    return jax.vmap(lambda value: reduce_error_norm(value, error_norm))(
        fourth_difference
    )


def genz_malik_estimate(
    values: Array,
    data: GenzMalikData,
    error_norm: ErrorNorm = MaxNorm(),
) -> LocalCubatureEstimate:
    """Return both local estimates and per-axis fourth-difference evidence."""
    values = jnp.asarray(values)
    high = _weighted_payload_sum(values, data.high_weights)
    low = _weighted_payload_sum(values, data.low_weights)
    error = jnp.abs(high - low)
    axis_difference = _axis_fourth_differences(values, data, error_norm)
    all_finite = (
        jnp.all(jnp.isfinite(values))
        & jnp.all(jnp.isfinite(high))
        & jnp.all(jnp.isfinite(low))
        & jnp.all(jnp.isfinite(error))
        & jnp.all(jnp.isfinite(axis_difference))
    )
    return LocalCubatureEstimate(
        value=high,
        error=error,
        axis_difference=axis_difference,
        nonfinite=~all_finite,
    )


def select_split_axis(axis_difference: Array) -> Array:
    """Select the largest fourth difference, with lowest-axis ties."""
    axis_difference = jnp.asarray(axis_difference)
    if axis_difference.ndim != 1 or axis_difference.shape[0] == 0:
        raise ValueError("axis_difference must be a nonempty one-dimensional array")
    return jnp.argmax(axis_difference)
