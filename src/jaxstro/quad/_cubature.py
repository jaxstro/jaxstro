"""Local Genz-Malik rule construction and estimation."""

import itertools
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ._multidim import evaluate_multidim
from ._tensor import validate_b1_dimension
from .domains import Hyperrectangle
from .result import QuadStatus
from .tolerance import ErrorNorm, MaxNorm, tolerance_threshold
from .tolerance import error_norm as reduce_error_norm

RUNNING = -1


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


class _GenzMalikHostData(NamedTuple):
    """Cached host arrays and static orbit metadata with no JAX tracers."""

    points: np.ndarray
    high_weights: np.ndarray
    low_weights: np.ndarray
    lambda2_axis_indices: np.ndarray
    lambda4_axis_indices: np.ndarray
    dimension: int
    point_count: int
    center_slice: slice
    lambda2_axis_slice: slice
    lambda4_axis_slice: slice
    lambda4_pair_slice: slice
    lambda5_corner_slice: slice


class LocalCubatureEstimate(NamedTuple):
    """Higher rule, embedded error, split evidence, and finiteness."""

    value: Array
    error: Array
    axis_difference: Array
    nonfinite: Array


class CubatureCapacity(NamedTuple):
    """Static region-store and scan bounds validated before materialization."""

    point_count: int
    store_capacity: int
    max_refinements: int
    evaluation_refinement_limit: int
    region_refinement_limit: int


class CubatureRegionEstimate(NamedTuple):
    """One aligned estimate row per normalized leaf region."""

    value: Array
    error: Array
    error_norm: Array
    split_axis: Array
    nonfinite: Array


class CubatureReplayEvidence(NamedTuple):
    """Private stopped leaf metadata reserved for the Phase B4 replay owner."""

    lower: Array
    upper: Array
    active: Array


class CubatureControllerResult(NamedTuple):
    """Private primal controller output plus stopped future replay metadata."""

    value: Array
    error: Array
    error_norm: Array
    tolerance: Array
    status: Array
    evaluations: Array
    refinements: Array
    active_regions: Array
    deepest_depth: Array
    evidence: CubatureReplayEvidence


class CubatureState(NamedTuple):
    """Fixed-capacity regional controller state."""

    lower: Array
    upper: Array
    local_value: Array
    local_error: Array
    local_error_norm: Array
    split_axis: Array
    active: Array
    depth: Array
    value: Array
    error: Array
    error_norm: Array
    tolerance: Array
    evaluations: Array
    refinements: Array
    active_regions: Array
    status: Array
    done: Array


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
def _genz_malik_data_cached(
    dimension: int,
    dtype_name: str,
) -> _GenzMalikHostData:
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

    return _GenzMalikHostData(
        points=points,
        high_weights=high_weights,
        low_weights=low_weights,
        lambda2_axis_indices=np.arange(
            lambda2_axis_slice.start,
            lambda2_axis_slice.stop,
            dtype=np.int32,
        ).reshape(dimension, 2),
        lambda4_axis_indices=np.arange(
            lambda4_axis_slice.start,
            lambda4_axis_slice.stop,
            dtype=np.int32,
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
    host = _genz_malik_data_cached(dimension, target_dtype.name)
    return GenzMalikData(
        points=jnp.asarray(host.points),
        high_weights=jnp.asarray(host.high_weights),
        low_weights=jnp.asarray(host.low_weights),
        lambda2_axis_indices=jnp.asarray(host.lambda2_axis_indices),
        lambda4_axis_indices=jnp.asarray(host.lambda4_axis_indices),
        dimension=host.dimension,
        point_count=host.point_count,
        center_slice=host.center_slice,
        lambda2_axis_slice=host.lambda2_axis_slice,
        lambda4_axis_slice=host.lambda4_axis_slice,
        lambda4_pair_slice=host.lambda4_pair_slice,
        lambda5_corner_slice=host.lambda5_corner_slice,
    )


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


def genz_malik_point_count(dimension: int) -> int:
    """Return the closed-form local-rule cost without constructing the rule."""
    validate_b1_dimension(dimension)
    return 2**dimension + 2 * dimension**2 + 2 * dimension + 1


def validate_cubature_capacity(
    *,
    dimension: int,
    max_evaluations: int,
    max_regions: int | None,
) -> CubatureCapacity:
    """Validate static capacities before rule or payload materialization."""
    validate_b1_dimension(dimension)
    if (
        not isinstance(max_evaluations, int)
        or isinstance(max_evaluations, bool)
        or max_evaluations <= 0
    ):
        raise ValueError("max_evaluations must be a positive integer")
    if (
        not isinstance(max_regions, int)
        or isinstance(max_regions, bool)
        or max_regions <= 0
    ):
        raise ValueError("max_regions must be a positive integer")

    point_count = genz_malik_point_count(dimension)
    if max_evaluations < point_count:
        raise ValueError(
            f"initial Genz-Malik rule requires {point_count} evaluations, "
            f"exceeding max_evaluations={max_evaluations}"
        )

    evaluation_refinements = (max_evaluations - point_count) // (2 * point_count)
    region_refinements = max_regions - 1
    max_refinements = min(evaluation_refinements, region_refinements)
    store_capacity = max_refinements + 1
    reachable_evaluations = point_count * (1 + 2 * max_refinements)
    max_int32 = np.iinfo(np.int32).max
    if store_capacity > max_int32 or reachable_evaluations > max_int32:
        raise ValueError(
            "reachable cubature work exceeds JAX int32 indexing: "
            f"store_capacity={store_capacity}, "
            f"evaluations={reachable_evaluations}"
        )

    # Limits are clipped just above the reachable scan so even enormous
    # declarations remain representable as JAX scalar comparisons.
    unreachable = max_refinements + 1
    return CubatureCapacity(
        point_count=point_count,
        store_capacity=store_capacity,
        max_refinements=max_refinements,
        evaluation_refinement_limit=min(evaluation_refinements, unreachable),
        region_refinement_limit=min(region_refinements, unreachable),
    )


def select_region(active: Array, local_error_norm: Array) -> Array:
    """Select the largest active local error, with lowest-region ties."""
    active = jnp.asarray(active)
    local_error_norm = jnp.asarray(local_error_norm)
    if (
        active.ndim != 1
        or local_error_norm.ndim != 1
        or active.shape != local_error_norm.shape
        or active.shape[0] == 0
    ):
        raise ValueError("active and local_error_norm must be aligned nonempty vectors")
    return jnp.argmax(jnp.where(active, local_error_norm, -jnp.inf))


def cubature_termination_status(
    *,
    nonfinite,
    converged,
    midpoint_collapsed,
    has_evaluation_capacity,
    has_region_capacity,
) -> Array:
    """Apply the complete cubature status precedence without heuristics."""
    running = jnp.asarray(RUNNING, dtype=jnp.int32)
    status = jnp.where(
        has_region_capacity,
        running,
        jnp.asarray(QuadStatus.MAX_REGIONS, dtype=jnp.int32),
    )
    status = jnp.where(
        has_evaluation_capacity,
        status,
        jnp.asarray(QuadStatus.MAX_EVALUATIONS, dtype=jnp.int32),
    )
    status = jnp.where(
        midpoint_collapsed,
        jnp.asarray(QuadStatus.ROUNDOFF_LIMITED, dtype=jnp.int32),
        status,
    )
    status = jnp.where(
        converged,
        jnp.asarray(QuadStatus.CONVERGED, dtype=jnp.int32),
        status,
    )
    return jnp.where(
        nonfinite,
        jnp.asarray(QuadStatus.NONFINITE_INTEGRAND, dtype=jnp.int32),
        status,
    )


def _payload_factors(values: Array, factors: Array) -> Array:
    shape = (factors.shape[0],) + (1,) * (values.ndim - 1)
    return values * factors.reshape(shape)


def _active_leaf_sum(values: Array, active: Array) -> Array:
    """Reduce only active leaf rows in the store's deterministic row order."""
    values = jnp.asarray(values)
    active = jnp.asarray(active)
    mask_shape = (active.shape[0],) + (1,) * (values.ndim - 1)
    masked = jnp.where(
        active.reshape(mask_shape),
        values,
        jnp.zeros_like(values),
    )
    return jnp.sum(masked, axis=0)


def evaluate_cubature_regions(
    fun,
    domain: Hyperrectangle,
    lower: Array,
    upper: Array,
    *,
    args,
    measure,
    error_norm: ErrorNorm,
    data: GenzMalikData,
) -> CubatureRegionEstimate:
    """Evaluate one or more normalized leaves in one atomic point batch."""
    lower = jnp.asarray(lower, dtype=data.points.dtype)
    upper = jnp.asarray(upper, dtype=data.points.dtype)
    if (
        lower.ndim != 2
        or upper.shape != lower.shape
        or lower.shape[1] != data.dimension
        or lower.shape[0] == 0
    ):
        raise ValueError(
            "cubature region bounds must have shape (region_count, dimension)"
        )
    region_count = lower.shape[0]
    width = upper - lower
    reference = lower[:, None, :] + width[:, None, :] * data.points[None, :, :]
    flattened = reference.reshape((region_count * data.point_count, data.dimension))
    evaluated = evaluate_multidim(
        fun,
        domain,
        flattened,
        args=args,
        measure=measure,
    )
    weighted = _payload_factors(evaluated.values, evaluated.weights)
    region_jacobian = jnp.prod(jnp.abs(width), axis=-1)
    region_index = jnp.repeat(
        jnp.arange(region_count, dtype=jnp.int32),
        data.point_count,
    )
    weighted = _payload_factors(weighted, region_jacobian[region_index])
    region_values = weighted.reshape(
        (region_count, data.point_count) + weighted.shape[1:]
    )
    estimates = jax.vmap(
        lambda values: genz_malik_estimate(
            values,
            data,
            error_norm=error_norm,
        )
    )(region_values)
    local_error_norm = jax.vmap(lambda value: reduce_error_norm(value, error_norm))(
        estimates.error
    )
    split_axis = jax.vmap(select_split_axis)(estimates.axis_difference)
    nonfinite = (
        estimates.nonfinite
        | ~jnp.isfinite(local_error_norm)
        | ~jnp.asarray(evaluated.valid)
    )
    return CubatureRegionEstimate(
        value=estimates.value,
        error=estimates.error,
        error_norm=local_error_norm,
        split_axis=split_axis.astype(jnp.int32),
        nonfinite=nonfinite,
    )


def _selected_midpoint(state: CubatureState) -> tuple[Array, Array, Array]:
    region = select_region(state.active, state.local_error_norm)
    axis = state.split_axis[region]
    lower = state.lower[region, axis]
    upper = state.upper[region, axis]
    midpoint = jnp.asarray(0.5, dtype=state.lower.dtype) * (lower + upper)
    collapsed = (midpoint == lower) | (midpoint == upper)
    return region, midpoint, collapsed


def _state_status(
    state: CubatureState,
    capacity: CubatureCapacity,
    *,
    nonfinite,
) -> Array:
    _region, _midpoint, midpoint_collapsed = _selected_midpoint(state)
    return cubature_termination_status(
        nonfinite=nonfinite,
        converged=state.error_norm <= state.tolerance,
        midpoint_collapsed=midpoint_collapsed,
        has_evaluation_capacity=(
            state.refinements < capacity.evaluation_refinement_limit
        ),
        has_region_capacity=(state.refinements < capacity.region_refinement_limit),
    )


def _global_nonfinite(
    value: Array,
    error: Array,
    error_norm: Array,
    tolerance: Array,
) -> Array:
    return ~(
        jnp.all(jnp.isfinite(value))
        & jnp.all(jnp.isfinite(error))
        & jnp.isfinite(error_norm)
        & jnp.isfinite(tolerance)
    )


def cubature_controller(
    fun,
    domain: Hyperrectangle,
    *,
    args,
    measure,
    epsabs,
    epsrel,
    max_evaluations: int,
    max_regions: int,
    error_norm: ErrorNorm,
    zero: Array,
    data: GenzMalikData,
    capacity: CubatureCapacity,
) -> CubatureControllerResult:
    """Run the fixed-capacity h-adaptive Genz-Malik region scan."""
    del max_evaluations, max_regions
    dimension = data.dimension
    store_capacity = capacity.store_capacity
    value_dtype = jnp.result_type(zero, data.points)
    zero = jnp.asarray(zero, dtype=value_dtype)
    initial_lower = jnp.zeros((1, dimension), dtype=data.points.dtype)
    initial_upper = jnp.ones((1, dimension), dtype=data.points.dtype)
    initial = evaluate_cubature_regions(
        fun,
        domain,
        initial_lower,
        initial_upper,
        args=args,
        measure=measure,
        error_norm=error_norm,
        data=data,
    )
    value = initial.value[0]
    error = initial.error[0]
    global_error_norm = reduce_error_norm(error, error_norm)
    tolerance = tolerance_threshold(
        value,
        epsabs=epsabs,
        epsrel=epsrel,
        norm=error_norm,
    )
    lower = jnp.zeros((store_capacity, dimension), dtype=data.points.dtype)
    upper = jnp.zeros((store_capacity, dimension), dtype=data.points.dtype)
    upper = upper.at[0].set(1.0)
    local_value = jnp.zeros((store_capacity,) + zero.shape, dtype=value_dtype)
    local_value = local_value.at[0].set(value)
    error_dtype = jnp.result_type(jnp.real(zero), data.points)
    local_error = jnp.zeros((store_capacity,) + zero.shape, dtype=error_dtype)
    local_error = local_error.at[0].set(error)
    local_error_norm = jnp.zeros(
        (store_capacity,),
        dtype=global_error_norm.dtype,
    )
    local_error_norm = local_error_norm.at[0].set(initial.error_norm[0])
    split_axis = jnp.zeros((store_capacity,), dtype=jnp.int32)
    split_axis = split_axis.at[0].set(initial.split_axis[0])
    active = jnp.zeros((store_capacity,), dtype=jnp.bool_).at[0].set(True)
    depth = jnp.zeros((store_capacity,), dtype=jnp.int32)
    initial_nonfinite = initial.nonfinite[0] | _global_nonfinite(
        value,
        error,
        global_error_norm,
        tolerance,
    )
    state = CubatureState(
        lower=lower,
        upper=upper,
        local_value=local_value,
        local_error=local_error,
        local_error_norm=local_error_norm,
        split_axis=split_axis,
        active=active,
        depth=depth,
        value=value,
        error=error,
        error_norm=global_error_norm,
        tolerance=tolerance,
        evaluations=jnp.asarray(capacity.point_count, dtype=jnp.int32),
        refinements=jnp.asarray(0, dtype=jnp.int32),
        active_regions=jnp.asarray(1, dtype=jnp.int32),
        status=jnp.asarray(RUNNING, dtype=jnp.int32),
        done=jnp.asarray(False),
    )
    initial_status = _state_status(
        state,
        capacity,
        nonfinite=initial_nonfinite,
    )
    state = state._replace(
        status=initial_status,
        done=initial_status != RUNNING,
    )

    def split(operand: CubatureState) -> CubatureState:
        region, midpoint, _collapsed = _selected_midpoint(operand)
        axis = operand.split_axis[region]
        parent_lower = operand.lower[region]
        parent_upper = operand.upper[region]
        parent_depth = operand.depth[region]
        left_upper = parent_upper.at[axis].set(midpoint)
        right_lower = parent_lower.at[axis].set(midpoint)
        child_lower = jnp.stack((parent_lower, right_lower))
        child_upper = jnp.stack((left_upper, parent_upper))
        children = evaluate_cubature_regions(
            fun,
            domain,
            child_lower,
            child_upper,
            args=args,
            measure=measure,
            error_norm=error_norm,
            data=data,
        )
        append_index = operand.active_regions
        new_local_value = (
            operand.local_value.at[region]
            .set(children.value[0])
            .at[append_index]
            .set(children.value[1])
        )
        new_local_error = (
            operand.local_error.at[region]
            .set(children.error[0])
            .at[append_index]
            .set(children.error[1])
        )
        new_active = operand.active.at[append_index].set(True)
        new_value = _active_leaf_sum(new_local_value, new_active)
        new_error = _active_leaf_sum(new_local_error, new_active)
        new_error_norm = reduce_error_norm(new_error, error_norm)
        new_tolerance = tolerance_threshold(
            new_value,
            epsabs=epsabs,
            epsrel=epsrel,
            norm=error_norm,
        )
        child_depth = parent_depth + 1
        next_state = operand._replace(
            lower=operand.lower.at[region]
            .set(child_lower[0])
            .at[append_index]
            .set(child_lower[1]),
            upper=operand.upper.at[region]
            .set(child_upper[0])
            .at[append_index]
            .set(child_upper[1]),
            local_value=new_local_value,
            local_error=new_local_error,
            local_error_norm=operand.local_error_norm.at[region]
            .set(children.error_norm[0])
            .at[append_index]
            .set(children.error_norm[1]),
            split_axis=operand.split_axis.at[region]
            .set(children.split_axis[0])
            .at[append_index]
            .set(children.split_axis[1]),
            active=new_active,
            depth=operand.depth.at[region]
            .set(child_depth)
            .at[append_index]
            .set(child_depth),
            value=new_value,
            error=new_error,
            error_norm=new_error_norm,
            tolerance=new_tolerance,
            evaluations=operand.evaluations + 2 * capacity.point_count,
            refinements=operand.refinements + 1,
            active_regions=operand.active_regions + 1,
        )
        child_nonfinite = jnp.any(children.nonfinite) | _global_nonfinite(
            new_value,
            new_error,
            new_error_norm,
            new_tolerance,
        )
        new_status = _state_status(
            next_state,
            capacity,
            nonfinite=child_nonfinite,
        )
        return next_state._replace(
            status=new_status,
            done=new_status != RUNNING,
        )

    def scan_body(current: CubatureState, _unused):
        next_state = jax.lax.cond(
            current.done,
            lambda operand: operand,
            split,
            current,
        )
        return next_state, None

    state, _ = jax.lax.scan(
        scan_body,
        state,
        xs=None,
        length=capacity.max_refinements,
    )
    return CubatureControllerResult(
        value=state.value,
        error=state.error,
        error_norm=state.error_norm,
        tolerance=state.tolerance,
        status=state.status,
        evaluations=state.evaluations,
        refinements=state.refinements,
        active_regions=state.active_regions,
        deepest_depth=jnp.max(jnp.where(state.active, state.depth, 0)),
        evidence=CubatureReplayEvidence(
            lower=state.lower,
            upper=state.upper,
            active=state.active,
        ),
    )


__all__ = [
    "CubatureCapacity",
    "CubatureControllerResult",
    "CubatureRegionEstimate",
    "GenzMalikData",
    "LocalCubatureEstimate",
    "RUNNING",
    "cubature_controller",
    "cubature_termination_status",
    "evaluate_cubature_regions",
    "genz_malik_data",
    "genz_malik_estimate",
    "genz_malik_point_count",
    "select_region",
    "select_split_axis",
    "validate_cubature_capacity",
]
