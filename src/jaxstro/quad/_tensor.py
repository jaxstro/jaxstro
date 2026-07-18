"""Fixed tensor-product rule construction on the unit hyperrectangle."""

import math
from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ._chebyshev import chebyshev_rule_data
from ._multidim import evaluate_multidim
from ._recurrence import gaussian_rule_data
from ._tanh_sinh import tanh_sinh_rule_data, tanh_sinh_rule_point_count
from .measures import LebesgueMeasure
from .result import QuadStatus
from .rules import (
    ClenshawCurtisRule,
    FejerIIRule,
    FejerIRule,
    GaussianRule,
    TanhSinhRule,
)
from .tolerance import ErrorNorm, tolerance_threshold
from .tolerance import error_norm as reduce_error_norm


class TensorRuleData(NamedTuple):
    points: Array
    weights: Array
    point_count: int


class TensorReplayEvidence(NamedTuple):
    """Stopped accepted-formula metadata reserved for the Phase B4 replay owner."""

    levels: Array
    active_node_ids: Array


class AdaptiveTensorControllerResult(NamedTuple):
    value: Array
    error: Array
    frontier_error: Array
    tolerance: Array
    status: Array
    evaluations: Array
    refinements: Array
    levels: Array
    evidence: TensorReplayEvidence


class AdaptiveTensorCapacity(NamedTuple):
    initial_evaluations: int
    max_level: int
    max_refinements: int


class AdaptiveTensorTables(NamedTuple):
    nodes: Array
    weights: Array
    canonical_ids: Array
    counts: Array
    initial_level: int
    max_level: int


class _TensorCache(NamedTuple):
    canonical_ids: Array
    points: Array
    values: Array
    active: Array
    node_count: Array
    evaluations: Array
    nonfinite: Array


class TensorState(NamedTuple):
    levels: Array
    value: Array
    error: Array
    frontier_error: Array
    directional_candidate_values: Array
    directional_error: Array
    directional_new_cost: Array
    canonical_node_table: Array
    cached_points: Array
    cached_values: Array
    cache_active: Array
    cache_count: Array
    evaluations: Array
    refinements: Array
    status: Array
    done: Array
    accepted_node_ids: Array


def validate_b1_dimension(dimension: int) -> None:
    if dimension < 2 or dimension > 8:
        raise ValueError("Phase B1 deterministic methods require dimension 2 through 8")


def _reduced_dyadic(index: int, level: int) -> tuple[int, int]:
    if index == 0:
        return 0, 0
    while level > 0 and index % 2 == 0:
        index //= 2
        level -= 1
    return index, level


def canonical_cc_axis_ids(level: int) -> Array:
    """Return exact reduced dyadic-angle identities for one CC level."""
    if isinstance(level, bool) or not isinstance(level, int) or level < 0:
        raise ValueError("Clenshaw-Curtis level must be a nonnegative integer")
    denominator = 1 << level
    return jnp.asarray(
        [_reduced_dyadic(index, level) for index in range(denominator + 1)],
        dtype=jnp.int32,
    )


def canonical_tensor_ids(levels: Array) -> Array:
    """Construct exact Cartesian canonical IDs from static host-side levels."""
    host_levels = np.asarray(levels)
    if host_levels.ndim != 1 or host_levels.size == 0:
        raise ValueError("tensor levels must be a nonempty one-dimensional array")
    if not np.issubdtype(host_levels.dtype, np.integer) or np.any(host_levels < 0):
        raise ValueError("tensor levels must contain nonnegative integers")
    axes = [canonical_cc_axis_ids(int(level)) for level in host_levels.tolist()]
    meshes = jnp.meshgrid(
        *(jnp.arange(axis.shape[0], dtype=jnp.int32) for axis in axes),
        indexing="ij",
    )
    return jnp.concatenate(
        [axes[axis][mesh.reshape(-1)] for axis, mesh in enumerate(meshes)],
        axis=-1,
    )


def choose_tensor_axis(directional_error: Array, new_cost: Array):
    """Choose maximum error-per-new-node profit with lowest-axis ties."""
    directional_error = jnp.asarray(directional_error)
    new_cost = jnp.asarray(new_cost)
    profit = directional_error / jnp.maximum(new_cost, 1)
    axis = jnp.argmax(profit)
    return axis, jnp.sum(directional_error)


def count_representable_new_nodes(base_points: Array, candidate_points: Array) -> Array:
    """Count candidate coordinate tuples absent from the representable base set."""
    base_points = jnp.asarray(base_points)
    candidate_points = jnp.asarray(candidate_points)
    if (
        base_points.ndim != 2
        or candidate_points.ndim != 2
        or base_points.shape[1:] != candidate_points.shape[1:]
    ):
        raise ValueError("base and candidate points must have matching coordinate axes")
    matches_base = jnp.all(
        candidate_points[:, None, :] == base_points[None, :, :],
        axis=-1,
    )
    matches_candidate = jnp.all(
        candidate_points[:, None, :] == candidate_points[None, :, :],
        axis=-1,
    )
    index = jnp.arange(candidate_points.shape[0])
    has_previous_duplicate = jnp.any(
        matches_candidate & (index[None, :] < index[:, None]),
        axis=1,
    )
    is_new = ~jnp.any(matches_base, axis=1) & ~has_previous_duplicate
    return jnp.sum(is_new, dtype=jnp.int32)


def _formula_point_count(levels: list[int] | tuple[int, ...]) -> int:
    return math.prod((1 << level) + 1 for level in levels)


@lru_cache(maxsize=None)
def _adaptive_tensor_capacity_cached(
    initial_level: int,
    dimension: int,
    max_evaluations: int,
) -> AdaptiveTensorCapacity:
    base_nodes = (1 << initial_level) + 1
    base_count = base_nodes**dimension
    one_axis_delta = (1 << initial_level) * base_nodes ** (dimension - 1)
    initial_evaluations = base_count + dimension * one_axis_delta
    if max_evaluations < initial_evaluations:
        raise ValueError(
            "initial adaptive tensor frontier requires "
            f"{initial_evaluations} evaluations, "
            f"exceeding max_evaluations={max_evaluations}"
        )

    max_level = initial_level + 1
    while ((1 << (max_level + 1)) + 1) * base_nodes ** (
        dimension - 1
    ) <= max_evaluations:
        max_level += 1

    levels = [initial_level] * dimension
    max_refinements = 0
    while True:
        candidates = []
        for axis in range(dimension):
            refined = levels.copy()
            refined[axis] += 1
            candidates.append((_formula_point_count(refined), axis, refined))
        count, _axis, refined = min(candidates)
        if count > max_evaluations:
            break
        levels = refined
        max_refinements += 1
    return AdaptiveTensorCapacity(
        initial_evaluations=initial_evaluations,
        max_level=max_level,
        max_refinements=max_refinements,
    )


def validate_adaptive_tensor_capacity(
    *,
    initial_level: int,
    dimension: int,
    max_evaluations: int,
) -> AdaptiveTensorCapacity:
    """Validate static workspace sizes before payload inference or tracing."""
    validate_b1_dimension(dimension)
    if (
        not isinstance(max_evaluations, int)
        or isinstance(max_evaluations, bool)
        or max_evaluations <= 0
    ):
        raise ValueError("adaptive max_evaluations must be a positive integer")
    return _adaptive_tensor_capacity_cached(
        initial_level,
        dimension,
        max_evaluations,
    )


def adaptive_tensor_tables(
    *,
    initial_level: int,
    max_level: int,
    dtype,
) -> AdaptiveTensorTables:
    """Construct static padded CC ladders used by the adaptive scan."""
    max_axis_nodes = (1 << max_level) + 1
    node_rows = []
    weight_rows = []
    id_rows = []
    counts = []
    for level in range(initial_level, max_level + 1):
        nodes, weights = _unit_rule_data(
            ClenshawCurtisRule((1 << level) + 1),
            dtype,
        )
        count = nodes.shape[0]
        pad = max_axis_nodes - count
        node_rows.append(jnp.pad(nodes, (0, pad)))
        weight_rows.append(jnp.pad(weights, (0, pad)))
        id_rows.append(
            jnp.pad(
                canonical_cc_axis_ids(level),
                ((0, pad), (0, 0)),
                constant_values=-1,
            )
        )
        counts.append(count)
    return AdaptiveTensorTables(
        nodes=jnp.stack(node_rows),
        weights=jnp.stack(weight_rows),
        canonical_ids=jnp.stack(id_rows),
        counts=jnp.asarray(counts, dtype=jnp.int32),
        initial_level=initial_level,
        max_level=max_level,
    )


def _candidate_formula(
    levels: Array,
    tables: AdaptiveTensorTables,
    max_evaluations: int,
) -> tuple[Array, Array, Array, Array]:
    level_index = jnp.clip(
        levels - tables.initial_level,
        0,
        tables.max_level - tables.initial_level,
    )
    counts = tables.counts[level_index]
    point_count = jnp.prod(counts)
    flat = jnp.arange(max_evaluations, dtype=jnp.int32)
    active = flat < point_count
    safe_flat = jnp.minimum(flat, jnp.maximum(point_count - 1, 0))
    point_columns = []
    weight_columns = []
    id_columns = []
    dimension = levels.shape[0]
    for axis in range(dimension):
        divisor = jnp.prod(counts[axis + 1 :], dtype=jnp.int32)
        index = (safe_flat // divisor) % counts[axis]
        point_columns.append(tables.nodes[level_index[axis], index])
        weight_columns.append(tables.weights[level_index[axis], index])
        id_columns.append(tables.canonical_ids[level_index[axis], index])
    points = jnp.stack(point_columns, axis=-1)
    weights = jnp.prod(jnp.stack(weight_columns, axis=-1), axis=-1)
    canonical_ids = jnp.concatenate(id_columns, axis=-1)
    points = jnp.where(active[:, None], points, 0.0)
    weights = jnp.where(active, weights, 0.0)
    canonical_ids = jnp.where(active[:, None], canonical_ids, -1)
    return points, weights, canonical_ids, active


def _masked_representable_new_count(
    base_points: Array,
    base_active: Array,
    candidate_points: Array,
    candidate_active: Array,
) -> Array:
    matches_base = (
        jnp.all(
            candidate_points[:, None, :] == base_points[None, :, :],
            axis=-1,
        )
        & base_active[None, :]
    )
    matches_candidate = jnp.all(
        candidate_points[:, None, :] == candidate_points[None, :, :],
        axis=-1,
    )
    index = jnp.arange(candidate_points.shape[0])
    has_previous_duplicate = jnp.any(
        matches_candidate
        & candidate_active[None, :]
        & (index[None, :] < index[:, None]),
        axis=1,
    )
    is_new = candidate_active & ~jnp.any(matches_base, axis=1) & ~has_previous_duplicate
    return jnp.sum(is_new, dtype=jnp.int32)


def _normalized_rules(method, dimension: int):
    validate_b1_dimension(dimension)
    rules = (
        (method.rules,) * dimension
        if not isinstance(method.rules, tuple)
        else method.rules
    )
    if len(rules) != dimension:
        raise ValueError("TensorProduct requires one rule or one rule per axis")
    return rules


def _unit_rule_data(rule, dtype):
    with jax.ensure_compile_time_eval():
        if isinstance(rule, GaussianRule):
            data = gaussian_rule_data(rule, LebesgueMeasure())
            exact_constant = True
        elif isinstance(rule, (ClenshawCurtisRule, FejerIRule, FejerIIRule)):
            data = chebyshev_rule_data(rule, dtype=dtype)
            exact_constant = True
        elif isinstance(rule, TanhSinhRule):
            data = tanh_sinh_rule_data(
                rule,
                dtype=dtype,
                open_unit_interval=True,
            )
            exact_constant = False
        else:
            raise TypeError(f"unsupported tensor rule: {type(rule).__name__}")
        half = jnp.asarray(0.5, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        nodes = jnp.asarray(data.nodes, dtype=dtype)
        weights = half * jnp.asarray(data.weights, dtype=dtype)
        if exact_constant:
            _guard_unit_mass(weights, rule)
            weights = weights / jnp.sum(weights)
        return half * (nodes + one), weights


def _guard_unit_mass(weights, rule) -> None:
    """Reject residuals larger than representable scaling and reduction error."""
    host_weights = np.asarray(weights)
    dtype = host_weights.dtype
    eps = np.finfo(dtype).eps
    term_count = host_weights.size
    accumulated = term_count * eps
    if accumulated >= 1.0:
        raise ValueError(
            f"{type(rule).__name__} order is too large for a unit-mass "
            f"roundoff guard in {dtype.name}"
        )
    reduction_factor = accumulated / (1.0 - accumulated)
    l1_mass = float(np.sum(np.abs(host_weights), dtype=dtype))
    roundoff_bound = (eps + reduction_factor + eps * reduction_factor) * l1_mass
    residual = abs(1.0 - float(np.sum(host_weights, dtype=dtype)))
    if residual > roundoff_bound:
        raise ValueError(
            f"{type(rule).__name__} unit-mass residual exceeds roundoff: "
            f"{residual} > {roundoff_bound}"
        )


def _rule_point_count(rule, dtype) -> int:
    if isinstance(rule, (GaussianRule, ClenshawCurtisRule, FejerIRule, FejerIIRule)):
        return rule.order
    if isinstance(rule, TanhSinhRule):
        return tanh_sinh_rule_point_count(
            rule,
            dtype=dtype,
            open_unit_interval=True,
        )
    raise TypeError(f"unsupported tensor rule: {type(rule).__name__}")


def tensor_point_count(method, dimension: int, dtype) -> int:
    """Return the exact product size without constructing the product mesh."""
    return math.prod(
        _rule_point_count(rule, dtype) for rule in _normalized_rules(method, dimension)
    )


def tensor_rule_data(method, dimension: int, dtype) -> TensorRuleData:
    """Construct a heterogeneous Cartesian-product rule on ``[0, 1]^d``."""
    axes = [
        _unit_rule_data(rule, dtype) for rule in _normalized_rules(method, dimension)
    ]
    point_count = math.prod(nodes.size for nodes, _weights in axes)
    point_meshes = jnp.meshgrid(
        *(nodes for nodes, _weights in axes),
        indexing="ij",
    )
    weight_meshes = jnp.meshgrid(
        *(weights for _nodes, weights in axes),
        indexing="ij",
    )
    points = jnp.stack([mesh.reshape(-1) for mesh in point_meshes], axis=-1)
    weights = jnp.prod(
        jnp.stack([mesh.reshape(-1) for mesh in weight_meshes], axis=-1),
        axis=-1,
    )
    return TensorRuleData(points, weights, point_count)


def _cache_from_state(state: TensorState) -> _TensorCache:
    return _TensorCache(
        canonical_ids=state.canonical_node_table,
        points=state.cached_points,
        values=state.cached_values,
        active=state.cache_active,
        node_count=state.cache_count,
        evaluations=state.evaluations,
        nonfinite=state.status == QuadStatus.NONFINITE_INTEGRAND,
    )


def _evaluate_formula_with_cache(
    fun,
    domain,
    *,
    args,
    measure,
    levels: Array,
    tables: AdaptiveTensorTables,
    cache: _TensorCache,
    zero: Array,
    max_evaluations: int,
) -> tuple[_TensorCache, Array, Array, Array]:
    points, weights, canonical_ids, active = _candidate_formula(
        levels,
        tables,
        max_evaluations,
    )
    zero = jnp.asarray(zero)

    def body(current: _TensorCache, row):
        point, canonical_id, row_active = row
        id_matches = current.active & jnp.all(
            current.canonical_ids == canonical_id[None, :],
            axis=-1,
        )
        point_matches = current.active & jnp.all(
            current.points == point[None, :],
            axis=-1,
        )
        matches = id_matches | point_matches
        known = jnp.any(matches)
        match_index = jnp.argmax(matches)

        def active_row(operand: _TensorCache):
            def reuse(existing: _TensorCache):
                return existing, existing.values[match_index]

            def evaluate_new(existing: _TensorCache):
                available = existing.node_count < max_evaluations

                def evaluate_and_store(store: _TensorCache):
                    evaluated = evaluate_multidim(
                        fun,
                        domain,
                        point[None, :],
                        args=args,
                        measure=measure,
                    )
                    contribution = evaluated.values[0] * evaluated.weights[0]
                    nonfinite = (
                        evaluated.nonfinite
                        | ~evaluated.valid
                        | ~jnp.all(jnp.isfinite(contribution))
                    )
                    index = store.node_count
                    updated = _TensorCache(
                        canonical_ids=store.canonical_ids.at[index].set(canonical_id),
                        points=store.points.at[index].set(point),
                        values=store.values.at[index].set(contribution),
                        active=store.active.at[index].set(True),
                        node_count=store.node_count + 1,
                        evaluations=store.evaluations + 1,
                        nonfinite=store.nonfinite | nonfinite,
                    )
                    return updated, contribution

                def exhausted(store: _TensorCache):
                    sentinel = jnp.full_like(zero, jnp.nan)
                    return store._replace(nonfinite=jnp.asarray(True)), sentinel

                return jax.lax.cond(
                    available,
                    evaluate_and_store,
                    exhausted,
                    existing,
                )

            return jax.lax.cond(known, reuse, evaluate_new, operand)

        def inactive_row(operand: _TensorCache):
            return operand, zero

        should_process = row_active & ~current.nonfinite
        return jax.lax.cond(
            should_process,
            active_row,
            inactive_row,
            current,
        )

    cache, row_values = jax.lax.scan(
        body,
        cache,
        (points, canonical_ids, active),
    )
    reshape = (max_evaluations,) + (1,) * zero.ndim
    value = jnp.sum(row_values * weights.reshape(reshape), axis=0)
    return cache, value, canonical_ids, active


def _directional_data(
    value: Array,
    candidate_values: Array,
    levels: Array,
    error_norm: ErrorNorm,
) -> tuple[Array, Array, Array, Array]:
    component_error = jnp.abs(candidate_values - value)
    directional_error = jnp.stack(
        [
            reduce_error_norm(component_error[axis], error_norm)
            for axis in range(levels.shape[0])
        ]
    )
    accepted_count = jnp.prod(jnp.left_shift(1, levels) + 1)
    directional_new_cost = jnp.stack(
        [
            jnp.prod(
                jnp.left_shift(
                    1,
                    levels + jax.nn.one_hot(axis, levels.shape[0], dtype=jnp.int32),
                )
                + 1
            )
            - accepted_count
            for axis in range(levels.shape[0])
        ]
    ).astype(jnp.int32)
    error = jnp.sum(component_error, axis=0)
    frontier_error = jnp.sum(directional_error)
    return error, frontier_error, directional_error, directional_new_cost


def _refresh_frontier(
    fun,
    domain,
    *,
    args,
    measure,
    levels: Array,
    value: Array,
    tables: AdaptiveTensorTables,
    cache: _TensorCache,
    zero: Array,
    max_evaluations: int,
    error_norm: ErrorNorm,
):
    candidate_values = []
    for axis in range(levels.shape[0]):
        candidate_levels = levels + jax.nn.one_hot(
            axis,
            levels.shape[0],
            dtype=jnp.int32,
        )
        cache, candidate_value, _ids, _active = _evaluate_formula_with_cache(
            fun,
            domain,
            args=args,
            measure=measure,
            levels=candidate_levels,
            tables=tables,
            cache=cache,
            zero=zero,
            max_evaluations=max_evaluations,
        )
        candidate_values.append(candidate_value)
    candidates = jnp.stack(candidate_values)
    error, frontier_error, directional_error, directional_new_cost = _directional_data(
        value, candidates, levels, error_norm
    )
    return (
        cache,
        candidates,
        error,
        frontier_error,
        directional_error,
        directional_new_cost,
    )


def _selected_representable_new_count(
    levels: Array,
    axis: Array,
    tables: AdaptiveTensorTables,
    max_evaluations: int,
) -> Array:
    base_points, _weights, _ids, base_active = _candidate_formula(
        levels,
        tables,
        max_evaluations,
    )
    refined_levels = levels + jax.nn.one_hot(
        axis,
        levels.shape[0],
        dtype=jnp.int32,
    )
    candidate_points, _weights, _ids, candidate_active = _candidate_formula(
        refined_levels,
        tables,
        max_evaluations,
    )
    return _masked_representable_new_count(
        base_points,
        base_active,
        candidate_points,
        candidate_active,
    )


def _frontier_missing_count(
    levels: Array,
    tables: AdaptiveTensorTables,
    cache: _TensorCache,
    max_evaluations: int,
) -> Array:
    out_of_range = jnp.any(levels + 1 > tables.max_level)
    dimension = levels.shape[0]
    empty_ids = jnp.full_like(cache.canonical_ids, -1)
    empty_points = jnp.zeros_like(cache.points)
    empty_active = jnp.zeros_like(cache.active)

    class MissingState(NamedTuple):
        canonical_ids: Array
        points: Array
        active: Array
        missing_count: Array

    missing = MissingState(
        canonical_ids=empty_ids,
        points=empty_points,
        active=empty_active,
        missing_count=jnp.asarray(0, dtype=jnp.int32),
    )

    def collect_formula(current: MissingState, candidate_levels: Array):
        points, _weights, canonical_ids, active = _candidate_formula(
            candidate_levels,
            tables,
            max_evaluations,
        )

        def body(store: MissingState, row):
            point, canonical_id, row_active = row
            cache_match = jnp.any(
                cache.active
                & (
                    jnp.all(
                        cache.canonical_ids == canonical_id[None, :],
                        axis=-1,
                    )
                    | jnp.all(cache.points == point[None, :], axis=-1)
                )
            )
            missing_match = jnp.any(
                store.active
                & (
                    jnp.all(
                        store.canonical_ids == canonical_id[None, :],
                        axis=-1,
                    )
                    | jnp.all(store.points == point[None, :], axis=-1)
                )
            )
            add = row_active & ~cache_match & ~missing_match
            index = jnp.minimum(store.missing_count, max_evaluations - 1)
            updated = MissingState(
                canonical_ids=store.canonical_ids.at[index].set(
                    jnp.where(add, canonical_id, store.canonical_ids[index])
                ),
                points=store.points.at[index].set(
                    jnp.where(add, point, store.points[index])
                ),
                active=store.active.at[index].set(store.active[index] | add),
                missing_count=store.missing_count + add.astype(jnp.int32),
            )
            return updated, None

        return jax.lax.scan(
            body,
            current,
            (points, canonical_ids, active),
        )[0]

    clipped_levels = jnp.minimum(levels, tables.max_level - 1)
    for axis in range(dimension):
        candidate_levels = clipped_levels + jax.nn.one_hot(
            axis,
            dimension,
            dtype=jnp.int32,
        )
        missing = collect_formula(missing, candidate_levels)
    return jnp.where(
        out_of_range,
        jnp.asarray(max_evaluations + 1, dtype=jnp.int32),
        missing.missing_count,
    )


def adaptive_tensor_controller(
    fun,
    domain,
    *,
    args,
    measure,
    initial_level: int,
    epsabs,
    epsrel,
    max_evaluations: int,
    error_norm: ErrorNorm,
    zero: Array,
    capacity: AdaptiveTensorCapacity,
) -> AdaptiveTensorControllerResult:
    """Run one fixed-capacity anisotropic CC frontier scan."""
    dtype = jnp.result_type(domain.lower, domain.upper, 0.0)
    tables = adaptive_tensor_tables(
        initial_level=initial_level,
        max_level=capacity.max_level,
        dtype=dtype,
    )
    dimension = domain.dimension
    levels = jnp.full((dimension,), initial_level, dtype=jnp.int32)
    value_dtype = jnp.result_type(zero, tables.nodes)
    zero = jnp.asarray(zero, dtype=value_dtype)
    cache = _TensorCache(
        canonical_ids=jnp.full(
            (max_evaluations, 2 * dimension),
            -1,
            dtype=jnp.int32,
        ),
        points=jnp.zeros((max_evaluations, dimension), dtype=dtype),
        values=jnp.zeros((max_evaluations,) + zero.shape, dtype=value_dtype),
        active=jnp.zeros((max_evaluations,), dtype=jnp.bool_),
        node_count=jnp.asarray(0, dtype=jnp.int32),
        evaluations=jnp.asarray(0, dtype=jnp.int32),
        nonfinite=jnp.asarray(False),
    )
    cache, value, accepted_ids, accepted_active = _evaluate_formula_with_cache(
        fun,
        domain,
        args=args,
        measure=measure,
        levels=levels,
        tables=tables,
        cache=cache,
        zero=zero,
        max_evaluations=max_evaluations,
    )
    (
        cache,
        candidates,
        error,
        frontier_error,
        directional_error,
        directional_new_cost,
    ) = _refresh_frontier(
        fun,
        domain,
        args=args,
        measure=measure,
        levels=levels,
        value=value,
        tables=tables,
        cache=cache,
        zero=zero,
        max_evaluations=max_evaluations,
        error_norm=error_norm,
    )
    tolerance = tolerance_threshold(
        value,
        epsabs=epsabs,
        epsrel=epsrel,
        norm=error_norm,
    )
    nonfinite = (
        cache.nonfinite
        | ~jnp.all(jnp.isfinite(value))
        | ~jnp.all(jnp.isfinite(error))
        | ~jnp.all(jnp.isfinite(directional_error))
        | ~jnp.isfinite(frontier_error)
        | ~jnp.isfinite(tolerance)
    )
    converged = frontier_error <= tolerance
    running = jnp.asarray(-1, dtype=jnp.int32)
    status = jnp.where(
        nonfinite,
        jnp.asarray(QuadStatus.NONFINITE_INTEGRAND, dtype=jnp.int32),
        jnp.where(
            converged,
            jnp.asarray(QuadStatus.CONVERGED, dtype=jnp.int32),
            running,
        ),
    )
    state = TensorState(
        levels=levels,
        value=value,
        error=error,
        frontier_error=frontier_error,
        directional_candidate_values=candidates,
        directional_error=directional_error,
        directional_new_cost=directional_new_cost,
        canonical_node_table=cache.canonical_ids,
        cached_points=cache.points,
        cached_values=cache.values,
        cache_active=cache.active,
        cache_count=cache.node_count,
        evaluations=cache.evaluations,
        refinements=jnp.asarray(0, dtype=jnp.int32),
        status=status,
        done=status != running,
        accepted_node_ids=jnp.where(accepted_active[:, None], accepted_ids, -1),
    )

    def running_step(current: TensorState) -> TensorState:
        axis, selected_frontier_error = choose_tensor_axis(
            current.directional_error,
            current.directional_new_cost,
        )
        current_tolerance = tolerance_threshold(
            current.value,
            epsabs=epsabs,
            epsrel=epsrel,
            norm=error_norm,
        )
        converged_now = selected_frontier_error <= current_tolerance
        cache_now = _cache_from_state(current)
        representable_new = _selected_representable_new_count(
            current.levels,
            axis,
            tables,
            max_evaluations,
        )
        accepted_levels = current.levels + jax.nn.one_hot(
            axis,
            dimension,
            dtype=jnp.int32,
        )
        refresh_cost = _frontier_missing_count(
            accepted_levels,
            tables,
            cache_now,
            max_evaluations,
        )
        can_accept = current.evaluations + refresh_cost <= max_evaluations

        def stop_without_accept(operand: TensorState) -> TensorState:
            stop_status = jnp.where(
                converged_now,
                jnp.asarray(QuadStatus.CONVERGED, dtype=jnp.int32),
                jnp.where(
                    representable_new == 0,
                    jnp.asarray(QuadStatus.ROUNDOFF_LIMITED, dtype=jnp.int32),
                    jnp.asarray(QuadStatus.MAX_EVALUATIONS, dtype=jnp.int32),
                ),
            )
            return operand._replace(
                frontier_error=selected_frontier_error,
                status=stop_status,
                done=jnp.asarray(True),
            )

        def accept_and_refresh(operand: TensorState) -> TensorState:
            cache_before = _cache_from_state(operand)
            accepted_value = operand.directional_candidate_values[axis]
            (
                cache_after,
                new_candidates,
                new_error,
                new_frontier_error,
                new_directional_error,
                new_directional_cost,
            ) = _refresh_frontier(
                fun,
                domain,
                args=args,
                measure=measure,
                levels=accepted_levels,
                value=accepted_value,
                tables=tables,
                cache=cache_before,
                zero=zero,
                max_evaluations=max_evaluations,
                error_norm=error_norm,
            )
            _accepted_points, _accepted_weights, accepted_ids, accepted_active = (
                _candidate_formula(
                    accepted_levels,
                    tables,
                    max_evaluations,
                )
            )
            nonfinite_after = (
                cache_after.nonfinite
                | ~jnp.all(jnp.isfinite(accepted_value))
                | ~jnp.all(jnp.isfinite(new_error))
                | ~jnp.all(jnp.isfinite(new_directional_error))
                | ~jnp.isfinite(new_frontier_error)
            )
            new_status = jnp.where(
                nonfinite_after,
                jnp.asarray(QuadStatus.NONFINITE_INTEGRAND, dtype=jnp.int32),
                running,
            )
            return TensorState(
                levels=accepted_levels,
                value=accepted_value,
                error=new_error,
                frontier_error=new_frontier_error,
                directional_candidate_values=new_candidates,
                directional_error=new_directional_error,
                directional_new_cost=new_directional_cost,
                canonical_node_table=cache_after.canonical_ids,
                cached_points=cache_after.points,
                cached_values=cache_after.values,
                cache_active=cache_after.active,
                cache_count=cache_after.node_count,
                evaluations=cache_after.evaluations,
                refinements=operand.refinements + 1,
                status=new_status,
                done=nonfinite_after,
                accepted_node_ids=jnp.where(
                    accepted_active[:, None],
                    accepted_ids,
                    -1,
                ),
            )

        should_stop = converged_now | (representable_new == 0) | ~can_accept
        return jax.lax.cond(
            should_stop,
            stop_without_accept,
            accept_and_refresh,
            current,
        )

    def scan_body(current: TensorState, _unused):
        next_state = jax.lax.cond(
            current.done,
            lambda operand: operand,
            running_step,
            current,
        )
        return next_state, None

    state, _ = jax.lax.scan(
        scan_body,
        state,
        xs=None,
        length=capacity.max_refinements,
    )

    def finalize_running(current: TensorState) -> TensorState:
        axis, final_frontier_error = choose_tensor_axis(
            current.directional_error,
            current.directional_new_cost,
        )
        final_tolerance = tolerance_threshold(
            current.value,
            epsabs=epsabs,
            epsrel=epsrel,
            norm=error_norm,
        )
        converged_now = final_frontier_error <= final_tolerance
        representable_new = _selected_representable_new_count(
            current.levels,
            axis,
            tables,
            max_evaluations,
        )
        final_status = jnp.where(
            converged_now,
            jnp.asarray(QuadStatus.CONVERGED, dtype=jnp.int32),
            jnp.where(
                representable_new == 0,
                jnp.asarray(QuadStatus.ROUNDOFF_LIMITED, dtype=jnp.int32),
                jnp.asarray(QuadStatus.MAX_EVALUATIONS, dtype=jnp.int32),
            ),
        )
        return current._replace(
            frontier_error=final_frontier_error,
            status=final_status,
            done=jnp.asarray(True),
        )

    state = jax.lax.cond(
        state.done,
        lambda operand: operand,
        finalize_running,
        state,
    )
    final_tolerance = tolerance_threshold(
        state.value,
        epsabs=epsabs,
        epsrel=epsrel,
        norm=error_norm,
    )
    return AdaptiveTensorControllerResult(
        value=state.value,
        error=state.error,
        frontier_error=state.frontier_error,
        tolerance=final_tolerance,
        status=state.status,
        evaluations=state.evaluations,
        refinements=state.refinements,
        levels=state.levels,
        evidence=TensorReplayEvidence(
            levels=state.levels,
            active_node_ids=state.accepted_node_ids,
        ),
    )


__all__ = [
    "AdaptiveTensorControllerResult",
    "TensorReplayEvidence",
    "TensorRuleData",
    "adaptive_tensor_controller",
    "adaptive_tensor_tables",
    "canonical_cc_axis_ids",
    "canonical_tensor_ids",
    "choose_tensor_axis",
    "count_representable_new_nodes",
    "tensor_point_count",
    "tensor_rule_data",
    "validate_adaptive_tensor_capacity",
    "validate_b1_dimension",
]
