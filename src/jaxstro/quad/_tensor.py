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
    """Stopped accepted-formula metadata consumed by the Phase B4 replay owner."""

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
    represented_ids: Array
    counts: Array
    initial_level: int
    max_level: int


class _TensorCache(NamedTuple):
    point_keys: Array
    canonical_ids: Array
    points: Array
    values: Array
    hash_slots: Array
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
    cache: _TensorCache
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


class _RepresentedAxisLevel(NamedTuple):
    nodes: np.ndarray
    canonical_ids: np.ndarray
    represented_ids: np.ndarray
    formula_represented_ids: np.ndarray


def _target_dtype_name(dtype) -> str:
    selected = jnp.dtype(dtype)
    if selected not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise TypeError("adaptive Clenshaw-Curtis dtype must be float32 or float64")
    return selected.name


def _cc_unit_nodes(level: int, dtype) -> np.ndarray:
    order = (1 << level) + 1
    with jax.ensure_compile_time_eval():
        selected_dtype = jnp.dtype(dtype)
        index = jnp.arange(order, dtype=selected_dtype)
        half = jnp.asarray(0.5, dtype=selected_dtype)
        one = jnp.asarray(1.0, dtype=selected_dtype)
        nodes = half * (jnp.cos(jnp.pi * index / (order - 1)) + one)
        return np.asarray(nodes)


def _represented_cc_axis_level(
    level: int,
    dtype_name: str,
    *,
    canonical_owner: dict[tuple[int, int], float],
    represented_owner: dict[float, int],
) -> _RepresentedAxisLevel:
    """Build one target-dtype level while preserving nested alias ownership."""
    dtype = np.dtype(dtype_name)
    raw_nodes = _cc_unit_nodes(level, dtype_name)
    raw_ids = np.asarray(
        [_reduced_dyadic(index, level) for index in range((1 << level) + 1)],
        dtype=np.int32,
    )
    formula_nodes: list[float] = []
    formula_represented_ids: list[int] = []
    for raw_node, raw_id in zip(raw_nodes, raw_ids, strict=True):
        canonical_id = (int(raw_id[0]), int(raw_id[1]))
        point = canonical_owner.setdefault(canonical_id, float(raw_node))
        if point == 0.0:
            point = 0.0
        represented_id = represented_owner.setdefault(point, len(represented_owner))
        formula_nodes.append(point)
        formula_represented_ids.append(represented_id)

    compact_index: dict[int, int] = {}
    compact_nodes: list[float] = []
    compact_ids: list[np.ndarray] = []
    compact_represented_ids: list[int] = []
    for point, canonical_id, represented_id in zip(
        formula_nodes,
        raw_ids,
        formula_represented_ids,
        strict=True,
    ):
        if represented_id in compact_index:
            continue
        compact_index[represented_id] = len(compact_nodes)
        compact_nodes.append(point)
        compact_ids.append(canonical_id)
        compact_represented_ids.append(represented_id)
    return _RepresentedAxisLevel(
        nodes=np.asarray(compact_nodes, dtype=dtype),
        canonical_ids=np.asarray(compact_ids, dtype=np.int32),
        represented_ids=np.asarray(compact_represented_ids, dtype=np.int32),
        formula_represented_ids=np.asarray(
            formula_represented_ids,
            dtype=np.int32,
        ),
    )


@lru_cache(maxsize=None)
def _represented_cc_axis_metadata_cached(
    max_level: int,
    dtype_name: str,
) -> tuple[_RepresentedAxisLevel, ...]:
    canonical_owner: dict[tuple[int, int], float] = {}
    represented_owner: dict[float, int] = {}
    levels: list[_RepresentedAxisLevel] = []
    for level in range(max_level + 1):
        levels.append(
            _represented_cc_axis_level(
                level,
                dtype_name,
                canonical_owner=canonical_owner,
                represented_owner=represented_owner,
            )
        )
    return tuple(levels)


def represented_cc_axis_counts(
    *,
    initial_level: int,
    max_level: int,
    dtype,
) -> Array:
    """Return target-dtype distinct CC coordinate counts for a level ladder."""
    if (
        isinstance(initial_level, bool)
        or not isinstance(initial_level, int)
        or initial_level < 0
        or isinstance(max_level, bool)
        or not isinstance(max_level, int)
        or max_level < initial_level
    ):
        raise ValueError("represented CC levels must be ordered nonnegative integers")
    metadata = _represented_cc_axis_metadata_cached(
        max_level,
        _target_dtype_name(dtype),
    )
    return jnp.asarray(
        [metadata[level].nodes.size for level in range(initial_level, max_level + 1)],
        dtype=jnp.int32,
    )


def _represented_formula_cardinalities(
    levels: Array,
    *,
    represented_counts: Array,
    initial_level: int,
    max_level: int,
) -> tuple[Array, Array]:
    """Return accepted and one-axis represented-node cardinality growth."""
    levels = jnp.asarray(levels, dtype=jnp.int32)
    represented_counts = jnp.asarray(represented_counts, dtype=jnp.int32)
    level_index = levels - initial_level
    safe_index = jnp.clip(level_index, 0, represented_counts.shape[0] - 1)
    accepted_axis_counts = represented_counts[safe_index]
    accepted = jnp.prod(accepted_axis_counts, dtype=jnp.int32)
    unavailable = jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
    directional = []
    for axis in range(levels.shape[0]):
        next_in_range = levels[axis] < max_level
        next_index = jnp.minimum(safe_index[axis] + 1, represented_counts.shape[0] - 1)
        next_axis_count = represented_counts[next_index]
        candidate_axis_counts = accepted_axis_counts.at[axis].set(next_axis_count)
        candidate = jnp.prod(candidate_axis_counts, dtype=jnp.int32)
        directional.append(jnp.where(next_in_range, candidate - accepted, unavailable))
    return accepted, jnp.stack(directional)


def _represented_frontier_cardinality(
    levels: Array,
    *,
    represented_counts: Array,
    initial_level: int,
    max_level: int,
) -> Array:
    """Count the accepted grid plus disjoint one-axis represented-node slabs."""
    accepted, directional = _represented_formula_cardinalities(
        levels,
        represented_counts=represented_counts,
        initial_level=initial_level,
        max_level=max_level,
    )
    unavailable = jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
    has_unavailable = jnp.any(directional == unavailable)
    finite_directional = jnp.where(directional == unavailable, 0, directional)
    cardinality = accepted + jnp.sum(finite_directional, dtype=jnp.int32)
    return jnp.where(has_unavailable, unavailable, cardinality)


@lru_cache(maxsize=None)
def _adaptive_tensor_capacity_cached(
    initial_level: int,
    dimension: int,
    max_evaluations: int,
    dtype_name: str,
) -> AdaptiveTensorCapacity:
    canonical_owner: dict[tuple[int, int], float] = {}
    represented_owner: dict[float, int] = {}
    represented_counts: list[int] = []

    def axis_count(level: int) -> int:
        while len(represented_counts) <= level:
            next_level = len(represented_counts)
            metadata = _represented_cc_axis_level(
                next_level,
                dtype_name,
                canonical_owner=canonical_owner,
                represented_owner=represented_owner,
            )
            represented_counts.append(int(metadata.nodes.size))
        return represented_counts[level]

    initial_evaluations = 0
    for level in range(2, initial_level + 1):
        base_nodes = axis_count(level)
        base_count = base_nodes**dimension
        if base_count > max_evaluations:
            raise ValueError(
                "initial adaptive tensor frontier requires at least "
                f"{base_count} evaluations by level {level}, "
                f"exceeding max_evaluations={max_evaluations}"
            )
        next_nodes = axis_count(level + 1)
        one_axis_delta = (next_nodes - base_nodes) * base_nodes ** (dimension - 1)
        level_frontier = base_count + dimension * one_axis_delta
        if level_frontier > max_evaluations:
            if level == initial_level:
                raise ValueError(
                    "initial adaptive tensor frontier requires "
                    f"{level_frontier} evaluations, "
                    f"exceeding max_evaluations={max_evaluations}"
                )
            raise ValueError(
                "initial adaptive tensor frontier requires at least "
                f"{level_frontier} evaluations by level {level + 1}, "
                f"exceeding max_evaluations={max_evaluations}"
            )
        if level == initial_level:
            initial_evaluations = level_frontier

    base_nodes = axis_count(initial_level)

    max_level = initial_level + 1
    while True:
        current_nodes = axis_count(max_level)
        candidate_nodes = axis_count(max_level + 1)
        if candidate_nodes == current_nodes:
            max_level += 1
            break
        if candidate_nodes * base_nodes ** (dimension - 1) > max_evaluations:
            break
        max_level += 1

    levels = [initial_level] * dimension
    max_refinements = 0
    while True:
        current_count = math.prod(axis_count(level) for level in levels)
        candidates = []
        for axis in range(dimension):
            refined = levels.copy()
            refined[axis] += 1
            candidates.append(
                (
                    math.prod(axis_count(level) for level in refined),
                    axis,
                    refined,
                )
            )
        count, _axis, refined = min(candidates)
        if count > max_evaluations or count == current_count:
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
    dtype=None,
) -> AdaptiveTensorCapacity:
    """Validate static workspace sizes before payload inference or tracing."""
    validate_b1_dimension(dimension)
    if (
        not isinstance(max_evaluations, int)
        or isinstance(max_evaluations, bool)
        or max_evaluations <= 0
    ):
        raise ValueError("adaptive max_evaluations must be a positive integer")
    selected_dtype = jnp.asarray(0.0).dtype if dtype is None else dtype
    return _adaptive_tensor_capacity_cached(
        initial_level,
        dimension,
        max_evaluations,
        _target_dtype_name(selected_dtype),
    )


def adaptive_tensor_tables(
    *,
    initial_level: int,
    max_level: int,
    dtype,
) -> AdaptiveTensorTables:
    """Construct compact represented-coordinate CC ladders for the controller."""
    dtype_name = _target_dtype_name(dtype)
    metadata = _represented_cc_axis_metadata_cached(max_level, dtype_name)
    max_axis_nodes = max(
        metadata[level].nodes.size for level in range(initial_level, max_level + 1)
    )
    node_rows = []
    weight_rows = []
    id_rows = []
    represented_id_rows = []
    counts = []
    for level in range(initial_level, max_level + 1):
        _raw_nodes, raw_weights = _unit_rule_data(
            ClenshawCurtisRule((1 << level) + 1),
            dtype,
        )
        level_metadata = metadata[level]
        represented_to_index = {
            int(represented_id): index
            for index, represented_id in enumerate(level_metadata.represented_ids)
        }
        host_weights = np.asarray(raw_weights)
        compact_weights = np.zeros(
            level_metadata.nodes.size, dtype=np.dtype(dtype_name)
        )
        for weight, represented_id in zip(
            host_weights,
            level_metadata.formula_represented_ids,
            strict=True,
        ):
            index = represented_to_index[int(represented_id)]
            compact_weights[index] = np.asarray(
                compact_weights[index] + weight,
                dtype=np.dtype(dtype_name),
            )
        count = int(level_metadata.nodes.size)
        pad = max_axis_nodes - count
        node_rows.append(
            jnp.pad(jnp.asarray(level_metadata.nodes, dtype=dtype), (0, pad))
        )
        weight_rows.append(jnp.pad(jnp.asarray(compact_weights, dtype=dtype), (0, pad)))
        id_rows.append(
            jnp.pad(
                jnp.asarray(level_metadata.canonical_ids, dtype=jnp.int32),
                ((0, pad), (0, 0)),
                constant_values=-1,
            )
        )
        represented_id_rows.append(
            jnp.pad(
                jnp.asarray(level_metadata.represented_ids, dtype=jnp.int32),
                (0, pad),
                constant_values=-1,
            )
        )
        counts.append(count)
    return AdaptiveTensorTables(
        nodes=jnp.stack(node_rows),
        weights=jnp.stack(weight_rows),
        canonical_ids=jnp.stack(id_rows),
        represented_ids=jnp.stack(represented_id_rows),
        counts=jnp.asarray(counts, dtype=jnp.int32),
        initial_level=initial_level,
        max_level=max_level,
    )


def _formula_structure(
    levels: Array,
    tables: AdaptiveTensorTables,
) -> tuple[Array, Array, Array]:
    level_index = jnp.clip(
        levels - tables.initial_level,
        0,
        tables.max_level - tables.initial_level,
    )
    counts = tables.counts[level_index]
    point_count = jnp.prod(counts, dtype=jnp.int32)
    return level_index, counts, point_count


def _formula_row(
    flat: Array,
    *,
    level_index: Array,
    counts: Array,
    tables: AdaptiveTensorTables,
) -> tuple[Array, Array, Array, Array]:
    point_columns = []
    weight_columns = []
    id_columns = []
    represented_key_columns = []
    dimension = counts.shape[0]
    for axis in range(dimension):
        divisor = jnp.prod(counts[axis + 1 :], dtype=jnp.int32)
        index = (flat // divisor) % counts[axis]
        point_columns.append(tables.nodes[level_index[axis], index])
        weight_columns.append(tables.weights[level_index[axis], index])
        id_columns.append(tables.canonical_ids[level_index[axis], index])
        represented_key_columns.append(tables.represented_ids[level_index[axis], index])
    points = jnp.stack(point_columns, axis=-1)
    weights = jnp.prod(jnp.stack(weight_columns, axis=-1), axis=-1)
    canonical_ids = jnp.concatenate(id_columns, axis=-1)
    represented_key = jnp.stack(represented_key_columns)
    return points, weights, canonical_ids, represented_key


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


def _hash_capacity(max_evaluations: int) -> int:
    return max(4, 1 << (2 * max_evaluations - 1).bit_length())


def _hash_point_key(point_key: Array, mask: int) -> Array:
    hashed = jnp.asarray(2166136261, dtype=jnp.uint32)
    for coordinate in point_key:
        hashed = (
            hashed
            ^ (
                jnp.asarray(coordinate, dtype=jnp.uint32)
                + jnp.asarray(0x9E3779B9, dtype=jnp.uint32)
            )
        ) * jnp.asarray(16777619, dtype=jnp.uint32)
    return jnp.asarray(
        hashed & jnp.asarray(mask, dtype=jnp.uint32),
        dtype=jnp.int32,
    )


class _CacheLookup(NamedTuple):
    found: Array
    value_index: Array
    hash_slot: Array


class _CacheProbeState(NamedTuple):
    slot: Array
    probes: Array
    found: Array
    value_index: Array
    insertion_slot: Array
    done: Array


def _cache_lookup(cache: _TensorCache, point_key: Array) -> _CacheLookup:
    """Find one represented tuple with exact open-addressed hash probing."""
    hash_capacity = cache.hash_slots.shape[0]
    mask = hash_capacity - 1
    initial_slot = _hash_point_key(point_key, mask)
    initial = _CacheProbeState(
        slot=initial_slot,
        probes=jnp.asarray(0, dtype=jnp.int32),
        found=jnp.asarray(False),
        value_index=jnp.asarray(0, dtype=jnp.int32),
        insertion_slot=jnp.asarray(-1, dtype=jnp.int32),
        done=jnp.asarray(False),
    )

    def probe(current: _CacheProbeState) -> _CacheProbeState:
        cached_index = cache.hash_slots[current.slot]
        occupied = cached_index >= 0
        safe_index = jnp.maximum(cached_index, 0)
        same_key = occupied & jnp.all(
            cache.point_keys[safe_index] == point_key,
            axis=-1,
        )
        empty = ~occupied
        exhausted = current.probes + 1 >= hash_capacity
        done = same_key | empty | exhausted
        return _CacheProbeState(
            slot=jnp.where(done, current.slot, (current.slot + 1) & mask),
            probes=current.probes + 1,
            found=same_key,
            value_index=jnp.where(same_key, cached_index, 0),
            insertion_slot=jnp.where(empty, current.slot, -1),
            done=done,
        )

    result = jax.lax.while_loop(
        lambda current: ~current.done,
        probe,
        initial,
    )
    return _CacheLookup(
        found=result.found,
        value_index=result.value_index,
        hash_slot=result.insertion_slot,
    )


class _FormulaEvaluationState(NamedTuple):
    cache: _TensorCache
    value: Array
    canonical_ids: Array


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
    level_index, counts, point_count = _formula_structure(levels, tables)
    zero = jnp.asarray(zero)
    dimension = levels.shape[0]
    initial = _FormulaEvaluationState(
        cache=cache,
        value=zero,
        canonical_ids=jnp.full(
            (max_evaluations, 2 * dimension),
            -1,
            dtype=jnp.int32,
        ),
    )

    def body(flat: Array, current: _FormulaEvaluationState):
        point, weight, canonical_id, point_key = _formula_row(
            flat,
            level_index=level_index,
            counts=counts,
            tables=tables,
        )
        lookup = _cache_lookup(current.cache, point_key)

        def reuse(existing: _TensorCache):
            return existing, existing.values[lookup.value_index]

        def evaluate_new(existing: _TensorCache):
            available = (existing.node_count < max_evaluations) & (
                lookup.hash_slot >= 0
            )

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
                    point_keys=store.point_keys.at[index].set(point_key),
                    canonical_ids=store.canonical_ids.at[index].set(canonical_id),
                    points=store.points.at[index].set(point),
                    values=store.values.at[index].set(contribution),
                    hash_slots=store.hash_slots.at[lookup.hash_slot].set(index),
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

        def process(active_cache: _TensorCache):
            return jax.lax.cond(
                lookup.found,
                reuse,
                evaluate_new,
                active_cache,
            )

        def skip(nonfinite_cache: _TensorCache):
            return nonfinite_cache, zero

        updated_cache, contribution = jax.lax.cond(
            current.cache.nonfinite,
            skip,
            process,
            current.cache,
        )
        return _FormulaEvaluationState(
            cache=updated_cache,
            value=current.value + contribution * weight,
            canonical_ids=current.canonical_ids.at[flat].set(canonical_id),
        )

    evaluated = jax.lax.fori_loop(
        0,
        point_count,
        body,
        initial,
    )
    active = jnp.arange(max_evaluations, dtype=jnp.int32) < point_count
    return (
        evaluated.cache,
        evaluated.value,
        evaluated.canonical_ids,
        active,
    )


def _directional_data(
    value: Array,
    candidate_values: Array,
    directional_new_cost: Array,
    error_norm: ErrorNorm,
) -> tuple[Array, Array, Array, Array]:
    component_error = jnp.abs(candidate_values - value)
    directional_error = jnp.stack(
        [
            reduce_error_norm(component_error[axis], error_norm)
            for axis in range(candidate_values.shape[0])
        ]
    )
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
    _accepted_count, directional_new_cost = _represented_formula_cardinalities(
        levels,
        represented_counts=tables.counts,
        initial_level=tables.initial_level,
        max_level=tables.max_level,
    )
    error, frontier_error, directional_error, directional_new_cost = _directional_data(
        value,
        candidates,
        directional_new_cost,
        error_norm,
    )
    return (
        cache,
        candidates,
        error,
        frontier_error,
        directional_error,
        directional_new_cost,
    )


def _formula_node_ids(
    levels: Array,
    tables: AdaptiveTensorTables,
    max_evaluations: int,
) -> tuple[Array, Array]:
    level_index, counts, point_count = _formula_structure(levels, tables)
    ids = jnp.full(
        (max_evaluations, 2 * levels.shape[0]),
        -1,
        dtype=jnp.int32,
    )

    def store_id(flat: Array, current: Array) -> Array:
        _point, _weight, canonical_id, _point_key = _formula_row(
            flat,
            level_index=level_index,
            counts=counts,
            tables=tables,
        )
        return current.at[flat].set(canonical_id)

    ids = jax.lax.fori_loop(0, point_count, store_id, ids)
    active = jnp.arange(max_evaluations, dtype=jnp.int32) < point_count
    return ids, active


def adaptive_tensor_replay_formula(
    levels: Array,
    tables: AdaptiveTensorTables,
    max_evaluations: int,
) -> tuple[Array, Array, Array]:
    """Materialize the accepted normalized tensor formula at fixed capacity."""
    level_index, counts, point_count = _formula_structure(levels, tables)
    points = jnp.zeros(
        (max_evaluations, levels.shape[0]),
        dtype=tables.nodes.dtype,
    )
    weights = jnp.zeros((max_evaluations,), dtype=tables.weights.dtype)

    def store_row(flat: Array, carry: tuple[Array, Array]):
        current_points, current_weights = carry
        point, weight, _canonical_id, _point_key = _formula_row(
            flat,
            level_index=level_index,
            counts=counts,
            tables=tables,
        )
        return (
            current_points.at[flat].set(point),
            current_weights.at[flat].set(weight),
        )

    points, weights = jax.lax.fori_loop(
        0,
        point_count,
        store_row,
        (points, weights),
    )
    active = jnp.arange(max_evaluations, dtype=jnp.int32) < point_count
    return points, weights, active


def _frontier_refresh_cost(
    levels: Array,
    accepted_axis: Array,
    tables: AdaptiveTensorTables,
) -> Array:
    """Predict the exact new represented tuples after one accepted refinement."""
    current = _represented_frontier_cardinality(
        levels,
        represented_counts=tables.counts,
        initial_level=tables.initial_level,
        max_level=tables.max_level,
    )
    accepted_levels = levels + jax.nn.one_hot(
        accepted_axis,
        levels.shape[0],
        dtype=jnp.int32,
    )
    refreshed = _represented_frontier_cardinality(
        accepted_levels,
        represented_counts=tables.counts,
        initial_level=tables.initial_level,
        max_level=tables.max_level,
    )
    unavailable = jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
    return jnp.where(
        (current == unavailable) | (refreshed == unavailable),
        unavailable,
        jnp.maximum(refreshed - current, 0),
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
        point_keys=jnp.full(
            (max_evaluations, dimension),
            -1,
            dtype=jnp.int32,
        ),
        canonical_ids=jnp.full(
            (max_evaluations, 2 * dimension),
            -1,
            dtype=jnp.int32,
        ),
        points=jnp.zeros((max_evaluations, dimension), dtype=dtype),
        values=jnp.zeros((max_evaluations,) + zero.shape, dtype=value_dtype),
        hash_slots=jnp.full(
            (_hash_capacity(max_evaluations),),
            -1,
            dtype=jnp.int32,
        ),
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
        cache=cache,
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
        representable_new = current.directional_new_cost[axis]
        accepted_levels = current.levels + jax.nn.one_hot(
            axis,
            dimension,
            dtype=jnp.int32,
        )
        refresh_cost = _frontier_refresh_cost(
            current.levels,
            axis,
            tables,
        )
        remaining_capacity = max_evaluations - current.evaluations
        can_accept = refresh_cost <= remaining_capacity

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
                cache=operand.cache,
                zero=zero,
                max_evaluations=max_evaluations,
                error_norm=error_norm,
            )
            accepted_ids, accepted_active = _formula_node_ids(
                accepted_levels,
                tables,
                max_evaluations,
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
                cache=cache_after,
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
        representable_new = current.directional_new_cost[axis]
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
    "adaptive_tensor_replay_formula",
    "adaptive_tensor_tables",
    "canonical_cc_axis_ids",
    "canonical_tensor_ids",
    "choose_tensor_axis",
    "represented_cc_axis_counts",
    "tensor_point_count",
    "tensor_rule_data",
    "validate_adaptive_tensor_capacity",
    "validate_b1_dimension",
]
