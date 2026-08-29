"""Exact identities and hierarchical rules for Smolyak integration."""

import itertools
import math
from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ._chebyshev import chebyshev_rule_data
from ._multidim import evaluate_multidim
from .result import QuadStatus
from .rules import ClenshawCurtisRule, FixedRuleData
from .tolerance import ErrorNorm, tolerance_threshold
from .tolerance import error_norm as reduce_error_norm

RUNNING = -1

DyadicIdentity = tuple[int, int]


class HierarchicalRule(NamedTuple):
    identities: tuple[DyadicIdentity, ...]
    points: Array
    weights: Array


SparseNodeIdentity = tuple[DyadicIdentity, ...]
SparseIndex = tuple[int, ...]


class SparseRuleData(NamedTuple):
    identities: tuple[SparseNodeIdentity, ...]
    points: Array
    weights: Array
    increment_weights: Array
    indices: tuple[SparseIndex, ...]
    frontier_mask: Array
    point_count: int
    index_count: int


class _SparseHostData(NamedTuple):
    identities: tuple[SparseNodeIdentity, ...]
    points: np.ndarray
    weights: np.ndarray
    increment_weights: np.ndarray
    indices: tuple[SparseIndex, ...]
    frontier_mask: np.ndarray


class SparseReplayEvidence(NamedTuple):
    """Stopped adaptive sparse formula metadata consumed by Phase B4 replay."""

    indices: Array
    active: Array
    node_ids: Array
    coefficients: Array
    node_active: Array


class AdaptiveSparseControllerResult(NamedTuple):
    value: Array
    error: Array
    frontier_error: Array
    tolerance: Array
    status: Array
    evaluations: Array
    refinements: Array
    level: Array
    evidence: SparseReplayEvidence


class _AdaptiveSparseTables(NamedTuple):
    points: Array
    weights: Array
    identities: Array
    counts: Array


class _SparseCache(NamedTuple):
    identities: Array
    values: Array
    active: Array
    accepted: Array
    coefficients: Array
    evaluations: Array
    nonfinite: Array
    exhausted: Array


class _SparseAdaptiveState(NamedTuple):
    accepted_indices: Array
    accepted_active: Array
    accepted_count: Array
    frontier_indices: Array
    frontier_active: Array
    frontier_evaluated: Array
    frontier_surplus: Array
    frontier_norm: Array
    frontier_new_cost: Array
    cache: _SparseCache
    value: Array
    evaluations: Array
    refinements: Array
    status: Array
    done: Array


def canonical_cc_identity(level: int, index: int) -> DyadicIdentity:
    """Reduce one Clenshaw-Curtis angle index to an exact dyadic identity."""
    if isinstance(level, bool) or not isinstance(level, int) or level < 1:
        raise ValueError("Clenshaw-Curtis level must be a positive integer")
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("Clenshaw-Curtis index must be an integer")
    if level == 1:
        if index != 0:
            raise ValueError("Clenshaw-Curtis index is outside its level")
        return 1, 1
    angle_level = level - 1
    denominator = 1 << angle_level
    if index < 0 or index > denominator:
        raise ValueError("Clenshaw-Curtis index is outside its level")
    if index == 0:
        return 0, 0
    while angle_level > 0 and index % 2 == 0:
        index //= 2
        angle_level -= 1
    return index, angle_level


def identities_to_points(identities: Array, dtype) -> Array:
    """Map canonical dyadic identities to their exact special coordinates."""
    identities = jnp.asarray(identities)
    if identities.shape[-1:] != (2,):
        raise ValueError("dyadic identities must have a final axis of length 2")
    selected_dtype = jnp.dtype(dtype)
    numerator = identities[..., 0]
    denominator_power = identities[..., 1]
    numerator_value = numerator.astype(selected_dtype)
    denominator = jnp.exp2(denominator_power.astype(selected_dtype))
    general = jnp.asarray(0.5, dtype=selected_dtype) * (
        jnp.asarray(1.0, dtype=selected_dtype)
        - jnp.cos(jnp.pi * numerator_value / denominator)
    )
    return jnp.where(
        (numerator == 0) & (denominator_power == 0),
        jnp.asarray(0.0, dtype=selected_dtype),
        jnp.where(
            (numerator == 1) & (denominator_power == 0),
            jnp.asarray(1.0, dtype=selected_dtype),
            jnp.where(
                (numerator == 1) & (denominator_power == 1),
                jnp.asarray(0.5, dtype=selected_dtype),
                general,
            ),
        ),
    )


def identity_to_point(identity: DyadicIdentity, dtype) -> Array:
    """Create a unit-interval coordinate after exact identity coalescing."""
    return identities_to_points(jnp.asarray(identity), dtype)


def unit_clenshaw_curtis(level: int, dtype) -> FixedRuleData:
    """Construct the level-indexed nested Clenshaw-Curtis rule on ``[0, 1]``."""
    if isinstance(level, bool) or not isinstance(level, int) or level < 1:
        raise ValueError("sparse Clenshaw-Curtis level must be a positive integer")
    selected_dtype = jnp.dtype(dtype)
    if level == 1:
        return FixedRuleData(
            nodes=jnp.asarray([0.5], dtype=selected_dtype),
            weights=jnp.asarray([1.0], dtype=selected_dtype),
            degree=1,
            nested=True,
        )
    data = chebyshev_rule_data(
        ClenshawCurtisRule((1 << (level - 1)) + 1),
        dtype=selected_dtype,
    )
    half = jnp.asarray(0.5, dtype=selected_dtype)
    return FixedRuleData(
        nodes=half * (jnp.asarray(1.0, dtype=selected_dtype) - data.nodes),
        weights=half * data.weights,
        degree=data.degree,
        nested=True,
    )


def hierarchical_rule(level: int, dtype) -> HierarchicalRule:
    """Return ``Q_level - Q_(level-1)`` coalesced by exact dyadic identity."""
    selected_dtype = jnp.dtype(dtype)
    host_dtype = np.dtype(selected_dtype.name)
    scalar = host_dtype.type
    with jax.ensure_compile_time_eval():
        current = unit_clenshaw_curtis(level, selected_dtype)
        current_weights = np.asarray(current.weights, dtype=host_dtype)
        weights: dict[DyadicIdentity, np.floating] = {}
        for index, weight in enumerate(current_weights):
            identity = canonical_cc_identity(level, index)
            weights[identity] = scalar(weights.get(identity, scalar(0)) + weight)

        if level > 1:
            previous = unit_clenshaw_curtis(level - 1, selected_dtype)
            previous_weights = np.asarray(previous.weights, dtype=host_dtype)
            for index, weight in enumerate(previous_weights):
                identity = canonical_cc_identity(level - 1, index)
                weights[identity] = scalar(weights.get(identity, scalar(0)) - weight)

        identities = tuple(
            identity for identity in sorted(weights) if weights[identity] != scalar(0)
        )
        points = jnp.asarray(
            [identity_to_point(identity, selected_dtype) for identity in identities],
            dtype=selected_dtype,
        )
        difference_weights = jnp.asarray(
            [weights[identity] for identity in identities],
            dtype=selected_dtype,
        )
    return HierarchicalRule(
        identities=identities,
        points=points,
        weights=difference_weights,
    )


def _validate_dimension(dimension: int) -> None:
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("sparse-grid dimension must be a positive integer")


def _anisotropy_weights(
    anisotropy: tuple[float, ...] | None,
    dimension: int,
) -> tuple[float, ...]:
    _validate_dimension(dimension)
    if anisotropy is None:
        return (1.0,) * dimension
    if len(anisotropy) != dimension:
        raise ValueError("Smolyak anisotropy requires one weight per dimension")
    return anisotropy


def _fixed_index_set(
    level: int,
    anisotropy: tuple[float, ...] | None,
    dimension: int,
) -> tuple[SparseIndex, ...]:
    weights = _anisotropy_weights(anisotropy, dimension)
    indices: list[SparseIndex] = []

    def append_indices(
        axis: int,
        remaining: float,
        current: tuple[int, ...],
    ) -> None:
        if axis == dimension:
            indices.append(current)
            return
        weight = weights[axis]
        max_excess = int(math.floor(remaining / weight))
        for excess in range(max_excess + 1):
            append_indices(
                axis + 1,
                remaining - weight * excess,
                current + (excess + 1,),
            )

    append_indices(0, float(level - 1), ())
    return tuple(sorted(indices, key=lambda index: (sum(index), index)))


def fixed_index_set(method, dimension: int) -> tuple[SparseIndex, ...]:
    """Enumerate the deterministic downward-closed fixed Smolyak index set."""
    return _fixed_index_set(method.level, method.anisotropy, dimension)


def _frontier_indices(indices: tuple[SparseIndex, ...]) -> set[SparseIndex]:
    accepted = set(indices)
    return {
        index
        for index in indices
        if not any(
            index[:axis] + (index[axis] + 1,) + index[axis + 1 :] in accepted
            for axis in range(len(index))
        )
    }


def sparse_axis_identities(level: int) -> tuple[DyadicIdentity, ...]:
    """Return exact identities in one hierarchical increment without floats."""
    point_count = 1 if level == 1 else (1 << (level - 1)) + 1
    return tuple(canonical_cc_identity(level, index) for index in range(point_count))


def fixed_sparse_node_identities(
    indices: tuple[SparseIndex, ...],
    *,
    node_limit: int | None = None,
    limit_name: str = "node_limit",
) -> tuple[SparseNodeIdentity, ...]:
    """Coalesce the exact multidimensional identity union before rule arrays."""
    identities: set[SparseNodeIdentity] = set()
    for index in indices:
        axes = [sparse_axis_identities(level) for level in index]
        for identity in itertools.product(*axes):
            identities.add(identity)
            if node_limit is not None and len(identities) > node_limit:
                raise ValueError(
                    f"fixed Smolyak requires more than {node_limit} unique nodes, "
                    f"exceeding {limit_name}={node_limit}"
                )
    return tuple(sorted(identities))


def _host_identity_to_point(
    identity: DyadicIdentity,
    dtype: np.dtype,
) -> np.floating:
    scalar = dtype.type
    if identity == (0, 0):
        return scalar(0)
    if identity == (1, 0):
        return scalar(1)
    if identity == (1, 1):
        return scalar(0.5)
    numerator, denominator_power = identity
    angle = scalar(np.pi) * scalar(numerator) / scalar(1 << denominator_power)
    return scalar(scalar(0.5) * (scalar(1.0) - np.cos(angle, dtype=dtype)))


@lru_cache(maxsize=None)
def _smolyak_host_data(
    level: int,
    anisotropy: tuple[float, ...] | None,
    dimension: int,
    dtype_name: str,
) -> _SparseHostData:
    indices = _fixed_index_set(level, anisotropy, dimension)
    dtype = np.dtype(dtype_name)
    scalar = dtype.type
    per_index: list[dict[SparseNodeIdentity, np.floating]] = []
    identities = fixed_sparse_node_identities(indices)

    for index in indices:
        axes = [hierarchical_rule(axis_level, dtype_name) for axis_level in index]
        axis_weights = [np.asarray(axis.weights, dtype=dtype) for axis in axes]
        increment: dict[SparseNodeIdentity, np.floating] = {}
        ranges = [range(len(axis.identities)) for axis in axes]
        for positions in itertools.product(*ranges):
            identity = tuple(
                axes[axis].identities[position]
                for axis, position in enumerate(positions)
            )
            weight = scalar(1)
            for axis, position in enumerate(positions):
                weight = scalar(weight * axis_weights[axis][position])
            if weight != scalar(0):
                increment[identity] = scalar(
                    increment.get(identity, scalar(0)) + weight
                )
        increment = {
            identity: weight
            for identity, weight in increment.items()
            if weight != scalar(0)
        }
        per_index.append(increment)

    identity_position = {
        identity: position for position, identity in enumerate(identities)
    }
    increment_weights = np.zeros((len(indices), len(identities)), dtype=dtype)
    for row, increment in enumerate(per_index):
        for identity, weight in increment.items():
            increment_weights[row, identity_position[identity]] = weight
    weights = np.sum(increment_weights, axis=0, dtype=dtype)
    points = np.asarray(
        [
            [
                _host_identity_to_point(axis_identity, dtype)
                for axis_identity in identity
            ]
            for identity in identities
        ],
        dtype=dtype,
    )
    frontier = _frontier_indices(indices)
    frontier_mask = np.asarray(
        [index in frontier for index in indices],
        dtype=np.bool_,
    )
    return _SparseHostData(
        identities=identities,
        points=points,
        weights=weights,
        increment_weights=increment_weights,
        indices=indices,
        frontier_mask=frontier_mask,
    )


def _target_sparse_dtype(dtype) -> str:
    selected = jnp.dtype(dtype)
    if selected not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise TypeError("sparse-grid dtype must be float32 or float64")
    return selected.name


def smolyak_host_data(method, dimension: int, dtype) -> _SparseHostData:
    """Return cached host data so capacities can be checked before tracing."""
    return _smolyak_host_data(
        method.level,
        method.anisotropy,
        dimension,
        _target_sparse_dtype(dtype),
    )


def materialize_smolyak_rule(host: _SparseHostData) -> SparseRuleData:
    """Create JAX arrays only after caller-owned static capacity checks."""
    return SparseRuleData(
        identities=host.identities,
        points=jnp.asarray(host.points),
        weights=jnp.asarray(host.weights),
        increment_weights=jnp.asarray(host.increment_weights),
        indices=host.indices,
        frontier_mask=jnp.asarray(host.frontier_mask),
        point_count=host.points.shape[0],
        index_count=len(host.indices),
    )


def smolyak_rule_data(method, dimension: int, dtype) -> SparseRuleData:
    """Construct one exact-node-coalesced fixed Smolyak rule."""
    return materialize_smolyak_rule(smolyak_host_data(method, dimension, dtype))


def is_admissible(
    candidate: SparseIndex,
    accepted: set[SparseIndex],
) -> bool:
    """Return whether every valid immediate backward neighbor is accepted."""
    for axis, component in enumerate(candidate):
        if component > 1:
            backward = list(candidate)
            backward[axis] -= 1
            if tuple(backward) not in accepted:
                return False
    return True


def admissible_forward_neighbors(
    accepted: set[SparseIndex],
    dimension: int,
) -> tuple[SparseIndex, ...]:
    """Enumerate the lexicographically ordered admissible forward frontier."""
    _validate_dimension(dimension)
    if any(len(index) != dimension for index in accepted):
        raise ValueError("accepted sparse indices must match dimension")
    candidates: set[SparseIndex] = set()
    for index in accepted:
        for axis in range(dimension):
            candidate = index[:axis] + (index[axis] + 1,) + index[axis + 1 :]
            if candidate not in accepted and is_admissible(candidate, accepted):
                candidates.add(candidate)
    return tuple(sorted(candidates))


def required_frontier_capacity(dimension: int, accepted_count: int) -> int:
    """Return the proved fixed frontier allocation for an adaptive declaration."""
    _validate_dimension(dimension)
    if (
        isinstance(accepted_count, bool)
        or not isinstance(accepted_count, int)
        or accepted_count < 1
    ):
        raise ValueError("accepted_count must be a positive integer")
    return 1 + dimension * accepted_count


def select_profit(
    indices: tuple[SparseIndex, ...],
    surplus_norm: Array,
    new_cost: Array,
):
    """Select maximum surplus-per-new-node with an explicit lexicographic tie."""
    if not indices:
        raise ValueError("profit selection requires at least one sparse index")
    surplus_norm = jnp.asarray(surplus_norm)
    new_cost = jnp.asarray(new_cost)
    if surplus_norm.shape != (len(indices),) or new_cost.shape != (len(indices),):
        raise ValueError("profit arrays must match the sparse-index count")
    order = np.asarray(sorted(range(len(indices)), key=indices.__getitem__))
    profit = jnp.where(
        new_cost > 0,
        surplus_norm / jnp.maximum(new_cost, 1),
        -jnp.inf,
    )
    ordered_slot = jnp.argmax(profit[jnp.asarray(order)])
    return jnp.asarray(order)[ordered_slot]


def sparse_termination_status(
    *,
    invalid,
    nonfinite,
    converged,
    all_active_roundoff,
    evaluation_exhausted,
    index_exhausted,
):
    """Apply the public adaptive sparse-grid status precedence."""
    status = jnp.asarray(RUNNING, dtype=jnp.int32)
    status = jnp.where(
        index_exhausted,
        jnp.asarray(QuadStatus.MAX_INDICES, dtype=jnp.int32),
        status,
    )
    status = jnp.where(
        evaluation_exhausted,
        jnp.asarray(QuadStatus.MAX_EVALUATIONS, dtype=jnp.int32),
        status,
    )
    status = jnp.where(
        all_active_roundoff,
        jnp.asarray(QuadStatus.ROUNDOFF_LIMITED, dtype=jnp.int32),
        status,
    )
    status = jnp.where(
        converged,
        jnp.asarray(QuadStatus.CONVERGED, dtype=jnp.int32),
        status,
    )
    status = jnp.where(
        nonfinite,
        jnp.asarray(QuadStatus.NONFINITE_INTEGRAND, dtype=jnp.int32),
        status,
    )
    return jnp.where(
        invalid,
        jnp.asarray(QuadStatus.INVALID_INPUT, dtype=jnp.int32),
        status,
    )


@lru_cache(maxsize=None)
def _adaptive_sparse_host_tables(
    max_nodes: int,
    dtype_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dtype = np.dtype(dtype_name)
    max_level = 1
    while (1 << max_level) + 1 <= max_nodes:
        max_level += 1
    rules = [hierarchical_rule(level, dtype_name) for level in range(1, max_level + 1)]
    max_axis_nodes = max(len(rule.identities) for rule in rules)
    points = np.zeros((max_level, max_axis_nodes), dtype=dtype)
    weights = np.zeros((max_level, max_axis_nodes), dtype=dtype)
    identities = np.full((max_level, max_axis_nodes, 2), -1, dtype=np.int32)
    counts = np.zeros((max_level,), dtype=np.int32)
    for row, rule in enumerate(rules):
        count = len(rule.identities)
        counts[row] = count
        points[row, :count] = np.asarray(rule.points, dtype=dtype)
        weights[row, :count] = np.asarray(rule.weights, dtype=dtype)
        identities[row, :count] = np.asarray(rule.identities, dtype=np.int32)
    return points, weights, identities, counts


def _adaptive_sparse_tables(max_nodes: int, dtype) -> _AdaptiveSparseTables:
    points, weights, identities, counts = _adaptive_sparse_host_tables(
        max_nodes,
        _target_sparse_dtype(dtype),
    )
    return _AdaptiveSparseTables(
        points=jnp.asarray(points),
        weights=jnp.asarray(weights),
        identities=jnp.asarray(identities),
        counts=jnp.asarray(counts),
    )


def _index_membership(index: Array, indices: Array, active: Array) -> Array:
    return jnp.any(active & jnp.all(indices == index, axis=1))


def _jax_is_admissible(
    candidate: Array,
    accepted_indices: Array,
    accepted_active: Array,
) -> Array:
    dimension = candidate.shape[0]

    def axis_is_admissible(axis):
        backward = candidate.at[axis].add(-1)
        return (candidate[axis] == 1) | _index_membership(
            backward,
            accepted_indices,
            accepted_active,
        )

    return jnp.all(jax.vmap(axis_is_admissible)(jnp.arange(dimension)))


def _formula_layout(
    index: Array,
    tables: _AdaptiveSparseTables,
    max_nodes: int,
) -> tuple[Array, Array, Array, Array]:
    """Materialize one padded hierarchical tensor formula from traced levels."""
    dimension = index.shape[0]
    max_level = tables.counts.shape[0]
    valid_level = jnp.all((index >= 1) & (index <= max_level))
    safe_levels = jnp.clip(index - 1, 0, max_level - 1)
    counts = tables.counts[safe_levels]

    def capped_product(total, count):
        safe = total <= max_nodes // jnp.maximum(count, 1)
        return jnp.where(safe, total * count, max_nodes + 1), None

    point_count, _ = jax.lax.scan(
        capped_product,
        jnp.asarray(1, dtype=jnp.int32),
        counts,
    )
    slots = jnp.arange(max_nodes, dtype=jnp.int32)

    def unravel_axis(remainder, axis):
        reverse_axis = dimension - axis - 1
        count = counts[reverse_axis]
        position = remainder % count
        return remainder // count, position

    _, reverse_positions = jax.lax.scan(
        unravel_axis,
        slots,
        jnp.arange(dimension, dtype=jnp.int32),
    )
    positions = jnp.swapaxes(reverse_positions[::-1], 0, 1)

    def row(position):
        points = tables.points[safe_levels, position]
        weights = tables.weights[safe_levels, position]
        identities = tables.identities[safe_levels, position]
        return points, jnp.prod(weights), identities.reshape(-1)

    points, weights, identities = jax.vmap(row)(positions)
    active = valid_level & (slots < point_count) & (point_count <= max_nodes)
    return points, weights, identities, active


def _cache_lookup(cache: _SparseCache, identity: Array) -> tuple[Array, Array]:
    matches = cache.active & jnp.all(cache.identities == identity, axis=1)
    return jnp.any(matches), jnp.asarray(jnp.argmax(matches), dtype=jnp.int32)


def _evaluate_sparse_formula(
    fun,
    domain,
    *,
    args,
    measure,
    index: Array,
    tables: _AdaptiveSparseTables,
    cache: _SparseCache,
    zero: Array,
    max_evaluations: int,
    max_nodes: int,
) -> tuple[_SparseCache, Array, Array, Array]:
    points, weights, identities, formula_active = _formula_layout(
        index,
        tables,
        max_nodes,
    )
    value_dtype = cache.values.dtype
    formula_slots = jnp.full((max_nodes,), -1, dtype=jnp.int32)

    def scan_row(carry, row):
        current_cache, surplus, slots_out = carry
        slot, point, weight, identity, active = row
        found, cache_slot = _cache_lookup(current_cache, identity)
        free_slot = jnp.asarray(
            jnp.argmax(~current_cache.active),
            dtype=jnp.int32,
        )
        can_insert = (
            active
            & ~found
            & (current_cache.evaluations < max_evaluations)
            & jnp.any(~current_cache.active)
        )
        cannot_insert = active & ~found & ~can_insert

        def evaluate_new(operand):
            cache_before, _surplus, _slots = operand
            evaluated = evaluate_multidim(
                fun,
                domain,
                point[None, :],
                args=args,
                measure=measure,
            )
            contribution = jnp.asarray(
                evaluated.values[0] * evaluated.weights[0],
                dtype=value_dtype,
            )
            cache_after = cache_before._replace(
                identities=cache_before.identities.at[free_slot].set(identity),
                values=cache_before.values.at[free_slot].set(contribution),
                active=cache_before.active.at[free_slot].set(True),
                evaluations=cache_before.evaluations + 1,
                nonfinite=cache_before.nonfinite
                | evaluated.nonfinite
                | ~evaluated.valid
                | ~jnp.all(jnp.isfinite(contribution)),
            )
            return cache_after

        current_cache = jax.lax.cond(
            can_insert,
            evaluate_new,
            lambda operand: operand[0],
            (current_cache, surplus, slots_out),
        )
        chosen_slot = jnp.where(found, cache_slot, free_slot)
        usable = active & (found | can_insert)
        contribution = current_cache.values[chosen_slot]
        reshape = (1,) * zero.ndim
        weighted = contribution * jnp.reshape(weight, reshape)
        surplus = surplus + jnp.where(usable, weighted, jnp.zeros_like(weighted))
        slots_out = slots_out.at[slot].set(jnp.where(usable, chosen_slot, -1))
        current_cache = current_cache._replace(
            exhausted=current_cache.exhausted | cannot_insert,
        )
        return (current_cache, surplus, slots_out), None

    rows = (
        jnp.arange(max_nodes, dtype=jnp.int32),
        points,
        weights,
        identities,
        formula_active,
    )
    (cache, surplus, formula_slots), _ = jax.lax.scan(
        scan_row,
        (cache, jnp.asarray(zero), formula_slots),
        rows,
    )
    active_slots = formula_slots >= 0
    safe_slots = jnp.maximum(formula_slots, 0)
    new_cost = jnp.sum(
        active_slots & ~cache.accepted[safe_slots],
        dtype=jnp.int32,
    )
    return cache, surplus, new_cost, formula_slots


def _mark_formula_accepted(
    cache: _SparseCache,
    formula_slots: Array,
    formula_weights: Array,
) -> _SparseCache:
    def mark(carry, row):
        accepted, coefficients = carry
        slot, weight = row
        return jax.lax.cond(
            slot >= 0,
            lambda values: (
                values[0].at[slot].set(True),
                values[1].at[slot].add(weight),
            ),
            lambda values: values,
            (accepted, coefficients),
        ), None

    (accepted, coefficients), _ = jax.lax.scan(
        mark,
        (cache.accepted, cache.coefficients),
        (formula_slots, formula_weights),
    )
    return cache._replace(accepted=accepted, coefficients=coefficients)


def _insert_admissible_frontier(
    state: _SparseAdaptiveState,
) -> _SparseAdaptiveState:
    dimension = state.accepted_indices.shape[1]

    def accepted_row(current, row):
        index, accepted_active = row

        def axis_row(frontier_carry, axis):
            candidate = index.at[axis].add(1)
            present = _index_membership(
                candidate,
                current.accepted_indices,
                current.accepted_active,
            ) | _index_membership(
                candidate,
                frontier_carry.frontier_indices,
                frontier_carry.frontier_active,
            )
            admissible = _jax_is_admissible(
                candidate,
                current.accepted_indices,
                current.accepted_active,
            )
            should_insert = accepted_active & admissible & ~present
            free = jnp.asarray(
                jnp.argmax(~frontier_carry.frontier_active),
                dtype=jnp.int32,
            )
            return jax.lax.cond(
                should_insert,
                lambda operand: operand._replace(
                    frontier_indices=operand.frontier_indices.at[free].set(candidate),
                    frontier_active=operand.frontier_active.at[free].set(True),
                    frontier_evaluated=operand.frontier_evaluated.at[free].set(False),
                ),
                lambda operand: operand,
                frontier_carry,
            ), None

        updated, _ = jax.lax.scan(
            axis_row,
            current,
            jnp.arange(dimension, dtype=jnp.int32),
        )
        return updated, None

    state, _ = jax.lax.scan(
        accepted_row,
        state,
        (state.accepted_indices, state.accepted_active),
    )
    return state


def _refresh_sparse_frontier(
    fun,
    domain,
    *,
    args,
    measure,
    tables,
    state,
    zero,
    max_evaluations,
    max_nodes,
    error_norm,
) -> _SparseAdaptiveState:
    state = _insert_admissible_frontier(state)

    def evaluate_row(current, slot):
        should_evaluate = (
            current.frontier_active[slot] & ~current.frontier_evaluated[slot]
        )

        def evaluate_candidate(operand):
            cache, surplus, new_cost, _ = _evaluate_sparse_formula(
                fun,
                domain,
                args=args,
                measure=measure,
                index=operand.frontier_indices[slot],
                tables=tables,
                cache=operand.cache,
                zero=zero,
                max_evaluations=max_evaluations,
                max_nodes=max_nodes,
            )
            norm = reduce_error_norm(surplus, error_norm)
            return operand._replace(
                cache=cache,
                frontier_evaluated=operand.frontier_evaluated.at[slot].set(True),
                frontier_surplus=operand.frontier_surplus.at[slot].set(surplus),
                frontier_norm=operand.frontier_norm.at[slot].set(norm),
                frontier_new_cost=operand.frontier_new_cost.at[slot].set(new_cost),
            )

        return jax.lax.cond(
            should_evaluate,
            evaluate_candidate,
            lambda operand: operand,
            current,
        ), None

    state, _ = jax.lax.scan(
        evaluate_row,
        state,
        jnp.arange(state.frontier_indices.shape[0], dtype=jnp.int32),
    )

    def recount_row(costs, slot):
        _, _, identities, formula_active = _formula_layout(
            state.frontier_indices[slot],
            tables,
            max_nodes,
        )

        def identity_is_new(identity, active):
            found, cache_slot = _cache_lookup(state.cache, identity)
            return active & found & ~state.cache.accepted[cache_slot]

        count = jnp.sum(
            jax.vmap(identity_is_new)(identities, formula_active),
            dtype=jnp.int32,
        )
        count = jnp.where(state.frontier_active[slot], count, 0)
        return costs.at[slot].set(count), None

    costs, _ = jax.lax.scan(
        recount_row,
        state.frontier_new_cost,
        jnp.arange(state.frontier_indices.shape[0], dtype=jnp.int32),
    )
    return state._replace(
        frontier_new_cost=costs,
        evaluations=state.cache.evaluations,
    )


def _select_jax_profit(state: _SparseAdaptiveState) -> Array:
    selectable = state.frontier_active & (state.frontier_new_cost > 0)
    profit = jnp.where(
        selectable,
        state.frontier_norm / jnp.maximum(state.frontier_new_cost, 1),
        -jnp.inf,
    )
    best = jnp.max(profit)
    tied = selectable & (profit == best)
    max_component = jnp.iinfo(jnp.int32).max
    candidates = jnp.where(
        tied[:, None],
        state.frontier_indices,
        max_component,
    )
    chosen = jnp.asarray(0, dtype=jnp.int32)
    alive = tied
    for axis in range(state.frontier_indices.shape[1]):
        minimum = jnp.min(jnp.where(alive, candidates[:, axis], max_component))
        alive = alive & (candidates[:, axis] == minimum)
        chosen = jnp.asarray(jnp.argmax(alive), dtype=jnp.int32)
    return chosen


def adaptive_sparse_controller(
    fun,
    domain,
    *,
    args,
    measure,
    initial_indices: tuple[SparseIndex, ...],
    epsabs,
    epsrel,
    max_evaluations: int,
    max_indices: int,
    max_frontier: int,
    max_nodes: int,
    error_norm: ErrorNorm,
    zero: Array,
) -> AdaptiveSparseControllerResult:
    """Run the fixed-capacity dimension-adaptive Smolyak frontier scan."""
    dimension = domain.dimension
    dtype = jnp.result_type(domain.lower, domain.upper, 0.0)
    tables = _adaptive_sparse_tables(max_nodes, dtype)
    value_dtype = jnp.result_type(zero, tables.points)
    zero = jnp.asarray(zero, dtype=value_dtype)
    initial_count = len(initial_indices)
    accepted_indices = jnp.ones((max_indices, dimension), dtype=jnp.int32)
    accepted_indices = accepted_indices.at[:initial_count].set(
        jnp.asarray(initial_indices, dtype=jnp.int32)
    )
    accepted_active = jnp.arange(max_indices) < initial_count
    cache = _SparseCache(
        identities=jnp.full((max_nodes, 2 * dimension), -1, dtype=jnp.int32),
        values=jnp.zeros((max_nodes,) + zero.shape, dtype=value_dtype),
        active=jnp.zeros((max_nodes,), dtype=jnp.bool_),
        accepted=jnp.zeros((max_nodes,), dtype=jnp.bool_),
        coefficients=jnp.zeros((max_nodes,), dtype=dtype),
        evaluations=jnp.asarray(0, dtype=jnp.int32),
        nonfinite=jnp.asarray(False),
        exhausted=jnp.asarray(False),
    )
    state = _SparseAdaptiveState(
        accepted_indices=accepted_indices,
        accepted_active=accepted_active,
        accepted_count=jnp.asarray(initial_count, dtype=jnp.int32),
        frontier_indices=jnp.ones((max_frontier, dimension), dtype=jnp.int32),
        frontier_active=jnp.zeros((max_frontier,), dtype=jnp.bool_),
        frontier_evaluated=jnp.zeros((max_frontier,), dtype=jnp.bool_),
        frontier_surplus=jnp.zeros(
            (max_frontier,) + zero.shape,
            dtype=value_dtype,
        ),
        frontier_norm=jnp.zeros((max_frontier,), dtype=dtype),
        frontier_new_cost=jnp.zeros((max_frontier,), dtype=jnp.int32),
        cache=cache,
        value=zero,
        evaluations=jnp.asarray(0, dtype=jnp.int32),
        refinements=jnp.asarray(0, dtype=jnp.int32),
        status=jnp.asarray(RUNNING, dtype=jnp.int32),
        done=jnp.asarray(False),
    )

    def initialize_index(current, slot):
        active = current.accepted_active[slot]

        def evaluate_initial(operand):
            cache_after, surplus, _, formula_slots = _evaluate_sparse_formula(
                fun,
                domain,
                args=args,
                measure=measure,
                index=operand.accepted_indices[slot],
                tables=tables,
                cache=operand.cache,
                zero=zero,
                max_evaluations=max_evaluations,
                max_nodes=max_nodes,
            )
            _, formula_weights, _, _ = _formula_layout(
                operand.accepted_indices[slot],
                tables,
                max_nodes,
            )
            cache_after = _mark_formula_accepted(
                cache_after,
                formula_slots,
                formula_weights,
            )
            return operand._replace(
                cache=cache_after,
                value=operand.value + surplus,
                evaluations=cache_after.evaluations,
            )

        return jax.lax.cond(
            active,
            evaluate_initial,
            lambda operand: operand,
            current,
        ), None

    state, _ = jax.lax.scan(
        initialize_index,
        state,
        jnp.arange(max_indices, dtype=jnp.int32),
    )
    state = _refresh_sparse_frontier(
        fun,
        domain,
        args=args,
        measure=measure,
        tables=tables,
        state=state,
        zero=zero,
        max_evaluations=max_evaluations,
        max_nodes=max_nodes,
        error_norm=error_norm,
    )

    def update_status(current: _SparseAdaptiveState) -> _SparseAdaptiveState:
        frontier_error = jnp.sum(
            jnp.where(current.frontier_active, current.frontier_norm, 0.0)
        )
        tolerance = tolerance_threshold(
            current.value,
            epsabs=epsabs,
            epsrel=epsrel,
            norm=error_norm,
        )
        any_frontier = jnp.any(current.frontier_active)
        all_roundoff = any_frontier & jnp.all(
            jnp.where(
                current.frontier_active,
                current.frontier_new_cost == 0,
                True,
            )
        )
        nonfinite = (
            current.cache.nonfinite
            | ~jnp.all(jnp.isfinite(current.value))
            | ~jnp.all(jnp.isfinite(current.frontier_surplus))
            | ~jnp.isfinite(frontier_error)
            | ~jnp.isfinite(tolerance)
        )
        status = sparse_termination_status(
            invalid=False,
            nonfinite=nonfinite,
            converged=frontier_error <= tolerance,
            all_active_roundoff=all_roundoff & (frontier_error > tolerance),
            evaluation_exhausted=current.cache.exhausted,
            index_exhausted=(current.accepted_count >= max_indices)
            & (frontier_error > tolerance),
        )
        return current._replace(
            status=status,
            done=status != RUNNING,
        )

    state = update_status(state)

    def step(current: _SparseAdaptiveState, _):
        def accept_best(operand):
            slot = _select_jax_profit(operand)
            index = operand.frontier_indices[slot]
            surplus = operand.frontier_surplus[slot]
            cache_after, _, _, formula_slots = _evaluate_sparse_formula(
                fun,
                domain,
                args=args,
                measure=measure,
                index=index,
                tables=tables,
                cache=operand.cache,
                zero=zero,
                max_evaluations=max_evaluations,
                max_nodes=max_nodes,
            )
            _, formula_weights, _, _ = _formula_layout(
                index,
                tables,
                max_nodes,
            )
            cache_after = _mark_formula_accepted(
                cache_after,
                formula_slots,
                formula_weights,
            )
            accepted_slot = operand.accepted_count
            next_state = operand._replace(
                accepted_indices=operand.accepted_indices.at[accepted_slot].set(index),
                accepted_active=operand.accepted_active.at[accepted_slot].set(True),
                accepted_count=operand.accepted_count + 1,
                frontier_active=operand.frontier_active.at[slot].set(False),
                frontier_evaluated=operand.frontier_evaluated.at[slot].set(False),
                cache=cache_after,
                value=operand.value + surplus,
                refinements=operand.refinements + 1,
            )
            next_state = _refresh_sparse_frontier(
                fun,
                domain,
                args=args,
                measure=measure,
                tables=tables,
                state=next_state,
                zero=zero,
                max_evaluations=max_evaluations,
                max_nodes=max_nodes,
                error_norm=error_norm,
            )
            return update_status(next_state)

        next_state = jax.lax.cond(
            current.done,
            lambda operand: operand,
            accept_best,
            current,
        )
        return next_state, None

    state, _ = jax.lax.scan(step, state, xs=None, length=max_indices)
    frontier_error = jnp.sum(jnp.where(state.frontier_active, state.frontier_norm, 0.0))
    error_shape = (max_frontier,) + (1,) * zero.ndim
    error = jnp.sum(
        jnp.where(
            state.frontier_active.reshape(error_shape),
            jnp.abs(state.frontier_surplus),
            jnp.zeros_like(jnp.abs(state.frontier_surplus)),
        ),
        axis=0,
    )
    tolerance = tolerance_threshold(
        state.value,
        epsabs=epsabs,
        epsrel=epsrel,
        norm=error_norm,
    )
    return AdaptiveSparseControllerResult(
        value=state.value,
        error=error,
        frontier_error=frontier_error,
        tolerance=tolerance,
        status=state.status,
        evaluations=state.evaluations,
        refinements=state.refinements,
        level=jnp.max(
            jnp.where(
                state.accepted_active[:, None],
                state.accepted_indices,
                0,
            )
        ),
        evidence=SparseReplayEvidence(
            indices=state.accepted_indices,
            active=state.accepted_active,
            node_ids=state.cache.identities,
            coefficients=state.cache.coefficients,
            node_active=state.cache.active & state.cache.accepted,
        ),
    )


__all__ = [
    "DyadicIdentity",
    "HierarchicalRule",
    "RUNNING",
    "AdaptiveSparseControllerResult",
    "SparseReplayEvidence",
    "SparseIndex",
    "SparseNodeIdentity",
    "SparseRuleData",
    "canonical_cc_identity",
    "adaptive_sparse_controller",
    "admissible_forward_neighbors",
    "fixed_sparse_node_identities",
    "fixed_index_set",
    "hierarchical_rule",
    "identities_to_points",
    "identity_to_point",
    "is_admissible",
    "materialize_smolyak_rule",
    "smolyak_host_data",
    "smolyak_rule_data",
    "required_frontier_capacity",
    "select_profit",
    "sparse_termination_status",
    "sparse_axis_identities",
    "unit_clenshaw_curtis",
]
