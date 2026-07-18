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
from .rules import ClenshawCurtisRule, FixedRuleData

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


def identity_to_point(identity: DyadicIdentity, dtype) -> Array:
    """Create a unit-interval coordinate after exact identity coalescing."""
    numerator, denominator_power = identity
    selected_dtype = jnp.dtype(dtype)
    if identity == (0, 0):
        return jnp.asarray(0.0, dtype=selected_dtype)
    if identity == (1, 0):
        return jnp.asarray(1.0, dtype=selected_dtype)
    if identity == (1, 1):
        return jnp.asarray(0.5, dtype=selected_dtype)
    numerator_value = jnp.asarray(numerator, dtype=selected_dtype)
    denominator = jnp.asarray(1 << denominator_power, dtype=selected_dtype)
    return jnp.asarray(0.5, dtype=selected_dtype) * (
        jnp.asarray(1.0, dtype=selected_dtype)
        - jnp.cos(jnp.pi * numerator_value / denominator)
    )


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


__all__ = [
    "DyadicIdentity",
    "HierarchicalRule",
    "SparseIndex",
    "SparseNodeIdentity",
    "SparseRuleData",
    "canonical_cc_identity",
    "fixed_sparse_node_identities",
    "fixed_index_set",
    "hierarchical_rule",
    "identity_to_point",
    "materialize_smolyak_rule",
    "smolyak_host_data",
    "smolyak_rule_data",
    "sparse_axis_identities",
    "unit_clenshaw_curtis",
]
