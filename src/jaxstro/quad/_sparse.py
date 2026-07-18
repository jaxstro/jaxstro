"""Exact identities and hierarchical rules for Smolyak integration."""

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


def canonical_cc_identity(level: int, index: int) -> DyadicIdentity:
    """Reduce one Clenshaw-Curtis angle index to an exact dyadic identity."""
    if isinstance(level, bool) or not isinstance(level, int) or level < 0:
        raise ValueError("Clenshaw-Curtis level must be a nonnegative integer")
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("Clenshaw-Curtis index must be an integer")
    denominator = 1 << level
    if index < 0 or index > denominator:
        raise ValueError("Clenshaw-Curtis index is outside its level")
    if index == 0:
        return 0, 0
    while level > 0 and index % 2 == 0:
        index //= 2
        level -= 1
    return index, level


def identity_to_point(identity: DyadicIdentity, dtype) -> Array:
    """Create a unit-interval coordinate after exact identity coalescing."""
    numerator, denominator_power = identity
    selected_dtype = jnp.dtype(dtype)
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
    data = chebyshev_rule_data(
        ClenshawCurtisRule((1 << level) + 1),
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


__all__ = [
    "DyadicIdentity",
    "HierarchicalRule",
    "canonical_cc_identity",
    "hierarchical_rule",
    "identity_to_point",
    "unit_clenshaw_curtis",
]
