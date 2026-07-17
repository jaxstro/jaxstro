"""Fixed tensor-product rule construction on the unit hyperrectangle."""

import math
from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array

from ._chebyshev import chebyshev_rule_data
from ._recurrence import gaussian_rule_data
from ._tanh_sinh import tanh_sinh_rule_data
from .measures import LebesgueMeasure
from .rules import (
    ClenshawCurtisRule,
    FejerIIRule,
    FejerIRule,
    GaussianRule,
    TanhSinhRule,
)


class TensorRuleData(NamedTuple):
    points: Array
    weights: Array
    point_count: int


def validate_b1_dimension(dimension: int) -> None:
    if dimension < 2 or dimension > 8:
        raise ValueError("Phase B1 deterministic methods require dimension 2 through 8")


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
    if isinstance(rule, GaussianRule):
        data = gaussian_rule_data(rule, LebesgueMeasure())
        exact_constant = True
    elif isinstance(rule, (ClenshawCurtisRule, FejerIRule, FejerIIRule)):
        data = chebyshev_rule_data(rule, dtype=dtype)
        exact_constant = True
    elif isinstance(rule, TanhSinhRule):
        data = tanh_sinh_rule_data(rule)
        exact_constant = False
    else:
        raise TypeError(f"unsupported tensor rule: {type(rule).__name__}")
    nodes = jnp.asarray(data.nodes, dtype=dtype)
    weights = 0.5 * jnp.asarray(data.weights, dtype=dtype)
    if exact_constant:
        weights = weights.at[-1].add(1.0 - jnp.sum(weights))
    return 0.5 * (nodes + 1.0), weights


def _rule_point_count(rule, dtype) -> int:
    if isinstance(rule, (GaussianRule, ClenshawCurtisRule, FejerIRule, FejerIIRule)):
        return rule.order
    if isinstance(rule, TanhSinhRule):
        return tanh_sinh_rule_data(rule).nodes.size
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


__all__ = ["TensorRuleData", "tensor_rule_data", "validate_b1_dimension"]
