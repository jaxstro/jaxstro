"""Fixed tensor-product rule construction on the unit hyperrectangle."""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ._chebyshev import chebyshev_rule_data
from ._recurrence import gaussian_rule_data
from ._tanh_sinh import tanh_sinh_rule_data, tanh_sinh_rule_point_count
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


__all__ = [
    "TensorRuleData",
    "tensor_point_count",
    "tensor_rule_data",
    "validate_b1_dimension",
]
