"""Shared Chebyshev interpolation substrate for fixed quadrature rules."""

import jax.numpy as jnp
from jaxtyping import Array

from .rules import (
    ClenshawCurtisRule,
    FejerIIRule,
    FejerIRule,
    FixedRuleData,
)


def _clenshaw_curtis_weights(theta: Array) -> Array:
    """Evaluate Trefethen's stable cosine series for Lobatto nodes."""
    intervals = theta.shape[0] - 1
    if intervals == 0:
        return jnp.asarray([2.0], dtype=theta.dtype)

    interior_theta = theta[1:-1]
    interior = jnp.ones_like(interior_theta)
    if intervals % 2 == 0:
        endpoint = 1.0 / (intervals**2 - 1.0)
        for index in range(1, intervals // 2):
            interior = interior - 2.0 * jnp.cos(2.0 * index * interior_theta) / (
                4.0 * index**2 - 1.0
            )
        interior = interior - jnp.cos(intervals * interior_theta) / (intervals**2 - 1.0)
    else:
        endpoint = 1.0 / intervals**2
        for index in range(1, (intervals + 1) // 2):
            interior = interior - 2.0 * jnp.cos(2.0 * index * interior_theta) / (
                4.0 * index**2 - 1.0
            )
    interior = 2.0 * interior / intervals
    return jnp.concatenate(
        (
            jnp.asarray([endpoint], dtype=theta.dtype),
            interior,
            jnp.asarray([endpoint], dtype=theta.dtype),
        )
    )


def _fejer_i_weights(theta: Array) -> Array:
    """Evaluate Fejer type I weights from exact Chebyshev moments."""
    order = theta.shape[0]
    degree = jnp.arange(1, order, dtype=theta.dtype)
    moments = jnp.where(
        degree % 2 == 0,
        2.0 / (1.0 - degree**2),
        0.0,
    )
    series = jnp.sum(
        moments[:, None] * jnp.cos(degree[:, None] * theta[None, :]), axis=0
    )
    return (2.0 + 2.0 * series) / order


def _fejer_ii_weights(theta: Array) -> Array:
    """Evaluate Fejer type II weights from its odd sine series."""
    order = theta.shape[0]
    index = jnp.arange(1, (order + 1) // 2 + 1, dtype=theta.dtype)
    odd = 2.0 * index - 1.0
    series = jnp.sum(jnp.sin(odd[:, None] * theta[None, :]) / odd[:, None], axis=0)
    return 4.0 * jnp.sin(theta) * series / (order + 1.0)


def chebyshev_rule_data(
    rule: ClenshawCurtisRule | FejerIRule | FejerIIRule,
) -> FixedRuleData:
    """Construct an interpolatory rule on a Chebyshev point family."""
    dtype = jnp.asarray(0.0).dtype
    index = jnp.arange(rule.order, dtype=dtype)
    if isinstance(rule, ClenshawCurtisRule):
        if rule.order == 1:
            theta = jnp.zeros((1,), dtype=dtype)
            nodes = jnp.zeros((1,), dtype=dtype)
        else:
            theta = jnp.pi * index / (rule.order - 1)
            nodes = jnp.cos(theta)
        weights = _clenshaw_curtis_weights(theta)
        nested = True
    elif isinstance(rule, FejerIRule):
        theta = jnp.pi * (2.0 * index + 1.0) / (2.0 * rule.order)
        weights = _fejer_i_weights(theta)
        nested = False
    elif isinstance(rule, FejerIIRule):
        theta = jnp.pi * (index + 1.0) / (rule.order + 1.0)
        weights = _fejer_ii_weights(theta)
        nested = False
    else:
        raise TypeError("unsupported Chebyshev rule")
    if not isinstance(rule, ClenshawCurtisRule):
        nodes = jnp.cos(theta)
    return FixedRuleData(
        nodes=nodes,
        weights=weights,
        degree=rule.order - 1,
        nested=nested,
    )


def clenshaw_curtis_nodes(n: int) -> tuple[Array, Array]:
    """Return Clenshaw-Curtis nodes and weights on ``[-1, 1]``."""
    data = chebyshev_rule_data(ClenshawCurtisRule(n))
    return data.nodes, data.weights


__all__ = ["chebyshev_rule_data", "clenshaw_curtis_nodes"]
