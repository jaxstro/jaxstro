"""Shared Chebyshev interpolation substrate for fixed quadrature rules."""

import jax.numpy as jnp
from jaxtyping import Array

from .rules import (
    ClenshawCurtisRule,
    FejerIIRule,
    FejerIRule,
    FixedRuleData,
)


def _chebyshev_moments(order: int, dtype) -> Array:
    degree = jnp.arange(order, dtype=dtype)
    return jnp.where(
        degree % 2 == 0,
        2.0 / (1.0 - degree**2),
        0.0,
    )


def _interpolatory_weights(theta: Array) -> Array:
    """Invert the well-conditioned cosine interpolation transform."""
    order = theta.shape[0]
    degree = jnp.arange(order, dtype=theta.dtype)
    transform = jnp.cos(degree[:, None] * theta[None, :])
    return jnp.linalg.solve(transform, _chebyshev_moments(order, theta.dtype))


def chebyshev_rule_data(
    rule: ClenshawCurtisRule | FejerIRule | FejerIIRule,
) -> FixedRuleData:
    """Construct an interpolatory rule on a Chebyshev point family."""
    dtype = jnp.asarray(0.0).dtype
    index = jnp.arange(rule.order, dtype=dtype)
    if isinstance(rule, ClenshawCurtisRule):
        if rule.order == 1:
            theta = jnp.zeros((1,), dtype=dtype)
        else:
            theta = jnp.pi * index / (rule.order - 1)
        nested = True
    elif isinstance(rule, FejerIRule):
        theta = jnp.pi * (2.0 * index + 1.0) / (2.0 * rule.order)
        nested = False
    elif isinstance(rule, FejerIIRule):
        theta = jnp.pi * (index + 1.0) / (rule.order + 1.0)
        nested = False
    else:
        raise TypeError("unsupported Chebyshev rule")
    nodes = jnp.cos(theta)
    weights = _interpolatory_weights(theta)
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
