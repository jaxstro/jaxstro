"""Nested fixed tanh-sinh formulas on the reference interval."""

import jax.numpy as jnp

from .rules import FixedRuleData, TanhSinhRule


def tanh_sinh_rule_data(rule: TanhSinhRule) -> FixedRuleData:
    """Construct a nested double-exponential formula on ``(-1, 1)``."""
    dtype = jnp.asarray(0.0).dtype
    step = jnp.asarray(2.0**-rule.level, dtype=dtype)
    extent = 3 * 2**rule.level
    index = jnp.arange(-extent, extent + 1, dtype=dtype)
    parameter = step * index
    transformed = 0.5 * jnp.pi * jnp.sinh(parameter)
    raw_nodes = jnp.tanh(transformed)
    endpoint = jnp.nextafter(
        jnp.asarray(1.0, dtype=dtype), jnp.asarray(0.0, dtype=dtype)
    )
    nodes = jnp.clip(raw_nodes, -endpoint, endpoint)
    weights = step * 0.5 * jnp.pi * jnp.cosh(parameter) / jnp.cosh(transformed) ** 2
    return FixedRuleData(
        nodes=nodes,
        weights=weights,
        degree=-1,
        nested=True,
    )


__all__ = ["tanh_sinh_rule_data"]
