"""Static fixed-rule configuration contracts."""

import jax
import jax.numpy as jnp
import pytest

from jaxstro.quad import (
    ClenshawCurtisRule,
    FejerIIRule,
    FejerIRule,
    GaussianRule,
    TanhSinhRule,
)
from jaxstro.quad.rules import FixedRuleData


@pytest.mark.parametrize(
    "factory",
    [GaussianRule, ClenshawCurtisRule, FejerIRule, FejerIIRule],
)
def test_rule_order_is_positive_static_metadata(factory) -> None:
    rule = factory(5)
    leaves, treedef = jax.tree.flatten(rule)
    assert leaves == []
    assert jax.tree.unflatten(treedef, leaves) == rule
    with pytest.raises(ValueError, match="positive integer"):
        factory(0)
    with pytest.raises(ValueError, match="positive integer"):
        factory(True)


def test_tanh_sinh_level_is_nonnegative_static_metadata() -> None:
    rule = TanhSinhRule(3)
    leaves, treedef = jax.tree.flatten(rule)
    assert leaves == []
    assert jax.tree.unflatten(treedef, leaves) == rule
    with pytest.raises(ValueError, match="nonnegative integer"):
        TanhSinhRule(-1)
    with pytest.raises(ValueError, match="nonnegative integer"):
        TanhSinhRule(False)


def test_fixed_rule_data_is_a_jax_array_record() -> None:
    data = FixedRuleData(
        nodes=jnp.asarray([-1.0, 1.0]),
        weights=jnp.asarray([1.0, 1.0]),
        degree=1,
        nested=False,
    )
    leaves = jax.tree.leaves(data)
    assert len(leaves) == 4
    assert jnp.array_equal(data.nodes, jnp.asarray([-1.0, 1.0]))
