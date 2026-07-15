"""Clenshaw-Curtis and Fejer fixed-rule contracts."""

import jax
import jax.numpy as jnp
import pytest

from jaxstro import quad
from jaxstro.numerics import quadrature as legacy_quadrature
from jaxstro.quad import ClenshawCurtisRule, FejerIIRule, FejerIRule
from jaxstro.quad._chebyshev import chebyshev_rule_data


@pytest.mark.parametrize(
    "rule",
    [ClenshawCurtisRule(9), FejerIRule(8), FejerIIRule(8)],
)
def test_chebyshev_rules_integrate_declared_polynomial_degree(rule) -> None:
    data = chebyshev_rule_data(rule)
    assert jnp.all(data.weights > 0.0)
    assert jnp.allclose(jnp.sum(data.weights), 2.0, atol=2e-13)
    for degree in range(data.degree + 1):
        expected = 0.0 if degree % 2 else 2.0 / (degree + 1)
        got = jnp.sum(data.weights * data.nodes**degree)
        assert jnp.allclose(got, expected, rtol=2e-11, atol=2e-11)


def test_clenshaw_curtis_is_nested_for_doubling_intervals() -> None:
    coarse = chebyshev_rule_data(ClenshawCurtisRule(5))
    fine = chebyshev_rule_data(ClenshawCurtisRule(9))
    assert coarse.nested is True
    assert jnp.allclose(coarse.nodes, fine.nodes[::2], atol=2e-15)


def test_fejer_rules_exclude_endpoints() -> None:
    for rule in (FejerIRule(8), FejerIIRule(8)):
        nodes = chebyshev_rule_data(rule).nodes
        assert jnp.all(jnp.abs(nodes) < 1.0)


def test_clenshaw_curtis_legacy_helper_is_canonical_identity() -> None:
    assert legacy_quadrature.clenshaw_curtis_nodes is quad.clenshaw_curtis_nodes
    assert quad.clenshaw_curtis_nodes.__module__ == "jaxstro.quad._chebyshev"


def test_chebyshev_rule_construction_compiles() -> None:
    construct = jax.jit(lambda: chebyshev_rule_data(FejerIRule(7)))
    data = construct()
    assert data.nodes.shape == (7,)
    assert data.weights.shape == (7,)
