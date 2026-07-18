import jax.numpy as jnp
import pytest

from jaxstro.quad import ClenshawCurtisRule
from jaxstro.quad._chebyshev import chebyshev_rule_data
from jaxstro.quad._sparse import (
    canonical_cc_identity,
    hierarchical_rule,
    identity_to_point,
    unit_clenshaw_curtis,
)


def test_same_nested_node_has_same_reduced_identity():
    assert canonical_cc_identity(2, 1) == canonical_cc_identity(3, 2)
    assert canonical_cc_identity(2, 2) == canonical_cc_identity(4, 8)
    assert canonical_cc_identity(4, 0) == (0, 0)
    assert canonical_cc_identity(4, 16) == (1, 0)


@pytest.mark.parametrize(
    ("level", "index"),
    ((-1, 0), (1, -1), (1, 3), (True, 0), (1.5, 1)),
)
def test_canonical_identity_rejects_invalid_level_or_index(level, index):
    with pytest.raises(ValueError):
        canonical_cc_identity(level, index)


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
@pytest.mark.parametrize("level", range(1, 9))
def test_unit_rule_matches_phase_a_values_without_changing_dtype(level, dtype):
    sparse = unit_clenshaw_curtis(level, dtype)
    phase_a = chebyshev_rule_data(
        ClenshawCurtisRule((1 << level) + 1),
        dtype=dtype,
    )

    assert sparse.nodes.dtype == dtype
    assert sparse.weights.dtype == dtype
    assert jnp.array_equal(sparse.nodes, 0.5 * (1.0 - phase_a.nodes))
    assert jnp.array_equal(sparse.weights, 0.5 * phase_a.weights)
    assert jnp.allclose(
        jnp.sum(sparse.weights),
        1.0,
        rtol=0.0,
        atol=8.0 * jnp.finfo(dtype).eps,
    )


def test_identity_to_point_owns_endpoint_and_midpoint_values():
    assert identity_to_point((0, 0), jnp.float64) == 0.0
    assert identity_to_point((1, 0), jnp.float64) == 1.0
    assert jnp.allclose(
        identity_to_point((1, 1), jnp.float64),
        0.5,
        rtol=0.0,
        atol=jnp.finfo(jnp.float64).eps,
    )


def test_hierarchical_difference_annihilates_constant_after_level_one():
    for level in range(2, 7):
        rule = hierarchical_rule(level, jnp.float64)
        assert jnp.allclose(jnp.sum(rule.weights), 0.0, atol=2e-14)
        assert len(rule.identities) == rule.points.shape[0]


def test_hierarchical_weights_coalesce_before_float_conversion():
    rule = hierarchical_rule(4, jnp.float64)
    assert len(set(rule.identities)) == len(rule.identities)


def test_hierarchical_identity_sets_are_nested_and_keep_exact_endpoints():
    previous = set()
    for level in range(1, 9):
        rule = hierarchical_rule(level, jnp.float64)
        current = set(rule.identities)
        assert previous <= current
        assert (0, 0) in current
        assert (1, 0) in current
        previous = current


def test_hierarchical_rule_drops_only_exact_zero_weights():
    rule = hierarchical_rule(6, jnp.float64)
    assert jnp.all(rule.weights != 0.0)
