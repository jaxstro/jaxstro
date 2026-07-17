import jax
import jax.numpy as jnp
import pytest

from jaxstro import quad


def test_hyperrectangle_is_a_dynamic_bound_pytree():
    domain = quad.Hyperrectangle(
        jnp.array([0.0, 3.0]),
        jnp.array([2.0, -1.0]),
    )
    leaves, tree = jax.tree.flatten(domain)
    rebuilt = jax.tree.unflatten(tree, leaves)

    assert domain.dimension == 2
    assert len(leaves) == 2
    assert jnp.array_equal(rebuilt.lower, domain.lower)
    assert jnp.array_equal(rebuilt.upper, domain.upper)
    assert quad.hyperrectangle_orientation(domain) == -1.0
    assert quad.hyperrectangle_is_valid(domain)


def test_hyperrectangle_zero_volume_is_valid_but_has_zero_orientation():
    domain = quad.Hyperrectangle(
        jnp.array([0.0, 1.0]),
        jnp.array([2.0, 1.0]),
    )
    assert quad.hyperrectangle_is_valid(domain)
    assert quad.hyperrectangle_orientation(domain) == 0.0


@pytest.mark.parametrize(
    "lower, upper, error",
    [
        (jnp.zeros((2, 1)), jnp.ones((2, 1)), "one-dimensional"),
        (jnp.zeros(2), jnp.ones(3), "matching shapes"),
        (jnp.zeros(0), jnp.ones(0), "positive dimension"),
    ],
)
def test_hyperrectangle_rejects_invalid_static_shapes(lower, upper, error):
    with pytest.raises(ValueError, match=error):
        quad.Hyperrectangle(lower, upper)


def test_hyperrectangle_rejects_host_known_nonfinite_bounds():
    with pytest.raises(ValueError, match="must be finite"):
        quad.Hyperrectangle(jnp.array([0.0, jnp.inf]), jnp.ones(2))
