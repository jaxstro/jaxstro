import jax
import jax.numpy as jnp
import pytest

from jaxstro import quad
from jaxstro.quad._multidim import map_hyperrectangle


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


def test_hyperrectangle_normalizes_sequence_bounds_to_dynamic_array_leaves():
    domain = quad.Hyperrectangle([0.0, 3.0], (2.0, -1.0))
    leaves, tree = jax.tree.flatten(domain)
    rebuilt = jax.tree.unflatten(tree, leaves)

    assert domain.dimension == 2
    assert len(leaves) == 2
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    assert jnp.array_equal(rebuilt.lower, jnp.array([0.0, 3.0]))
    assert jnp.array_equal(rebuilt.upper, jnp.array([2.0, -1.0]))
    assert quad.hyperrectangle_is_valid(rebuilt)
    assert quad.hyperrectangle_orientation(rebuilt) == -1.0


@pytest.mark.parametrize(
    "lower, upper, expected_orientation",
    [
        (
            jnp.array([3, -2], dtype=jnp.int32),
            jnp.array([-1, 4], dtype=jnp.int32),
            -1.0,
        ),
        (
            jnp.array([5, 0], dtype=jnp.uint32),
            jnp.array([1, 4], dtype=jnp.uint32),
            -1.0,
        ),
    ],
)
def test_hyperrectangle_normalizes_integer_bounds_before_orientation(
    lower, upper, expected_orientation
):
    domain = quad.Hyperrectangle(lower, upper)

    assert jnp.issubdtype(domain.lower.dtype, jnp.floating)
    assert domain.lower.dtype == domain.upper.dtype
    assert quad.hyperrectangle_orientation(domain) == expected_orientation


def test_hyperrectangle_avoids_integer_overflow_in_large_volume():
    domain = quad.Hyperrectangle(
        jnp.array([0, 0], dtype=jnp.int32),
        jnp.array([100_000, 100_000], dtype=jnp.int32),
    )

    mapped = map_hyperrectangle(domain, jnp.array([[0.5, 0.5]]))

    assert jnp.issubdtype(domain.lower.dtype, jnp.floating)
    assert mapped.jacobian == 10_000_000_000.0


def test_hyperrectangle_normalizes_mixed_numeric_bounds_to_one_dtype():
    domain = quad.Hyperrectangle(
        jnp.array([0, -2], dtype=jnp.int32),
        jnp.array([1.5, 4.0], dtype=jnp.float64),
    )

    assert domain.lower.dtype == jnp.dtype(jnp.float64)
    assert domain.upper.dtype == domain.lower.dtype
    assert jnp.array_equal(domain.lower, jnp.array([0.0, -2.0]))
    assert jnp.array_equal(domain.upper, jnp.array([1.5, 4.0]))


@pytest.mark.parametrize(
    "lower, upper",
    [
        (
            jnp.array([False, True]),
            jnp.array([True, True]),
        ),
        (
            jnp.array([0.0 + 0.0j, 1.0 + 0.0j]),
            jnp.array([1.0 + 0.0j, 2.0 + 0.0j]),
        ),
    ],
)
def test_hyperrectangle_rejects_nonreal_numeric_bounds(lower, upper):
    with pytest.raises(TypeError, match="real numeric"):
        quad.Hyperrectangle(lower, upper)


def test_hyperrectangle_normalization_preserves_jit_behavior():
    @jax.jit
    def orientation(lower, upper):
        return quad.hyperrectangle_orientation(quad.Hyperrectangle(lower, upper))

    assert orientation(jnp.array([0.0, 3.0]), jnp.array([2.0, -1.0])) == -1.0
    assert (
        orientation(
            jnp.array([5, 0], dtype=jnp.uint32),
            jnp.array([1, 4], dtype=jnp.uint32),
        )
        == -1.0
    )


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
