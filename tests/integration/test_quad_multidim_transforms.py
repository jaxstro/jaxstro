import jax
import jax.numpy as jnp

from jaxstro import quad
from jaxstro.quad._multidim import map_hyperrectangle


def test_map_composes_with_jit_and_vmap_over_bounds():
    reference = jnp.array([[0.25, 0.75]])

    @jax.jit
    def mapped(lower, upper):
        return map_hyperrectangle(
            quad.Hyperrectangle(lower, upper),
            reference,
        ).x

    lowers = jnp.array([[0.0, 0.0], [1.0, -1.0]])
    uppers = jnp.array([[1.0, 2.0], [3.0, 1.0]])
    values = jax.vmap(mapped)(lowers, uppers)
    assert values.shape == (2, 1, 2)


def test_traced_nonfinite_bounds_fail_dynamically():
    valid = jax.jit(
        lambda upper: quad.hyperrectangle_is_valid(
            quad.Hyperrectangle(jnp.zeros(2), upper)
        )
    )
    assert not valid(jnp.array([1.0, jnp.inf]))
