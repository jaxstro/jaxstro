"""JAX transform validation for quantities."""

import jax
import jax.numpy as jnp

from jaxstro import quantity as q


def test_jit_through_unit_construction_and_conversion():
    convert = jax.jit(lambda x: (x * q.cm).to_value(q.m))

    assert convert(250.0) == 2.5


def test_vmap_over_quantity_values():
    convert = jax.vmap(lambda x: (x * q.km / q.s).to_value(q.cm / q.s))

    assert jnp.allclose(convert(jnp.array([1.0, 2.0])), jnp.array([1.0e5, 2.0e5]))


def test_grad_through_arithmetic_and_scale_factors():
    def loss(x):
        radius = x * q.m
        return (radius.to_value(q.cm) ** 2) / 100.0

    assert jax.grad(loss)(2.0) == 400.0


def test_unit_metadata_is_static_auxiliary_data():
    leaves, treedef = jax.tree_util.tree_flatten(jnp.array([1.0, 2.0]) * q.cm)

    assert len(leaves) == 1
    assert jnp.all(leaves[0] == jnp.array([1.0, 2.0]))
    assert "cm" in str(treedef)
