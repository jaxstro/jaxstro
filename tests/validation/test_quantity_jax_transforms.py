"""JAX transform validation for quantities."""

import jax
import jax.numpy as jnp
import pytest

from jaxstro import quantity as q
from jaxstro.quantity.errors import DimensionError


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


@pytest.mark.parametrize("first", [q.rad, q.dimensionless])
def test_jit_sin_rechecks_angle_semantics_for_each_static_unit(first):
    sin = jax.jit(q.math.sin)
    second = q.dimensionless if first is q.rad else q.rad

    for unit in (first, second):
        if unit is q.rad:
            assert jnp.allclose(sin(1.0 * unit).value, jnp.sin(1.0))
        else:
            with pytest.raises(DimensionError, match="tagged angle"):
                sin(1.0 * unit)


def test_jit_scaled_dimensionless_log_uses_canonical_magnitude():
    percent = q.Unit("percent", 0.01, q.dimensionless.dimensions)

    result = jax.jit(q.math.log)(50.0 * percent)
    assert jnp.allclose(result.value, jnp.log(0.5))


@pytest.mark.parametrize("first", [q.rad, q.dimensionless])
def test_jit_log_rejects_angle_after_either_cache_order(first):
    log = jax.jit(q.math.log)
    second = q.dimensionless if first is q.rad else q.rad

    for unit in (first, second):
        if unit is q.dimensionless:
            assert jnp.allclose(log(1.0 * unit).value, 0.0)
        else:
            with pytest.raises(DimensionError, match="untagged dimensionless"):
                log(1.0 * unit)


@pytest.mark.parametrize("first", [q.rad, q.dimensionless])
def test_jit_conversion_rejects_implicit_angle_retagging_in_both_orders(first):
    to_radians = jax.jit(lambda value: value.to_value(q.rad))
    second = q.dimensionless if first is q.rad else q.rad

    for unit in (first, second):
        if unit is q.rad:
            assert jnp.allclose(to_radians(1.0 * unit), 1.0)
        else:
            with pytest.raises(DimensionError):
                to_radians(1.0 * unit)
