"""Gradient checks for quantity math wrappers."""

import jax
import jax.numpy as jnp
import pytest

from jaxstro import quantity as q


def test_grad_through_sqrt_and_conversion():
    def loss(x):
        area = (x * q.m) ** 2
        return q.math.sqrt(area).to_value(q.cm)

    assert jax.grad(loss)(2.0) == 100.0


def test_grad_through_where_and_reduction():
    cond = jnp.array([True, False, True])

    def loss(x):
        left = x * q.cm
        right = (2.0 * x) * q.cm
        return q.math.sum(q.math.where(cond, left, right)).to_value(q.cm)

    assert jax.grad(loss)(jnp.array([1.0, 2.0, 3.0])).tolist() == [1.0, 2.0, 1.0]


def test_quantity_grad_has_physical_derivative_unit_in_both_length_bases():
    derivative = q.grad(lambda x: x**3)

    centimetres = derivative(3.0 * q.cm)
    metres = derivative(0.03 * q.m)
    assert centimetres.unit == q.cm**2
    assert centimetres.value == pytest.approx(27.0)
    assert metres.unit == q.m**2
    assert metres.to_value(q.cm**2) == pytest.approx(27.0)


def test_quantity_grad_composes_with_jit_and_vmap():
    derivative = jax.jit(jax.vmap(q.grad(lambda x: x**3)))

    result = derivative(jnp.array([2.0, 3.0]) * q.cm)
    assert result.unit == q.cm**2
    assert jnp.allclose(result.value, jnp.array([12.0, 27.0]))


def test_quantity_grad_of_sine_converts_inverse_degree_scale():
    derivative = q.grad(q.math.sin)(60.0 * q.deg)

    assert derivative.to_value(q.dimensionless) == pytest.approx(0.5)


def test_quantity_grad_requires_quantity_input_and_scalar_quantity_output():
    with pytest.raises(TypeError, match="Quantity input"):
        q.grad(lambda x: x**3)(3.0)
    with pytest.raises(TypeError, match="Quantity output"):
        q.grad(lambda x: x.value**3)(3.0 * q.cm)
    with pytest.raises(ValueError, match="scalar"):
        q.grad(lambda x: jnp.array([1.0, 2.0]) * x)(3.0 * q.cm)
