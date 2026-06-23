"""Gradient checks for quantity math wrappers."""

import jax
import jax.numpy as jnp

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
