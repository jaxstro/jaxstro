"""Canonical sampled-data integration contracts."""

import jax.numpy as jnp

from jaxstro import quad
from jaxstro.numerics import integration


def test_sampled_ownership_is_inverted() -> None:
    assert quad.trapezoid.__module__ == "jaxstro.quad.sampled"
    assert integration.trapz is quad.trapezoid
    assert integration.cumulative_trapz is quad.cumulative_trapezoid
    assert integration.simpson is quad.simpson
    assert integration.cumulative_simpson is quad.cumulative_simpson


def test_trapezoid_uniform_dx() -> None:
    y = jnp.asarray([1.0, 2.0, 4.0])
    assert jnp.array_equal(quad.trapezoid(y, dx=0.25), 1.125)


def test_simpson_uniform_dx() -> None:
    y = jnp.asarray([0.0, 1.0, 4.0])
    assert jnp.array_equal(quad.simpson(y, dx=0.5), 4.0 / 3.0)


def test_explicit_x_takes_precedence_over_dx() -> None:
    x = jnp.asarray([0.0, 1.0, 2.0])
    y = x**2
    assert jnp.array_equal(quad.trapezoid(y, x=x, dx=99.0), 3.0)
    assert jnp.allclose(quad.simpson(y, x=x, dx=99.0), 8.0 / 3.0)
