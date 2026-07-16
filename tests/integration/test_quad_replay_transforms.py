"""JAX transformation contracts for adaptive replay derivatives."""

import jax
import jax.numpy as jnp
import pytest

from jaxstro import quad


def _integrate(theta, *, lower=0.0, upper=1.0, gradient="replay"):
    return quad.integrate(
        lambda x, args: jnp.exp(args * x),
        quad.Interval(lower, upper),
        args=theta,
        method=quad.GaussKronrod(21),
        epsabs=1e-10,
        epsrel=1e-10,
        max_evaluations=147,
        max_regions=4,
        gradient=gradient,
    )


def test_gauss_kronrod_replay_matches_analytic_parameter_derivative() -> None:
    theta = 0.7
    expected = ((theta - 1.0) * jnp.exp(theta) + 1.0) / theta**2
    actual = jax.grad(lambda value: _integrate(value).value)(theta)
    assert jnp.allclose(actual, expected, rtol=2e-8, atol=2e-10)


def test_gauss_kronrod_replay_matches_moving_bound_identity() -> None:
    lower_grad, upper_grad = jax.jacrev(
        lambda lower, upper: _integrate(0.7, lower=lower, upper=upper).value,
        argnums=(0, 1),
    )(0.2, 1.3)
    assert jnp.allclose(lower_grad, -jnp.exp(0.7 * 0.2), rtol=2e-8)
    assert jnp.allclose(upper_grad, jnp.exp(0.7 * 1.3), rtol=2e-8)


def test_gauss_kronrod_coincident_bound_tangents_are_not_zeroed() -> None:
    lower_grad, upper_grad = jax.jacrev(
        lambda lower, upper: _integrate(0.7, lower=lower, upper=upper).value,
        argnums=(0, 1),
    )(0.4, 0.4)
    value = jnp.exp(0.7 * 0.4)
    assert jnp.allclose(lower_grad, -value, rtol=2e-8)
    assert jnp.allclose(upper_grad, value, rtol=2e-8)


def test_stop_mode_remains_exactly_zero() -> None:
    assert jax.grad(lambda theta: _integrate(theta, gradient="stop").value)(0.7) == 0.0


def test_unknown_gradient_mode_fails_eagerly() -> None:
    with pytest.raises(ValueError, match='gradient must be "replay" or "stop"'):
        _integrate(0.7, gradient="through")
