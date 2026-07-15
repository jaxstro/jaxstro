"""JAX transformation contracts for fixed quadrature."""

import jax
import jax.numpy as jnp

from jaxstro.quad import GaussianRule, Interval, fixed


def test_fixed_jit_with_dynamic_bounds_and_parameter() -> None:
    evaluate = jax.jit(
        lambda lower, upper, scale: fixed(
            lambda x, args: args * x**2,
            Interval(lower, upper),
            args=scale,
            rule=GaussianRule(5),
        )
    )
    assert jnp.allclose(evaluate(0.0, 2.0, 3.0), 8.0, atol=2e-11)


def test_fixed_vmap_over_explicit_parameters() -> None:
    evaluate = jax.vmap(
        lambda scale: fixed(
            lambda x, args: args * x,
            Interval(0.0, 1.0),
            args=scale,
            rule=GaussianRule(4),
        )
    )
    got = evaluate(jnp.asarray([1.0, 2.0, 4.0]))
    assert jnp.allclose(got, jnp.asarray([0.5, 1.0, 2.0]), atol=2e-12)


def test_fixed_gradient_tracks_parameter_and_moving_bound() -> None:
    parameter_gradient = jax.grad(
        lambda scale: fixed(
            lambda x, args: jnp.exp(args * x),
            Interval(0.0, 1.0),
            args=scale,
            rule=GaussianRule(32),
        )
    )(0.0)
    bound_gradient = jax.grad(
        lambda upper: fixed(
            lambda x: x**2,
            Interval(0.0, upper),
            rule=GaussianRule(8),
        )
    )(2.0)
    assert jnp.allclose(parameter_gradient, 0.5, atol=2e-12)
    assert jnp.allclose(bound_gradient, 4.0, atol=2e-12)


def test_fixed_complex_payload() -> None:
    got = fixed(
        lambda x: jnp.exp(1j * x),
        Interval(-jnp.pi, jnp.pi),
        rule=GaussianRule(32),
    )
    assert jnp.allclose(got, 0.0, atol=2e-12)
