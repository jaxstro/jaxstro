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


def test_replay_diagnostic_tangents_are_exact_zero_or_float0() -> None:
    _, tangent = jax.jvp(_integrate, (0.7,), (1.0,))

    assert tangent.error.estimate == 0.0
    assert tangent.error.norm == 0.0
    assert tangent.error.confidence_level == 0.0
    assert tangent.tolerance == 0.0
    assert tangent.error.kind.dtype == jax.dtypes.float0
    assert tangent.status.dtype == jax.dtypes.float0
    assert all(leaf.dtype == jax.dtypes.float0 for leaf in tangent.work)


def test_unknown_gradient_mode_fails_eagerly() -> None:
    with pytest.raises(ValueError, match='gradient must be "replay" or "stop"'):
        _integrate(0.7, gradient="through")


@pytest.mark.parametrize(
    ("method", "rtol"),
    [
        (quad.GaussKronrod(21), 3e-8),
        (quad.AdaptiveClenshawCurtis(17), 2e-7),
        (quad.AdaptiveTanhSinh(3), 2e-7),
    ],
)
def test_regional_replay_parameter_derivative(method, rtol) -> None:
    def integral(theta):
        return quad.integrate(
            lambda x, args: jnp.exp(-args * x),
            quad.Interval(0.0, 2.0),
            args=theta,
            method=method,
            epsabs=1e-10,
            epsrel=1e-10,
            max_evaluations=600,
            max_regions=12,
            gradient="replay",
        ).value

    theta = 0.8
    exponential = jnp.exp(-2.0 * theta)
    expected = (2.0 * theta * exponential + exponential - 1.0) / theta**2
    assert jnp.allclose(
        jax.grad(integral)(theta),
        expected,
        rtol=rtol,
        atol=2e-9,
    )


def test_adaptive_tanh_sinh_replay_on_right_infinite_domain() -> None:
    def integral(theta):
        return quad.integrate(
            lambda x, args: jnp.exp(-args * x),
            quad.RightInfinite(0.0),
            args=theta,
            method=quad.AdaptiveTanhSinh(3),
            epsabs=1e-9,
            epsrel=1e-9,
            max_evaluations=1200,
            max_regions=16,
            gradient="replay",
        ).value

    assert jnp.allclose(jax.grad(integral)(1.3), -1.0 / 1.3**2, rtol=3e-6)


def test_breakpoint_tangent_is_stopped() -> None:
    def value(breakpoint):
        return quad.integrate(
            lambda x: jnp.exp(x),
            quad.Interval(0.0, 1.0, breakpoints=(breakpoint,)),
            method=quad.GaussKronrod(21),
            epsabs=1e-10,
            epsrel=1e-10,
            max_evaluations=126,
            max_regions=4,
            gradient="replay",
        ).value

    _, tangent = jax.jvp(value, (0.4,), (1.0,))
    assert tangent == 0.0


@pytest.mark.parametrize(
    "method",
    [
        quad.GaussKronrod(21),
        quad.AdaptiveClenshawCurtis(17),
        quad.AdaptiveTanhSinh(3),
    ],
)
def test_regional_replay_preserves_accepted_primal_formula(method) -> None:
    options = dict(
        method=method,
        epsabs=1e-10,
        epsrel=1e-10,
        max_evaluations=600,
        max_regions=12,
    )
    replay = quad.integrate(
        lambda x: jnp.exp(x),
        quad.Interval(0.0, 1.0, breakpoints=(0.35,)),
        gradient="replay",
        **options,
    )
    stopped = quad.integrate(
        lambda x: jnp.exp(x),
        quad.Interval(0.0, 1.0, breakpoints=(0.65,)),
        gradient="stop",
        **options,
    )

    assert jnp.allclose(replay.value, stopped.value, rtol=2e-13, atol=2e-13)
