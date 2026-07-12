"""Analytic and adversarial validation for certified implicit roots."""

import jax
import jax.numpy as jnp
import pytest

from jaxstro.numerics import rootfinding


def _solve(f, theta, *, assumptions=None, slope_floor=1.0e-8):
    if assumptions is None:
        assumptions = rootfinding.ImplicitRootAssumptions(True, True)
    return rootfinding.implicit_bracketed_root(
        f,
        theta,
        0.0,
        4.0,
        assumptions=assumptions,
        max_steps=96,
        atol=1.0e-14,
        rtol=1.0e-14,
        safeguard_fraction=0.1,
        derivative_residual_atol=1.0e-12,
        derivative_width_atol=1.0e-12,
        derivative_slope_floor=slope_floor,
    )


@pytest.mark.parametrize(
    ("residual", "theta", "expected"),
    [
        (lambda x, t: x - t, 2.0, 1.0),
        (lambda x, t: x**2 - t, 2.0, 1.0 / (2.0 * jnp.sqrt(2.0))),
        (lambda x, t: jnp.exp(x) - t, 2.0, 0.5),
    ],
)
def test_certified_gradient_matches_analytic_and_central_fd(
    residual, theta, expected
) -> None:
    def solve(parameter):
        return _solve(residual, parameter).root

    theta = jnp.asarray(theta, dtype=jnp.float64)
    result = _solve(residual, theta)
    ad = jax.grad(solve)(theta)
    step = jnp.asarray(1.0e-5, dtype=theta.dtype)
    fd = (solve(theta + step) - solve(theta - step)) / (2.0 * step)

    assert result.certified
    assert ad == pytest.approx(float(expected), rel=1.0e-9)
    assert ad == pytest.approx(float(fd), rel=1.0e-7)


def test_rejected_assumption_returns_nan_value_and_gradient() -> None:
    assumptions = rootfinding.ImplicitRootAssumptions(False, True)

    def solve(theta):
        return _solve(lambda x, t: x - t, theta, assumptions=assumptions).root

    theta = jnp.asarray(2.0)
    result = _solve(lambda x, t: x - t, theta, assumptions=assumptions)

    assert result.status == rootfinding.DERIVATIVE_STATUS_ASSUMPTIONS_REJECTED
    assert not result.certified
    assert result.primal.converged
    assert jnp.isnan(result.root)
    assert jnp.isnan(jax.grad(solve)(theta))


def test_zero_slope_returns_nan_value_and_gradient() -> None:
    def solve(theta):
        return _solve(lambda x, t: (x - t) ** 3, theta).root

    theta = jnp.asarray(2.0)
    result = _solve(lambda x, t: (x - t) ** 3, theta)

    assert result.status == rootfinding.DERIVATIVE_STATUS_SLOPE_ILL_CONDITIONED
    assert not result.certified
    assert jnp.isnan(result.root)
    assert jnp.isnan(jax.grad(solve)(theta))


@pytest.mark.parametrize("floor", [0.0, -1.0, jnp.nan])
def test_invalid_slope_floor_fails_closed_under_jit_and_grad(floor) -> None:
    def solve(theta, slope_floor):
        return _solve(
            lambda x, t: (x - t) ** 3,
            theta,
            slope_floor=slope_floor,
        )

    theta = jnp.asarray(2.0)
    result = jax.jit(solve)(theta, jnp.asarray(floor))
    gradient = jax.grad(lambda value: solve(value, jnp.asarray(floor)).root)(theta)

    assert result.status == rootfinding.DERIVATIVE_STATUS_SLOPE_ILL_CONDITIONED
    assert not result.certified
    assert jnp.isnan(result.root)
    assert jnp.isnan(gradient)


def test_certified_jaxpr_contains_custom_root_but_no_while() -> None:
    def solve(theta):
        return _solve(lambda x, t: x - t, theta).root

    text = str(jax.make_jaxpr(jax.grad(solve))(jnp.asarray(2.0)))

    assert "custom_root" in text
    assert "while[" not in text
