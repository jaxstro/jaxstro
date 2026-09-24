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


def test_trapezoid_accepts_explicit_static_axis() -> None:
    y = jnp.arange(6.0).reshape(2, 3)
    assert jnp.allclose(quad.trapezoid(y, axis=0), jnp.array([1.5, 2.5, 3.5]))
    assert jnp.allclose(quad.trapezoid(y, axis=1), jnp.array([2.0, 8.0]))


def test_simpson_uniform_dx() -> None:
    y = jnp.asarray([0.0, 1.0, 4.0])
    assert jnp.array_equal(quad.simpson(y, dx=0.5), 4.0 / 3.0)


def test_explicit_x_takes_precedence_over_dx() -> None:
    x = jnp.asarray([0.0, 1.0, 2.0])
    y = x**2
    assert jnp.array_equal(quad.trapezoid(y, x=x, dx=99.0), 3.0)
    assert jnp.allclose(quad.simpson(y, x=x, dx=99.0), 8.0 / 3.0)


# Exact-for-polynomial checks allow rounding only.
ROUNDING_RTOL = 1e-13
NONUNIFORM_X = jnp.asarray([0.0, 0.3, 0.5, 1.1, 1.4, 2.2, 3.0])


def test_nonuniform_x_integrates_along_any_axis() -> None:
    # A linear integrand is exact under the trapezoid rule. Shape (7, 4) with
    # axis=0 used to broadcast diff(x) along the last axis without an error.
    columns = jnp.arange(1.0, 5.0)
    y = 2.0 * NONUNIFORM_X[:, None] * columns[None, :] + 1.0
    exact = columns * NONUNIFORM_X[-1] ** 2 + NONUNIFORM_X[-1]
    total = quad.trapezoid(y, x=NONUNIFORM_X, axis=0)
    cumulative = quad.cumulative_trapezoid(y, x=NONUNIFORM_X, axis=0)
    assert jnp.allclose(total, exact, rtol=ROUNDING_RTOL)
    assert jnp.allclose(cumulative[-1], exact, rtol=ROUNDING_RTOL)
    assert jnp.array_equal(total, quad.trapezoid(y.T, x=NONUNIFORM_X, axis=-1))
    assert jnp.array_equal(
        cumulative, quad.cumulative_trapezoid(y.T, x=NONUNIFORM_X, axis=-1).T
    )


def test_simpson_is_exact_for_quadratics_on_nonuniform_grids() -> None:
    import jax

    y = 3.0 * NONUNIFORM_X**2 - NONUNIFORM_X + 2.0
    x_end = NONUNIFORM_X[-1]
    exact = x_end**3 - 0.5 * x_end**2 + 2.0 * x_end
    assert jnp.allclose(quad.simpson(y, x=NONUNIFORM_X), exact, rtol=ROUNDING_RTOL)
    # Under jit the grid is traced; the result must not change.
    assert jnp.allclose(
        jax.jit(quad.simpson)(y, NONUNIFORM_X), exact, rtol=ROUNDING_RTOL
    )
    panel_ends = NONUNIFORM_X[::2]
    exact_cumulative = panel_ends**3 - 0.5 * panel_ends**2 + 2.0 * panel_ends
    assert jnp.allclose(
        quad.cumulative_simpson(y, x=NONUNIFORM_X), exact_cumulative, rtol=ROUNDING_RTOL
    )
    stacked = jnp.stack((y, 2.0 * y), axis=0)
    assert jnp.allclose(
        quad.simpson(stacked.T, x=NONUNIFORM_X, axis=0),
        jnp.asarray([exact, 2.0 * exact]),
        rtol=ROUNDING_RTOL,
    )
