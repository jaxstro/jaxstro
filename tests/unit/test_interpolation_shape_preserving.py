"""Tests for shape-preserving interpolation utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstro.numerics import interpolation


def test_cubic_hermite_interp_reproduces_cubic_with_supplied_derivatives():
    x = jnp.array([0.0, 1.0])
    y = x**3
    dydx = 3.0 * x**2
    x_new = jnp.linspace(0.0, 1.0, 11)

    result = interpolation.cubic_hermite_interp(x, y, dydx, x_new)

    np.testing.assert_allclose(result, x_new**3, atol=1e-12)


def test_cubic_hermite_interp_clamps_outside_domain_by_default():
    x = jnp.array([0.0, 1.0])
    y = jnp.array([2.0, 5.0])
    dydx = jnp.array([1.0, 1.0])
    x_new = jnp.array([-1.0, 0.5, 2.0])

    result = interpolation.cubic_hermite_interp(x, y, dydx, x_new)

    np.testing.assert_allclose(
        result[jnp.array([0, 2])],
        jnp.array([2.0, 5.0]),
        atol=1e-12,
    )


def test_pchip_slopes_zero_at_turning_point_and_plateau():
    x = jnp.array([0.0, 1.0, 2.0, 3.0])
    y = jnp.array([0.0, 1.0, 0.5, 0.5])

    slopes = interpolation.pchip_slopes(x, y)

    assert slopes.shape == y.shape
    np.testing.assert_allclose(slopes[1], 0.0, atol=1e-12)
    np.testing.assert_allclose(slopes[2], 0.0, atol=1e-12)


def test_monotone_cubic_interp_preserves_bounds_for_monotone_data():
    x = jnp.array([0.0, 1.0, 2.0, 4.0])
    y = jnp.array([0.0, 0.2, 0.8, 1.0])
    x_new = jnp.linspace(0.0, 4.0, 101)

    result = interpolation.monotone_cubic_interp(x, y, x_new)

    assert jnp.all(result >= y[0] - 1e-12)
    assert jnp.all(result <= y[-1] + 1e-12)
    assert jnp.all(jnp.diff(result) >= -1e-12)
    np.testing.assert_allclose(result[0], y[0], atol=1e-12)
    np.testing.assert_allclose(result[-1], y[-1], atol=1e-12)


def test_monotone_cubic_interp_supports_vector_values_on_axis():
    x = jnp.array([0.0, 1.0, 2.0, 3.0])
    y = jnp.array(
        [
            [0.0, 0.2, 0.8, 1.0],
            [1.0, 0.8, 0.2, 0.0],
        ]
    )
    x_new = jnp.array([0.25, 1.5, 2.75])

    result = interpolation.monotone_cubic_interp(x, y, x_new, axis=-1)

    assert result.shape == (2, 3)
    assert jnp.all(result[0] >= 0.0)
    assert jnp.all(result[0] <= 1.0)
    assert jnp.all(result[1] >= 0.0)
    assert jnp.all(result[1] <= 1.0)


def test_monotone_tabulated_function_is_pytree_and_jit_compatible():
    x = jnp.array([0.0, 1.0, 2.0, 3.0])
    y = jnp.array([0.0, 0.2, 0.8, 1.0])
    table = interpolation.MonotoneTabulatedFunction1D(x=x, y=y)

    @jax.jit
    def call_table(model, x_new):
        return model(x_new)

    result = call_table(table, jnp.array([0.5, 1.5, 2.5]))

    assert result.shape == (3,)
    assert jnp.all(result >= 0.0)
    assert jnp.all(result <= 1.0)


def test_shape_preserving_interpolation_is_jit_vmap_and_grad_compatible():
    x = jnp.array([0.0, 1.0, 2.0, 3.0])
    y = jnp.array([0.0, 0.2, 0.8, 1.0])
    x_new = jnp.array([0.25, 1.25, 2.25])

    @jax.jit
    def evaluate(values, xq):
        return interpolation.monotone_cubic_interp(x, values, xq)

    result = evaluate(y, x_new)
    vmapped = jax.vmap(lambda xq: interpolation.monotone_cubic_interp(x, y, xq))(x_new)
    grad_y = jax.grad(
        lambda values: jnp.sum(interpolation.monotone_cubic_interp(x, values, x_new))
    )(y)

    np.testing.assert_allclose(result, vmapped, atol=1e-12)
    assert jnp.all(jnp.isfinite(grad_y))


def test_shape_preserving_interpolation_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="strictly increasing"):
        interpolation.pchip_slopes(
            jnp.array([0.0, 1.0, 1.0]),
            jnp.array([0.0, 1.0, 2.0]),
        )

    with pytest.raises(ValueError, match="derivatives"):
        interpolation.cubic_hermite_interp(
            jnp.array([0.0, 1.0]),
            jnp.array([0.0, 1.0]),
            jnp.array([1.0]),
            jnp.asarray(0.5),
        )

    with pytest.raises(ValueError, match="axis length"):
        interpolation.monotone_cubic_interp(
            jnp.array([0.0, 1.0, 2.0]),
            jnp.array([0.0, 1.0]),
            jnp.asarray(0.5),
        )


def test_natural_cubic_spline_matches_scipy_cubic_spline():
    scipy_interpolate = pytest.importorskip("scipy.interpolate")
    x = jnp.array([0.12, 0.27, 0.41, 0.63, 0.89, 1.35])
    y = jnp.array([3.1, 2.4, 1.9, 1.2, 0.7, 0.4])
    x_new = jnp.linspace(float(x[0]), float(x[-1]), 73)

    coeffs = interpolation.natural_cubic_spline_coeffs(x, y)
    result = interpolation.eval_cubic_spline(x, coeffs, x_new)
    expected = scipy_interpolate.CubicSpline(
        np.asarray(x),
        np.asarray(y),
        bc_type="natural",
    )(np.asarray(x_new))

    np.testing.assert_allclose(result, expected, atol=1e-11, rtol=1e-11)


def test_natural_cubic_spline_coefficients_satisfy_natural_boundary_conditions():
    x = jnp.array([0.0, 0.5, 1.5, 2.0])
    y = jnp.array([0.0, 1.0, -0.25, 0.5])

    a, b, c, d = interpolation.natural_cubic_spline_coeffs(x, y)
    h = jnp.diff(x)
    left_second = 2.0 * c[0]
    right_second = 2.0 * c[-1] + 6.0 * d[-1] * h[-1]

    np.testing.assert_allclose(a, y[:-1], atol=1e-12)
    np.testing.assert_allclose(left_second, 0.0, atol=1e-12)
    np.testing.assert_allclose(right_second, 0.0, atol=1e-12)
    assert b.shape == c.shape == d.shape == (x.shape[0] - 1,)


def test_eval_cubic_spline_clamps_outside_domain_to_endpoint_values():
    x = jnp.array([0.0, 1.0, 2.0])
    y = jnp.array([1.0, 3.0, 2.0])
    coeffs = interpolation.natural_cubic_spline_coeffs(x, y)
    x_new = jnp.array([-1.0, 0.5, 3.0])

    result = interpolation.eval_cubic_spline(x, coeffs, x_new)

    np.testing.assert_allclose(result[0], y[0], atol=1e-12)
    np.testing.assert_allclose(result[-1], y[-1], atol=1e-12)


def test_natural_cubic_spline_wrapper_is_pytree_and_jit_compatible():
    x = jnp.array([0.0, 0.5, 1.0, 2.0])
    y = jnp.array([0.0, 1.0, 0.5, 0.25])
    spline = interpolation.NaturalCubicSpline1D(x=x, y=y)

    @jax.jit
    def call_spline(model, x_new):
        return model(x_new)

    result = call_spline(spline, jnp.array([0.25, 0.75, 1.5]))

    assert result.shape == (3,)
    assert jnp.all(jnp.isfinite(result))


def test_natural_cubic_spline_is_jit_vmap_and_grad_compatible():
    x = jnp.array([0.0, 0.5, 1.0, 2.0])
    y = jnp.array([0.0, 1.0, 0.5, 0.25])
    x_new = jnp.array([0.25, 0.75, 1.5])

    @jax.jit
    def evaluate(values, xq):
        coeffs = interpolation.natural_cubic_spline_coeffs(x, values)
        return interpolation.eval_cubic_spline(x, coeffs, xq)

    result = evaluate(y, x_new)
    vmapped = jax.vmap(lambda xq: evaluate(y, xq))(x_new)
    grad_y = jax.grad(lambda values: jnp.sum(evaluate(values, x_new)))(y)

    np.testing.assert_allclose(result, vmapped, atol=1e-12)
    assert jnp.all(jnp.isfinite(grad_y))
