"""Tests for JAX-native B-spline utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstro.numerics import (
    BSpline1D,
    bspline_basis,
    bspline_derivative,
    bspline_design_matrix,
    bspline_eval,
    bspline_eval_deboor,
    fit_bspline_lstsq,
    interpolation,
    open_uniform_knots,
    splines,
)


def test_open_uniform_basis_is_nonnegative_partition_of_unity():
    knots = open_uniform_knots(0.0, 1.0, n_basis=6, degree=3)
    x = jnp.linspace(0.0, 1.0, 17)

    basis = bspline_basis(knots, x, degree=3)

    assert basis.shape == (17, 6)
    assert jnp.all(basis >= 0.0)
    np.testing.assert_allclose(basis.sum(axis=-1), jnp.ones_like(x), atol=1e-12)
    assert int(jnp.max(jnp.sum(basis > 1e-12, axis=-1))) <= 4


def test_degree_one_clamped_spline_matches_interp1d():
    x_grid = jnp.array([0.0, 1.0, 2.0])
    y_grid = jnp.array([0.0, 1.0, 4.0])
    knots = open_uniform_knots(0.0, 2.0, n_basis=3, degree=1)
    x_new = jnp.array([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0])

    result = bspline_eval(knots, y_grid, x_new, degree=1)
    expected = interpolation.interp1d(x_grid, y_grid, x_new)

    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_single_span_cubic_reproduces_bernstein_polynomial():
    knots = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    coeffs = jnp.array([0.0, 0.0, 0.0, 1.0])
    x = jnp.linspace(0.0, 1.0, 11)

    result = bspline_eval(knots, coeffs, x, degree=3)

    np.testing.assert_allclose(result, x**3, atol=1e-12)


def test_bspline_eval_supports_vector_coefficients_on_axis():
    knots = jnp.array([0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0])
    coeffs = jnp.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    x = jnp.array([0.25, 0.75, 1.25])

    result = bspline_eval(knots, coeffs, x, degree=2, axis=-1)

    assert result.shape == (2, 3)
    np.testing.assert_allclose(result[0] + result[1], jnp.ones_like(x), atol=1e-12)


def test_bspline_eval_is_jit_vmap_and_grad_compatible():
    knots = open_uniform_knots(0.0, 1.0, n_basis=5, degree=3)
    coeffs = jnp.linspace(-1.0, 1.0, 5)
    x = jnp.array([0.2, 0.4, 0.8])

    @jax.jit
    def evaluate(c, x_values):
        return bspline_eval(knots, c, x_values, degree=3)

    values = evaluate(coeffs, x)
    vmapped = jax.vmap(lambda x_one: bspline_eval(knots, coeffs, x_one, degree=3))(x)
    grad_coeffs = jax.grad(
        lambda c: bspline_eval(knots, c, jnp.asarray(0.4), degree=3)
    )(coeffs)

    np.testing.assert_allclose(values, vmapped, atol=1e-12)
    np.testing.assert_allclose(
        grad_coeffs, bspline_basis(knots, jnp.asarray(0.4), degree=3), atol=1e-12
    )


def test_design_matrix_is_explicit_basis_spelling_for_sample_points():
    knots = open_uniform_knots(0.0, 1.0, n_basis=6, degree=3)
    x = jnp.linspace(0.0, 1.0, 9)

    design = bspline_design_matrix(knots, x, degree=3)

    assert design.shape == (9, 6)
    np.testing.assert_allclose(design, bspline_basis(knots, x, degree=3), atol=1e-12)


def test_cubic_derivative_reproduces_polynomial_derivative():
    knots = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    coeffs = jnp.array([0.0, 0.0, 0.0, 1.0])
    x = jnp.linspace(0.1, 0.9, 7)

    result = bspline_derivative(knots, coeffs, x, degree=3)

    np.testing.assert_allclose(result, 3.0 * x**2, atol=1e-12)


def test_bspline_derivative_matches_ad_gradient_inside_domain_and_zero_outside():
    knots = open_uniform_knots(0.0, 1.0, n_basis=6, degree=3)
    coeffs = jnp.sin(jnp.linspace(0.0, 1.0, 6))
    x = jnp.array([0.13, 0.41, 0.76])

    result = bspline_derivative(knots, coeffs, x, degree=3)
    expected = jax.vmap(
        jax.grad(lambda x_one: bspline_eval(knots, coeffs, x_one, degree=3))
    )(x)
    outside = bspline_derivative(knots, coeffs, jnp.array([-0.2, 1.2]), degree=3)

    np.testing.assert_allclose(result, expected, atol=1e-12)
    np.testing.assert_allclose(outside, jnp.zeros(2), atol=1e-12)


def test_fit_bspline_lstsq_recovers_fixed_knot_coefficients():
    knots = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    x = jnp.linspace(0.0, 1.0, 9)
    y = x**3

    coeffs = fit_bspline_lstsq(knots, x, y, degree=3)

    np.testing.assert_allclose(coeffs, jnp.array([0.0, 0.0, 0.0, 1.0]), atol=1e-12)
    np.testing.assert_allclose(bspline_eval(knots, coeffs, x, degree=3), y, atol=1e-12)


def test_fit_bspline_lstsq_supports_vector_values_and_sample_axis():
    knots = open_uniform_knots(0.0, 1.0, n_basis=5, degree=3)
    true_coeffs = jnp.array(
        [
            [0.0, 0.0, 0.3, 0.8, 1.0],
            [1.0, 0.8, 0.3, 0.0, 0.0],
        ]
    )
    x = jnp.linspace(0.0, 1.0, 11)
    y = bspline_eval(knots, true_coeffs, x, degree=3, axis=-1)

    recovered = fit_bspline_lstsq(knots, x, y, degree=3, sample_axis=-1)

    assert recovered.shape == true_coeffs.shape
    np.testing.assert_allclose(recovered, true_coeffs, atol=1e-12)


def test_deboor_evaluator_matches_basis_contraction_for_scalar_and_grid():
    knots = open_uniform_knots(0.0, 1.0, n_basis=6, degree=3)
    coeffs = jnp.sin(jnp.linspace(0.0, 1.0, 6))
    x = jnp.array([0.1, 0.33, 0.67, 0.9])

    result = bspline_eval_deboor(knots, coeffs, x, degree=3)
    expected = bspline_eval(knots, coeffs, x, degree=3)

    np.testing.assert_allclose(result, expected, atol=1e-12)
    np.testing.assert_allclose(
        bspline_eval_deboor(knots, coeffs, jnp.asarray(0.4), degree=3),
        bspline_eval(knots, coeffs, jnp.asarray(0.4), degree=3),
        atol=1e-12,
    )


def test_antiderivative_derivative_recovers_original_spline_and_integral():
    knots = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    coeffs = jnp.array([0.0, 0.0, 0.0, 1.0])
    anti_knots, anti_coeffs = splines.bspline_antiderivative(knots, coeffs, degree=3)
    x = jnp.linspace(0.1, 0.9, 7)

    recovered = bspline_derivative(anti_knots, anti_coeffs, x, degree=4)
    expected = bspline_eval(knots, coeffs, x, degree=3)

    np.testing.assert_allclose(recovered, expected, atol=1e-12)
    np.testing.assert_allclose(
        splines.bspline_integral(knots, coeffs, 0.0, 1.0, degree=3),
        0.25,
        atol=1e-12,
    )


def test_roughness_penalty_is_zero_for_linear_spline_second_derivative():
    knots = open_uniform_knots(0.0, 1.0, n_basis=4, degree=1)
    coeffs = jnp.linspace(0.0, 1.0, 4)
    penalty = splines.bspline_roughness_penalty(
        knots, coeffs, degree=1, derivative_order=2
    )
    assert penalty == pytest.approx(0.0)


def test_adaptive_open_uniform_knots_uses_quantile_interior_knots():
    x = jnp.array([0.0, 0.0, 1.0, 2.0, 10.0, 10.0])
    knots = splines.adaptive_open_uniform_knots(x, n_basis=5, degree=2)
    assert knots.shape == (8,)
    np.testing.assert_allclose(knots[:3], jnp.zeros(3), atol=1e-12)
    np.testing.assert_allclose(knots[-3:], jnp.full(3, 10.0), atol=1e-12)
    assert knots[3] > 0.0
    assert knots[4] < 10.0


def test_tensor_product_design_matrix_rowwise_kronecker_product():
    bx = jnp.array([[1.0, 0.0], [0.25, 0.75]])
    by = jnp.array([[0.2, 0.8, 0.0], [0.0, 0.5, 0.5]])
    design = splines.tensor_product_design_matrix(bx, by)
    expected = jnp.array(
        [
            [0.2, 0.8, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.125, 0.125, 0.0, 0.375, 0.375],
        ]
    )
    assert design.shape == (2, 6)
    np.testing.assert_allclose(design, expected, atol=1e-12)


def test_bspline1d_is_pytree_and_works_under_jit():
    knots = open_uniform_knots(0.0, 1.0, n_basis=5, degree=3)
    coeffs = jnp.linspace(0.0, 1.0, 5)
    spline = BSpline1D(knots=knots, coeffs=coeffs, degree=3)

    @jax.jit
    def call_spline(model, x):
        return model(x), model.basis(x)

    value, basis = call_spline(spline, jnp.asarray(0.5))

    np.testing.assert_allclose(value, spline(jnp.asarray(0.5)), atol=1e-12)
    np.testing.assert_allclose(basis.sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(
        spline.derivative(jnp.asarray(0.5)),
        bspline_derivative(knots, coeffs, jnp.asarray(0.5), degree=3),
        atol=1e-12,
    )


def test_eager_validation_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="nondecreasing"):
        bspline_basis(jnp.array([0.0, 1.0, 0.5, 1.0]), jnp.asarray(0.5), degree=1)

    with pytest.raises(ValueError, match="degree"):
        open_uniform_knots(0.0, 1.0, n_basis=3, degree=-1)

    with pytest.raises(ValueError, match="coefficient axis"):
        bspline_eval(
            open_uniform_knots(0.0, 1.0, n_basis=4, degree=2),
            jnp.ones(3),
            jnp.asarray(0.5),
            degree=2,
        )

    with pytest.raises(ValueError, match="positive degree"):
        bspline_derivative(
            jnp.array([0.0, 1.0]),
            jnp.ones(1),
            jnp.asarray(0.5),
            degree=0,
        )

    with pytest.raises(ValueError, match="sample axis"):
        fit_bspline_lstsq(
            open_uniform_knots(0.0, 1.0, n_basis=4, degree=2),
            jnp.linspace(0.0, 1.0, 5),
            jnp.ones(4),
            degree=2,
        )


def test_splines_module_reexports_public_api():
    assert splines.bspline_basis is bspline_basis
    assert splines.bspline_design_matrix is bspline_design_matrix
    assert splines.bspline_eval is bspline_eval
    assert splines.bspline_eval_deboor is bspline_eval_deboor
    assert splines.bspline_derivative is bspline_derivative
    assert splines.fit_bspline_lstsq is fit_bspline_lstsq
    assert splines.open_uniform_knots is open_uniform_knots
    assert splines.BSpline1D is BSpline1D
