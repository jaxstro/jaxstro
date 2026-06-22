# tests/test_special.py
"""Tests for jaxstro.numerics.special."""

import jax
import jax.numpy as jnp
import pytest

from jaxstro import constants
from jaxstro.numerics import special


class TestPlanckKernels:
    """Tests for stable Planck-function kernels in CGS units."""

    def test_planck_lambda_matches_direct_formula(self):
        wavelength_cm = jnp.array([1e-4, 2e-4, 5e-4])
        temperature = 5772.0
        expected = (
            2.0
            * constants.H_CGS
            * constants.C_CGS**2
            / wavelength_cm**5
            / jnp.expm1(
                constants.H_CGS
                * constants.C_CGS
                / (wavelength_cm * constants.K_B * temperature)
            )
        )
        result = special.planck_lambda_cgs(wavelength_cm, temperature)
        assert jnp.allclose(result, expected, rtol=1e-12, atol=0.0)

    def test_planck_nu_matches_direct_formula(self):
        frequency_hz = jnp.array([1e10, 1e12, 1e14])
        temperature = 3500.0
        expected = (
            2.0
            * constants.H_CGS
            * frequency_hz**3
            / constants.C_CGS**2
            / jnp.expm1(constants.H_CGS * frequency_hz / (constants.K_B * temperature))
        )
        result = special.planck_nu_cgs(frequency_hz, temperature)
        assert jnp.allclose(result, expected, rtol=1e-12, atol=0.0)

    def test_log_planck_lambda_stays_finite_in_wien_tail(self):
        wavelength_cm = jnp.array(1e-7)
        temperature = jnp.array(3000.0)
        log_value = special.log_planck_lambda_cgs(wavelength_cm, temperature)
        assert jnp.isfinite(log_value)
        assert log_value < 0.0

    def test_rayleigh_jeans_limit_for_planck_nu(self):
        frequency_hz = jnp.array(1e6)
        temperature = jnp.array(5000.0)
        result = special.planck_nu_cgs(frequency_hz, temperature)
        rayleigh_jeans = (
            2.0 * constants.K_B * temperature * frequency_hz**2 / (constants.C_CGS**2)
        )
        assert jnp.allclose(result, rayleigh_jeans, rtol=1e-6)

    def test_rejects_nonpositive_inputs_eagerly(self):
        with pytest.raises(ValueError, match="wavelength_cm"):
            special.planck_lambda_cgs(jnp.array([0.0]), 5000.0)
        with pytest.raises(ValueError, match="temperature"):
            special.planck_nu_cgs(jnp.array([1.0]), 0.0)


class TestLogWeights:
    """Tests for normalized log-weight helpers."""

    def test_log_normalize_exponentiates_to_one(self):
        log_weights = jnp.array([-1000.0, -1001.0, -999.0])
        normalized = special.log_normalize(log_weights)
        assert jnp.allclose(jnp.sum(jnp.exp(normalized)), 1.0)

    def test_normalize_log_weights_is_shift_invariant(self):
        log_weights = jnp.array([0.0, 1.0, 2.0])
        shifted = log_weights + 500.0
        assert jnp.allclose(
            special.normalize_log_weights(log_weights),
            special.normalize_log_weights(shifted),
        )

    def test_axis_argument_normalizes_rows(self):
        log_weights = jnp.array([[0.0, 1.0], [2.0, 3.0]])
        probs = special.normalize_log_weights(log_weights, axis=1)
        assert jnp.allclose(jnp.sum(probs, axis=1), jnp.ones(2))

    def test_jit_and_grad_compatible(self):
        @jax.jit
        def entropy(log_weights):
            probs = special.normalize_log_weights(log_weights)
            return -jnp.sum(probs * special.log_normalize(log_weights))

        log_weights = jnp.array([-2.0, 0.0, 1.0])
        assert jnp.isfinite(entropy(log_weights))
        assert jnp.all(jnp.isfinite(jax.grad(entropy)(log_weights)))


class TestOrthogonalPolynomialBases:
    """Tests for orthogonal polynomial basis recurrence helpers."""

    def test_legendre_basis_matches_low_order_polynomials(self):
        x = jnp.array([-0.5, 0.0, 0.5])
        basis = special.legendre_basis(x, degree=3)
        expected = jnp.stack(
            [
                jnp.ones_like(x),
                x,
                0.5 * (3.0 * x**2 - 1.0),
                0.5 * (5.0 * x**3 - 3.0 * x),
            ],
            axis=-1,
        )
        assert jnp.allclose(basis, expected)

    def test_chebyshev_t_basis_matches_low_order_polynomials(self):
        x = jnp.array([-0.5, 0.0, 0.5])
        basis = special.chebyshev_t_basis(x, degree=3)
        expected = jnp.stack(
            [
                jnp.ones_like(x),
                x,
                2.0 * x**2 - 1.0,
                4.0 * x**3 - 3.0 * x,
            ],
            axis=-1,
        )
        assert jnp.allclose(basis, expected)

    def test_laguerre_basis_matches_low_order_polynomials(self):
        x = jnp.array([0.0, 1.0, 2.0])
        basis = special.laguerre_basis(x, degree=2)
        expected = jnp.stack(
            [
                jnp.ones_like(x),
                1.0 - x,
                1.0 - 2.0 * x + 0.5 * x**2,
            ],
            axis=-1,
        )
        assert jnp.allclose(basis, expected)

    def test_degree_zero_and_jax_transforms(self):
        x = jnp.array([0.2, 0.4])
        assert jnp.allclose(special.legendre_basis(x, degree=0), jnp.ones((2, 1)))

        @jax.jit
        def evaluate(values):
            return special.chebyshev_t_basis(values, degree=4)

        assert evaluate(x).shape == (2, 5)
        grad = jax.grad(lambda z: jnp.sum(special.laguerre_basis(z, degree=3)))(
            jnp.array(0.5)
        )
        assert jnp.isfinite(grad)

    def test_rejects_negative_degree(self):
        with pytest.raises(ValueError, match="degree"):
            special.legendre_basis(jnp.array([0.0]), degree=-1)
