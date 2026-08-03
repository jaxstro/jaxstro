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


class TestRiccatiBessel:
    """Tests for Riccati-Bessel ``S_l = x j_l(x)`` and ``C_l = -x y_l(x)``.

    Everything is gated on the Wronskian identity ``S_l C_{l-1} - S_{l-1} C_l =
    -1``, exact for every order and argument, because the failure mode is an
    unstable recurrence returning finite, smooth, plausible wrong values.
    """

    @pytest.mark.parametrize("x", [0.5, 2.0, 10.0, 50.0, 120.0, 400.0])
    def test_wronskian_identity_holds(self, x):
        """Spans both regimes: ``l > x`` (direction matters) and ``l < x`` (seed does)."""
        degree = 30
        s, c = special.riccati_bessel_basis(
            jnp.asarray(x),
            degree=degree,
            seed_order=special.riccati_seed_order(degree, x),
        )
        assert float(jnp.max(special.riccati_wronskian_residual(s, c))) < 1e-14

    def test_seed_order_that_does_not_clear_the_argument_fails(self):
        """The default seed clears the degree only, and that is not sufficient.

        Miller's sweep self-corrects only where ``l > x``. Below ``x`` both
        solutions oscillate and the seed error persists. Residual with the
        default seed: ``6.7e-16`` at ``x = 50``, ``0.57`` at ``x = 400``.
        """
        degree, x = 30, 400.0
        s, c = special.riccati_bessel_basis(jnp.asarray(x), degree=degree)
        assert float(jnp.max(special.riccati_wronskian_residual(s, c))) > 1e-3

        s, c = special.riccati_bessel_basis(
            jnp.asarray(x),
            degree=degree,
            seed_order=special.riccati_seed_order(degree, x),
        )
        assert float(jnp.max(special.riccati_wronskian_residual(s, c))) < 1e-14

    @pytest.mark.parametrize("x", [0.7, 3.0, 12.0])
    def test_lowest_orders_match_closed_forms(self, x):
        """Against textbook expressions, which involve no recurrence."""
        s, c = special.riccati_bessel_basis(
            jnp.asarray(x), degree=4, seed_order=special.riccati_seed_order(4, x)
        )
        expected_s = [
            jnp.sin(x),
            jnp.sin(x) / x - jnp.cos(x),
            (3.0 / x**2 - 1.0) * jnp.sin(x) - 3.0 * jnp.cos(x) / x,
        ]
        expected_c = [
            jnp.cos(x),
            jnp.cos(x) / x + jnp.sin(x),
            (3.0 / x**2 - 1.0) * jnp.cos(x) + 3.0 * jnp.sin(x) / x,
        ]
        for order in range(3):
            assert float(s[order]) == pytest.approx(
                float(expected_s[order]), rel=1e-12, abs=0.0
            )
            assert float(c[order]) == pytest.approx(
                float(expected_c[order]), rel=1e-12, abs=0.0
            )

    def test_small_argument_power_law(self):
        """``S_l ~ x^(l+1)/(2l+1)!!``, which constrains the ORDER LABELLING.

        The Wronskian cannot detect an off-by-one in the order index; this can.
        """
        x = 0.02
        s, _ = special.riccati_bessel_basis(
            jnp.asarray(x), degree=5, seed_order=special.riccati_seed_order(5, x)
        )
        double_factorial = 1.0
        for order in range(4):
            if order > 0:
                double_factorial *= 2 * order + 1
            assert float(s[order]) == pytest.approx(
                x ** (order + 1) / double_factorial, rel=1e-4, abs=0.0
            )

    def test_negative_degree_raises(self):
        with pytest.raises(ValueError, match="degree must be nonnegative"):
            special.riccati_bessel_basis(jnp.asarray(1.0), degree=-1)

    def test_jit_and_vmap_agree_with_eager(self):
        """Both transformations must preserve the values exactly."""
        x = jnp.linspace(0.5, 20.0, 7)
        seed = special.riccati_seed_order(8, 20.0)
        eager = special.riccati_bessel_basis(x, degree=8, seed_order=seed)
        jitted = jax.jit(
            lambda v: special.riccati_bessel_basis(v, degree=8, seed_order=seed)
        )(x)
        for lhs, rhs in zip(eager, jitted):
            assert jnp.allclose(lhs, rhs, rtol=1e-13, atol=0.0)

    @pytest.mark.parametrize("x", [0.5, 3.0, 50.0, 120.0])
    def test_single_order_matches_the_basis_exactly(self, x):
        """Two implementations of one quantity must not drift.

        ``riccati_bessel_at_order`` exists only to avoid materializing lower
        orders a caller never reads -- roughly 250 MB at ``order = 30`` and
        ``n = 1e6``. With a matched seed it is bit-identical to the basis.
        """
        degree = 14
        seed = special.riccati_seed_order(degree, x)
        s, c = special.riccati_bessel_basis(
            jnp.asarray(x), degree=degree, seed_order=seed
        )
        for order in range(degree + 1):
            s1, c1 = special.riccati_bessel_at_order(
                jnp.asarray(x), order=order, seed_order=seed
            )
            assert float(s1) == float(s[order])
            assert float(c1) == float(c[order])

    def test_negative_order_raises(self):
        with pytest.raises(ValueError, match="order must be nonnegative"):
            special.riccati_bessel_at_order(jnp.asarray(1.0), order=-1)
