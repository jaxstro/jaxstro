# tests/test_special.py
"""Tests for jaxstro.numerics.special."""

import math

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
    def test_single_order_matches_the_basis_to_within_two_ulp(self, x):
        """Two implementations of one quantity must not drift.

        ``riccati_bessel_at_order`` exists only to avoid materializing lower
        orders a caller never reads -- roughly 250 MB at ``order = 30`` and
        ``n = 1e6``.

        **This asserted bitwise equality until 2026-08-03, and the relaxation is
        deliberate rather than a concession.** The two paths compute the same
        formula from provably identical inputs: with a matched seed their raw
        downward sweeps agree to **0 ULP**, measured at every ``(x, order)`` here.
        What differs is only that the basis normalises an ARRAY and
        ``at_order`` a SCALAR, so XLA selects different instruction sequences
        (FMA contraction in one and not the other) and the last bit can move.

        Bitwise equality therefore pins XLA's instruction selection, not the
        algorithm -- and a contract that can break on a compiler upgrade, while
        the mathematics is untouched, is pinning the wrong thing. ``C`` is still
        exact, because both paths compute it by the same scalar recurrence.

        The invariant that actually matters -- the two never *drift* -- is what
        two ULP still enforces. A genuine divergence (a changed seed, a changed
        recurrence, a normalisation applied in one path only) moves the result by
        orders of magnitude, not by one bit.
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
            got, want = float(s1), float(s[order])
            ulp = abs(got - want) / math.ulp(abs(want)) if want != 0.0 else 0.0
            assert ulp <= 2.0, (
                f"S_{order}(x={x}) differs by {ulp:.1f} ULP between "
                f"riccati_bessel_at_order and riccati_bessel_basis "
                f"({got!r} vs {want!r}). One or two ULP is instruction selection; "
                "more than that means the two implementations have genuinely "
                "drifted -- check the seed, the recurrence and the normalisation."
            )
            # C is computed by the identical scalar recurrence in both paths, so
            # it has no array/scalar asymmetry and must still be exact.
            assert float(c1) == float(c[order])

    def test_negative_order_raises(self):
        with pytest.raises(ValueError, match="order must be nonnegative"):
            special.riccati_bessel_at_order(jnp.asarray(1.0), order=-1)

    # ---- derivatives -----------------------------------------------------
    #
    # None of these existed until 2026-08-03, and their absence is exactly why
    # the function shipped returning finite VALUES with NaN DERIVATIVES over most
    # of its useful domain. Miller's sweep was seeded at 1e-280; the primal needs
    # only the ratio sin_x / s_asc[0] and so never noticed, but differentiating
    # that ratio carries 1 / s_asc[0]**2, and (1e-280)**2 = 1e-560 underflows
    # float64 to exactly zero. A forward-only test suite cannot see this.

    @pytest.mark.parametrize("x", [0.5, 2.0, 8.6, 15.0, 27.1, 38.3, 50.0])
    def test_the_derivative_is_finite_across_the_domain(self, x):
        """``jax.grad`` must return numbers, at the seed order we tell people to use.

        Parametrised over ``x`` specifically: the failure was *argument*
        dependent, not order dependent. The sweep ends near its seed wherever it
        has little growth -- the oscillatory region ``l < x`` -- so the NaN
        appeared at large ``x`` and hid at small ``x``. A single-point derivative
        test would have passed and proved nothing.
        """
        seed = special.riccati_seed_order(3, 60.0)

        def f(v):
            return special.riccati_bessel_basis(v, degree=3, seed_order=seed)[0]

        d = jax.jacfwd(f)(jnp.asarray(x))
        assert bool(jnp.all(jnp.isfinite(d))), (
            f"dS/dx is non-finite at x={x}. If the Miller seed or the final "
            "normalisation changed, check whether s_asc[0]**2 underflows -- the "
            "value stays finite and only the tangent dies, so nothing else "
            "catches it."
        )

    @pytest.mark.parametrize("x", [2.0, 8.6, 27.1])
    def test_the_derivative_matches_central_differences(self, x):
        """Finite is not enough -- it must be the RIGHT number.

        Guards against a repair that silences the NaN by zeroing or clamping the
        tangent, which would pass the finiteness test above while being wrong.
        """
        seed = special.riccati_seed_order(3, 60.0)

        def f(v):
            return special.riccati_bessel_basis(v, degree=3, seed_order=seed)[0]

        auto = jax.jacfwd(f)(jnp.asarray(x))
        h = 1.0e-6
        fd = (f(jnp.asarray(x + h)) - f(jnp.asarray(x - h))) / (2.0 * h)
        # 1e-7 is loose against the ~1e-9 agreement measured on 2026-08-03; it is
        # set by the central-difference truncation error, not by the autodiff.
        rel = jnp.abs(auto - fd) / jnp.maximum(jnp.abs(fd), 1e-12)
        assert float(jnp.max(rel)) < 1e-7, (
            f"dS/dx disagrees with central differences at x={x}: "
            f"max relative error {float(jnp.max(rel)):.3e}"
        )

    def test_the_single_order_path_is_differentiable_too(self):
        """``riccati_bessel_at_order`` carried the identical defect."""
        seed = special.riccati_seed_order(3, 60.0)
        for order in (0, 1, 3):
            d = jax.jacfwd(
                lambda v, o=order: special.riccati_bessel_at_order(
                    v, order=o, seed_order=seed
                )[0]
            )(jnp.asarray(27.1))
            assert bool(jnp.all(jnp.isfinite(d))), f"dS_{order}/dx is non-finite"

    def test_the_second_derivative_is_finite(self):
        """Consumers differentiate this more than once.

        micrax's thermal jet takes the Beth-Uhlenbeck scattering integral to
        third order in temperature, and ``delta_l(E)`` sits inside it -- so a
        first-order-only repair would move the failure one order out rather than
        removing it.
        """
        seed = special.riccati_seed_order(3, 60.0)

        def f(v):
            return special.riccati_bessel_basis(v, degree=3, seed_order=seed)[0]

        d2 = jax.jacfwd(jax.jacfwd(f))(jnp.asarray(27.1))
        assert bool(jnp.all(jnp.isfinite(d2))), "d2S/dx2 is non-finite"

    # ---- a documented limitation, not a passing grade ---------------------

    @pytest.mark.parametrize("sin_x_scale, max_residual", [(1e-3, 1e-12), (1e-6, 1e-9)])
    def test_accuracy_degrades_as_one_over_sin_x_near_multiples_of_pi(
        self, sin_x_scale, max_residual
    ):
        """The basis loses relative accuracy like ``eps / |sin x|`` at ``x = n pi``.

        **This pins a known defect rather than a guarantee.** The normalisation is
        ``s_asc * (sin_x / s_asc[0])``, and at ``x = n pi`` BOTH ``sin_x`` and the
        recurred ``s_asc[0]`` vanish, so the ratio is a 0/0 whose relative error
        grows as ``1 / |sin x|``. Measured 2026-08-03 at degree 30: Wronskian
        residual ``1.3e-13`` at ``|sin x| = 1e-3``, ``1.7e-10`` at ``1e-6``, and
        ``O(1)`` once ``sin x`` reaches machine epsilon.

        It is pre-existing and is NOT what the 2026-08-03 seed/normalisation work
        fixed -- that addressed NaN derivatives. It is recorded because ``x = k
        r_max`` sweeps through multiples of pi on any energy grid, and a reader
        who sees only the derivative fix would reasonably assume the function is
        now sound everywhere.

        In practice it is benign for quadrature: the damaged region has width
        ``~eps`` in ``x`` and the integrand is bounded, so a smooth measure barely
        samples it. It would NOT be benign for a root-find or an interpolation
        anchored near ``n pi``.

        The fix, when someone wants it, is to normalise against whichever of
        ``S_0 = sin x`` and ``S_1 = sin x / x - cos x`` is larger in magnitude:
        at ``x = n pi``, ``S_1 -> -cos x = +-1``, which is maximal exactly where
        ``S_0`` vanishes. **If this test starts failing because the residual got
        SMALLER, that is the fix landing -- tighten the bound, do not delete it.**
        """
        degree = 30
        x = float(math.pi - sin_x_scale)  # sin(pi - d) = sin d ~ d
        s, c = special.riccati_bessel_basis(
            jnp.asarray(x),
            degree=degree,
            seed_order=special.riccati_seed_order(degree, 60.0),
        )
        residual = float(jnp.max(jnp.abs(special.riccati_wronskian_residual(s, c))))
        assert residual < max_residual, (
            f"residual {residual:.3e} at |sin x| ~ {sin_x_scale:.0e} exceeds the "
            f"recorded {max_residual:.0e}; accuracy near n*pi got WORSE"
        )

    # ---- the mid-sweep rescale must not split the returned array ----------
    #
    # Miller's downward sweep divides by ``_RICCATI_RESCALE`` whenever it would
    # overflow. Until 2026-08-03 that division reached the carry and every LATER
    # output but never the orders already stacked, so a rescale firing inside the
    # retained window ``[0, degree]`` returned an array carrying two scales that
    # differ by the rescale factor exactly. Nothing forward-only notices: every
    # entry is finite, smooth and individually plausible.
    #
    # It needs a seed order well above the degree to appear, which is the
    # production regime (micrax's H-H solve: degree 11, seed_order 2118) and NOT
    # the regime the other tests here use.

    _CORRUPTED_BAND_DEGREE = 11
    _CORRUPTED_BAND_SEED = 2118

    def _basis_over(self, x):
        return jax.vmap(
            lambda v: special.riccati_bessel_basis(
                v,
                degree=self._CORRUPTED_BAND_DEGREE,
                seed_order=self._CORRUPTED_BAND_SEED,
            )
        )(x)

    def test_wronskian_holds_across_the_rescale_band_at_production_seed_order(self):
        """Dense sweep through the band where the mid-sweep rescale fires.

        ``x`` below roughly 5 makes the sweep grow fast enough per order that a
        rescale lands inside ``[0, degree]``. Measured on the pre-fix code over
        these 2000 points: 323 arguments corrupt, worst residual ``1.0e+150`` --
        which is the rescale factor itself, not round-off. The identity ``S_l
        C_{l-1} - S_{l-1} C_l = -1`` is exact for every order and argument, so
        any deviation is a defect and the tolerance is a numerics budget only.
        """
        x = jnp.geomspace(1.0e-3, 5.0, 2000)
        s, c = self._basis_over(x)
        residual = jax.vmap(special.riccati_wronskian_residual)(s, c)
        worst = float(jnp.max(jnp.abs(residual)))
        assert worst < 1e-10, (
            f"Wronskian residual {worst:.3e} over x in [1e-3, 5]. A residual near "
            f"{special._RICCATI_RESCALE:.3e} (or a power-of-two multiple of it) "
            "means a rescale fired inside the retained window and the returned "
            "array mixes two scales -- check that every stacked output is put "
            "back on a common scale after the sweep."
        )

    def test_the_rescale_band_result_is_not_vacuous(self):
        """Guards the sweep above against passing on zeros or constants.

        A repair that returned zeros, or that collapsed the array to one value,
        would satisfy the Wronskian nowhere -- but a repair that quietly returned
        ``-1``-preserving junk, or that a future refactor shortened to an empty
        axis, would. Pin the shape, the finiteness, the known closed form of
        ``S_0`` and ``C_0``, and the fact that the orders actually differ.
        """
        x = jnp.geomspace(1.0e-3, 5.0, 64)
        s, c = self._basis_over(x)
        degree = self._CORRUPTED_BAND_DEGREE
        assert s.shape == (x.shape[0], degree + 1)
        assert c.shape == (x.shape[0], degree + 1)
        assert bool(jnp.all(jnp.isfinite(s))) and bool(jnp.all(jnp.isfinite(c)))
        # Closed forms, which involve no recurrence.
        assert jnp.allclose(s[:, 0], jnp.sin(x), rtol=1e-12, atol=0.0)
        assert jnp.allclose(c[:, 0], jnp.cos(x), rtol=1e-12, atol=0.0)
        # S_l ~ x^(l+1)/(2l+1)!! falls steeply in l where x < 1, so below that the
        # orders must strictly decrease -- they cannot all be one repeated value.
        # (Above x ~ 1 the orders oscillate and monotonicity is not expected.)
        small = s[x < 0.2]
        assert small.shape[0] > 0
        assert bool(jnp.all(jnp.abs(small[:, :-1]) > jnp.abs(small[:, 1:])))
        assert bool(jnp.all(jnp.max(jnp.abs(s), axis=0) > 0.0))

    def test_the_derivative_survives_the_rescale_band(self):
        """The 2026-08-03 derivative repair must not regress in the fixed band.

        The scale bookkeeping and the ``stop_gradient`` normalisation touch the
        same expression, so a fix to one can silently break the other.
        """
        seed = self._CORRUPTED_BAND_SEED

        def f(v):
            return special.riccati_bessel_basis(v, degree=3, seed_order=seed)[0]

        for x in (0.05, 0.5, 2.0, 4.5):
            auto = jax.jacfwd(f)(jnp.asarray(x))
            assert bool(jnp.all(jnp.isfinite(auto))), f"dS/dx non-finite at x={x}"
            h = 1.0e-7 * x
            fd = (f(jnp.asarray(x + h)) - f(jnp.asarray(x - h))) / (2.0 * h)
            rel = jnp.abs(auto - fd) / jnp.maximum(jnp.abs(fd), 1e-300)
            assert float(jnp.max(rel)) < 1e-6, (
                f"dS/dx disagrees with central differences at x={x}: "
                f"max relative error {float(jnp.max(rel)):.3e}"
            )


# ---- the log basis: where the value basis provably cannot go ---------------


def test_the_log_basis_agrees_with_the_value_basis_where_both_are_valid():
    """Equivalence first, on the domain the existing basis actually spans.

    A new representation that is only checked where the old one FAILS has not
    been checked at all -- agreement in the well-conditioned region is what makes
    the extrapolation into the ill-conditioned one credible.
    """
    from jaxstro.numerics.special import (
        riccati_bessel_basis,
        riccati_bessel_log_basis,
        riccati_seed_order,
    )

    x = jnp.asarray([0.5, 2.0, 7.3, 25.0])
    degree = 12
    seed = riccati_seed_order(degree, 25.0)

    S, C = riccati_bessel_basis(x, degree=degree, seed_order=seed)
    sign_s, log_s, sign_c, log_c = riccati_bessel_log_basis(
        x, degree=degree, seed_order=seed
    )

    for name, got, want in (
        ("S", sign_s * jnp.exp(log_s), S),
        ("C", sign_c * jnp.exp(log_c), C),
    ):
        rel = jnp.max(jnp.abs(got - want) / (jnp.abs(want) + 1e-300))
        assert float(rel) < 1e-12, f"{name} log basis disagrees by {float(rel):.2e}"


def test_the_log_basis_is_finite_where_the_value_basis_returns_zero_and_nan():
    """**The defect this function was written for, pinned at its own argument.**

    For ``l >> x`` the two solutions leave float64 in opposite directions:
    ``S_l ~ x^(l+1)/(2l+1)!!`` underflows to zero while ``C_l ~ (2l-1)!!/x^l``
    overflows to inf. Measured here at ``l = 72``, ``x = 2e-3`` -- the
    configuration where micrax's H-H channel integral returned NaN on 2026-08-05
    for ``l = 68..72`` at 32 and 64 quadrature nodes per panel, and finite values
    at 16, because finer panels sample smaller ``k``.

    The log form must be finite and must bracket the true magnitudes, and the
    value form must still fail -- if it stops failing, the underflow moved and
    this test is measuring the wrong thing.
    """
    from jaxstro.numerics.special import (
        riccati_bessel_basis,
        riccati_bessel_log_basis,
        riccati_seed_order,
    )

    x = jnp.asarray([2.0e-3, 1.0e-4])
    degree = 72
    seed = riccati_seed_order(degree, 25.0)

    S, C = riccati_bessel_basis(x, degree=degree, seed_order=seed)
    assert bool(jnp.all(S[degree] == 0.0)), (
        "S no longer underflows here -- the log basis is being pinned at an "
        "argument that no longer exercises the defect. Find the new one."
    )
    assert not bool(jnp.all(jnp.isfinite(C[degree]))), (
        "C no longer overflows here -- same problem, other side."
    )

    _, log_s, _, log_c = riccati_bessel_log_basis(x, degree=degree, seed_order=seed)
    assert bool(jnp.all(jnp.isfinite(log_s))), "log|S| is not finite"
    assert bool(jnp.all(jnp.isfinite(log_c))), "log|C| is not finite"

    # The asymptotic magnitudes, as an independent check that these are the RIGHT
    # finite numbers rather than merely finite ones:
    #     log|S_l| ~ (l+1) log x - log (2l+1)!!
    #     log|C_l| ~ log (2l-1)!! - l log x
    import math

    for i, xv in enumerate((2.0e-3, 1.0e-4)):
        ln_dfact_hi = sum(math.log(2 * m + 1) for m in range(degree + 1))  # (2l+1)!!
        ln_dfact_lo = ln_dfact_hi - math.log(2 * degree + 1)  # (2l-1)!!
        want_s = (degree + 1) * math.log(xv) - ln_dfact_hi
        want_c = ln_dfact_lo - degree * math.log(xv)
        assert abs(float(log_s[degree, i]) - want_s) < 0.05, (
            f"log|S_72({xv})| = {float(log_s[degree, i]):.3f}, asymptotic {want_s:.3f}"
        )
        assert abs(float(log_c[degree, i]) - want_c) < 0.05, (
            f"log|C_72({xv})| = {float(log_c[degree, i]):.3f}, asymptotic {want_c:.3f}"
        )
