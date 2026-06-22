# tests/test_numerics.py
"""
Tests for jaxstro.numerics modules.

Verifies numerical utilities work correctly with JAX transforms.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxstro.numerics import (
    compensated,
    integration,
    interpolation,
    rootfinding,
    stats,
)


class TestSafeLog:
    """Tests for safe_log function."""

    def test_positive_values(self):
        """safe_log should match log for positive values."""
        x = jnp.array([1.0, 2.0, 10.0])
        result = stats.safe_log(x)
        expected = jnp.log(x)
        assert jnp.allclose(result, expected)

    def test_zero_finite(self):
        """safe_log(0) should be finite (not -inf)."""
        result = stats.safe_log(jnp.array(0.0))
        assert jnp.isfinite(result)

    def test_negative_finite(self):
        """safe_log of negative values should be finite."""
        result = stats.safe_log(jnp.array(-1.0))
        assert jnp.isfinite(result)

    def test_jit_compatible(self):
        """safe_log should work under jit."""
        f = jax.jit(stats.safe_log)
        result = f(jnp.array([0.0, 1.0, 2.0]))
        assert jnp.all(jnp.isfinite(result))

    def test_vmap_compatible(self):
        """safe_log should work under vmap."""
        xs = jnp.array([[0.0, 1.0], [2.0, 3.0]])
        result = jax.vmap(stats.safe_log)(xs)
        assert result.shape == (2, 2)


class TestSafeExp:
    """Tests for safe_exp function."""

    def test_normal_values(self):
        """safe_exp should match exp for normal values."""
        x = jnp.array([0.0, 1.0, 10.0])
        result = stats.safe_exp(x)
        expected = jnp.exp(x)
        assert jnp.allclose(result, expected)

    def test_large_values_clipped(self):
        """safe_exp should clip large exponents."""
        result = stats.safe_exp(jnp.array(1000.0))
        # Should be exp(100), not inf
        expected = jnp.exp(100.0)
        assert jnp.allclose(result, expected)

    def test_custom_max(self):
        """safe_exp should respect custom max_exp."""
        result = stats.safe_exp(jnp.array(100.0), max_exp=50.0)
        expected = jnp.exp(50.0)
        assert jnp.allclose(result, expected)


class TestSafeDiv:
    """Tests for safe_div function."""

    def test_normal_division(self):
        """safe_div should match regular division for normal values."""
        result = stats.safe_div(jnp.array(10.0), jnp.array(2.0))
        assert jnp.allclose(result, 5.0)

    def test_zero_denominator_finite(self):
        """safe_div with zero denominator should be finite."""
        result = stats.safe_div(jnp.array(1.0), jnp.array(0.0))
        assert jnp.isfinite(result)

    def test_differentiable(self):
        """safe_div should be differentiable."""

        def f(x):
            return stats.safe_div(x, x + 1.0)

        grad_f = jax.grad(f)
        result = grad_f(jnp.array(1.0))
        assert jnp.isfinite(result)


class TestRelativeError:
    """Tests for relative_error function."""

    def test_same_values(self):
        """relative_error of identical values should be 0."""
        x = jnp.array([1.0, 2.0, 3.0])
        result = stats.relative_error(x, x)
        assert jnp.allclose(result, 0.0)

    def test_known_error(self):
        """relative_error should compute correct value."""
        result = stats.relative_error(jnp.array(1.1), jnp.array(1.0))
        assert jnp.allclose(result, 0.1)

    def test_zero_old_value(self):
        """relative_error with zero old value should be finite."""
        result = stats.relative_error(jnp.array(1.0), jnp.array(0.0))
        assert jnp.isfinite(result)

    def test_always_positive(self):
        """relative_error should always be non-negative."""
        result = stats.relative_error(jnp.array(0.5), jnp.array(1.0))
        assert result >= 0


class TestCheckConvergence:
    """Tests for check_convergence function."""

    def test_converged(self):
        """check_convergence should return True for small changes."""
        x_old = jnp.array([1.0, 2.0, 3.0])
        x_new = jnp.array([1.0000001, 2.0000001, 3.0000001])
        assert stats.check_convergence(x_new, x_old, tol=1e-5)

    def test_not_converged(self):
        """check_convergence should return False for large changes."""
        x_old = jnp.array([1.0, 2.0, 3.0])
        x_new = jnp.array([1.1, 2.0, 3.0])
        assert not stats.check_convergence(x_new, x_old, tol=1e-5)

    def test_uses_max_error(self):
        """check_convergence should use maximum error across array."""
        x_old = jnp.array([1.0, 1.0])
        x_new = jnp.array([1.0000001, 1.1])  # Second element has 10% error
        assert not stats.check_convergence(x_new, x_old, tol=1e-5)


class TestLogsumexp:
    """Tests for logsumexp function."""

    def test_basic(self):
        """logsumexp should compute log(sum(exp(x)))."""
        x = jnp.array([0.0, 0.0])  # log(exp(0) + exp(0)) = log(2)
        result = stats.logsumexp(x)
        assert jnp.allclose(result, jnp.log(2.0))

    def test_stability(self):
        """logsumexp should handle large values stably."""
        x = jnp.array([1000.0, 1000.0])
        result = stats.logsumexp(x)
        # log(2 * exp(1000)) = 1000 + log(2)
        expected = 1000.0 + jnp.log(2.0)
        assert jnp.allclose(result, expected)


class TestInterp1d:
    """Tests for interp1d function."""

    def test_exact_points(self):
        """Interpolation at grid points should be exact."""
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([0.0, 1.0, 4.0])
        result = interpolation.interp1d(x, y, x)
        assert jnp.allclose(result, y)

    def test_midpoint(self):
        """Linear interpolation at midpoint."""
        x = jnp.array([0.0, 2.0])
        y = jnp.array([0.0, 4.0])
        result = interpolation.interp1d(x, y, jnp.array(1.0))
        assert jnp.allclose(result, 2.0)

    def test_extrapolation_clamped(self):
        """Values outside range should be clamped by default."""
        x = jnp.array([0.0, 1.0])
        y = jnp.array([0.0, 1.0])
        result = interpolation.interp1d(x, y, jnp.array(-1.0))
        assert jnp.allclose(result, 0.0)  # Clamped to first value


class TestTabulatedFunction1D:
    """Tests for TabulatedFunction1D class."""

    def test_callable(self):
        """TabulatedFunction1D should be callable."""
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([0.0, 1.0, 4.0])
        f = interpolation.TabulatedFunction1D(x, y)
        result = f(jnp.array(0.5))
        assert jnp.allclose(result, 0.5)

    def test_pytree(self):
        """TabulatedFunction1D should be a valid pytree."""
        x = jnp.array([0.0, 1.0])
        y = jnp.array([0.0, 1.0])
        f = interpolation.TabulatedFunction1D(x, y)

        # Should work in jit
        @jax.jit
        def use_table(table, x_new):
            return table(x_new)

        result = use_table(f, jnp.array(0.5))
        assert jnp.allclose(result, 0.5)


class TestCumulativeSimpson:
    """Tests for cumulative Simpson panel integration."""

    def test_cumulative_simpson_panel_endpoints_exact_for_cubic(self):
        x = jnp.linspace(0.0, 4.0, 9)
        y = x**3

        result = integration.cumulative_simpson(y, x)

        expected = x[::2] ** 4 / 4.0
        assert result.shape == expected.shape
        assert jnp.allclose(result, expected, atol=1e-12)

    def test_cumulative_simpson_supports_axis_and_dx(self):
        x = jnp.linspace(0.0, 4.0, 9)
        y = jnp.stack([x**2, 2.0 * x**2], axis=0)

        result = integration.cumulative_simpson(y, dx=0.5, axis=-1)

        expected = jnp.stack([x[::2] ** 3 / 3.0, 2.0 * x[::2] ** 3 / 3.0], axis=0)
        assert result.shape == (2, 5)
        assert jnp.allclose(result, expected, atol=1e-12)

    def test_cumulative_simpson_rejects_even_sample_count(self):
        with pytest.raises(ValueError, match="odd number"):
            integration.cumulative_simpson(jnp.ones(4))

    def test_cumulative_simpson_rejects_nonuniform_x(self):
        with pytest.raises(ValueError, match="uniform spacing"):
            integration.cumulative_simpson(
                jnp.ones(5),
                jnp.array([0.0, 0.5, 1.5, 2.0, 3.0]),
            )


class TestBisect:
    """Tests for bisect rootfinding."""

    def test_sqrt2(self):
        """Find sqrt(2) via bisection."""

        @jax.jit
        def find_root(a, b):
            return rootfinding.bisect(lambda x: x**2 - 2.0, a, b)

        root = find_root(jnp.array(1.0), jnp.array(2.0))
        assert jnp.allclose(root, jnp.sqrt(2.0), atol=1e-7)

    def test_linear(self):
        """Find root of linear function."""

        @jax.jit
        def find_root(a, b):
            return rootfinding.bisect(lambda x: 2.0 * x - 1.0, a, b)

        root = find_root(jnp.array(0.0), jnp.array(1.0))
        assert jnp.allclose(root, 0.5, atol=1e-7)

    def test_vmap_compatible(self):
        """bisect should work with vmap over brackets."""

        @jax.jit
        def find_roots(a_arr, b_arr):
            def solve_one(a, b):
                return rootfinding.bisect(lambda x: x**2 - 2.0, a, b)

            return jax.vmap(solve_one)(a_arr, b_arr)

        a_arr = jnp.array([1.0, 1.0])
        b_arr = jnp.array([2.0, 3.0])
        roots = find_roots(a_arr, b_arr)
        assert jnp.allclose(roots, jnp.sqrt(2.0), atol=1e-7)

    def test_grad_compatible(self):
        """bisect should be differentiable w.r.t. brackets."""

        @jax.jit
        def find_root(a, b):
            return rootfinding.bisect(lambda x: x**2 - 2.0, a, b)

        # Gradient w.r.t. left bracket should exist and be finite
        grad_fn = jax.grad(find_root, argnums=0)
        g = grad_fn(jnp.array(1.0), jnp.array(2.0))
        assert jnp.isfinite(g)


class TestNewton:
    """Tests for newton rootfinding (auto-grad version)."""

    def test_sqrt2(self):
        """Find sqrt(2) via Newton's method with auto derivative."""

        @jax.jit
        def find_root(x0):
            return rootfinding.newton(lambda x: x**2 - 2.0, x0)

        root = find_root(jnp.array(1.5))
        assert jnp.allclose(root, jnp.sqrt(2.0), atol=1e-9)

    def test_vmap_compatible(self):
        """newton should work with vmap over initial guesses."""

        @jax.jit
        def find_roots(x0_arr):
            def solve_one(x0):
                return rootfinding.newton(lambda x: x**2 - 2.0, x0)

            return jax.vmap(solve_one)(x0_arr)

        x0_arr = jnp.array([1.5, 2.0, 1.0])
        roots = find_roots(x0_arr)
        assert jnp.allclose(roots, jnp.sqrt(2.0), atol=1e-9)

    def test_grad_compatible(self):
        """newton should be differentiable w.r.t. initial guess."""

        @jax.jit
        def find_root(x0):
            return rootfinding.newton(lambda x: x**2 - 2.0, x0)

        # Gradient should exist and be finite
        grad_fn = jax.grad(find_root)
        g = grad_fn(jnp.array(1.5))
        assert jnp.isfinite(g)


class TestNewtonWithGrad:
    """Tests for newton_with_grad rootfinding (explicit derivative)."""

    def test_sqrt2(self):
        """Find sqrt(2) via Newton's method with explicit derivative."""

        @jax.jit
        def find_root(x0):
            return rootfinding.newton_with_grad(
                lambda x: x**2 - 2.0, lambda x: 2.0 * x, x0
            )

        root = find_root(jnp.array(1.5))
        assert jnp.allclose(root, jnp.sqrt(2.0), atol=1e-9)

    def test_grad_compatible(self):
        """newton_with_grad should be differentiable w.r.t. initial guess."""

        @jax.jit
        def find_root(x0):
            return rootfinding.newton_with_grad(
                lambda x: x**2 - 2.0, lambda x: 2.0 * x, x0
            )

        # Gradient should exist and be finite
        grad_fn = jax.grad(find_root)
        g = grad_fn(jnp.array(1.5))
        assert jnp.isfinite(g)


class TestNewtonPPF:
    """Tests for newton_ppf: generic fixed-iteration Newton PPF inversion.

    Analytic case: exponential distribution.
        CDF: F(x) = 1 - exp(-lam * x)
        PPF: x = -ln(1 - u) / lam   (closed form)
    """

    @staticmethod
    def _exp_cdf(x, lam):
        return 1.0 - jnp.exp(-lam * x)

    @staticmethod
    def _exp_ppf_true(u, lam):
        return -jnp.log1p(-u) / lam

    def test_matches_analytic_ppf(self):
        """newton_ppf should match the closed-form exponential PPF to ~1e-6."""
        lam = 1.7
        u = jnp.linspace(0.01, 0.99, 99)
        x0 = jnp.full_like(u, 1.0)  # neutral starting guess

        ppf = rootfinding.newton_ppf(
            u,
            lambda x: self._exp_cdf(x, lam),
            x0=x0,
            lo=0.0,
            hi=100.0,
        )
        expected = self._exp_ppf_true(u, lam)
        assert jnp.allclose(ppf, expected, atol=1e-6, rtol=1e-6)

    def test_jit_vmap_compatible(self):
        """newton_ppf should compose with jit (vectorized over u)."""
        lam = 2.3

        @jax.jit
        def solve(u):
            return rootfinding.newton_ppf(
                u,
                lambda x: self._exp_cdf(x, lam),
                x0=jnp.ones_like(u),
                lo=0.0,
                hi=100.0,
            )

        u = jnp.linspace(0.05, 0.95, 50)
        ppf = solve(u)
        assert jnp.allclose(ppf, self._exp_ppf_true(u, lam), atol=1e-6)

    def test_differentiable_wrt_u(self):
        """d(PPF)/du via jax.grad should match finite differences (~1e-6).

        Analytic: d/du [-ln(1-u)/lam] = 1 / (lam * (1 - u)).
        """
        lam = 1.3
        u0 = 0.4

        def solve(u):
            return rootfinding.newton_ppf(
                u,
                lambda x: self._exp_cdf(x, lam),
                x0=1.0,
                lo=0.0,
                hi=100.0,
            )

        ad = jax.grad(solve)(u0)
        eps = 1e-6
        fd = (solve(u0 + eps) - solve(u0 - eps)) / (2 * eps)
        analytic = 1.0 / (lam * (1.0 - u0))
        assert jnp.allclose(ad, fd, atol=1e-5, rtol=1e-5)
        assert jnp.allclose(ad, analytic, atol=1e-5, rtol=1e-5)

    def test_differentiable_wrt_param(self):
        """d(PPF)/d(lam) via jax.grad should match finite differences (~1e-6).

        Analytic: d/dlam [-ln(1-u)/lam] = ln(1-u)/lam**2 = -PPF/lam.
        """
        u0 = 0.6
        lam0 = 1.9

        def solve(lam):
            return rootfinding.newton_ppf(
                jnp.asarray(u0),
                lambda x: self._exp_cdf(x, lam),
                x0=1.0,
                lo=0.0,
                hi=100.0,
            )

        ad = jax.grad(solve)(lam0)
        eps = 1e-6
        fd = (solve(lam0 + eps) - solve(lam0 - eps)) / (2 * eps)
        analytic = jnp.log1p(-u0) / lam0**2
        assert jnp.allclose(ad, fd, atol=1e-5, rtol=1e-5)
        assert jnp.allclose(ad, analytic, atol=1e-5, rtol=1e-5)

    def test_explicit_pdf_path(self):
        """Passing an explicit pdf (CDF derivative) should also converge."""
        lam = 0.8
        u = jnp.linspace(0.05, 0.95, 40)
        ppf = rootfinding.newton_ppf(
            u,
            lambda x: self._exp_cdf(x, lam),
            x0=jnp.ones_like(u),
            lo=0.0,
            hi=200.0,
            pdf=lambda x: lam * jnp.exp(-lam * x),
        )
        assert jnp.allclose(ppf, self._exp_ppf_true(u, lam), atol=1e-6)


class TestTrapz:
    """Tests for trapezoidal integration."""

    def test_constant(self):
        """Integral of constant should be constant * length."""
        y = jnp.ones(5)
        x = jnp.linspace(0.0, 1.0, 5)
        result = integration.trapz(y, x)
        assert jnp.allclose(result, 1.0)

    def test_linear(self):
        """Integral of x from 0 to 1 should be 0.5."""
        x = jnp.linspace(0.0, 1.0, 101)
        y = x
        result = integration.trapz(y, x)
        assert jnp.allclose(result, 0.5, atol=1e-4)


class TestCumulativeTrapz:
    """Tests for cumulative trapezoidal integration."""

    def test_starts_at_zero(self):
        """Cumulative integral should start at 0."""
        y = jnp.ones(5)
        x = jnp.linspace(0.0, 1.0, 5)
        result = integration.cumulative_trapz(y, x)
        assert result[0] == 0.0

    def test_ends_at_total(self):
        """Cumulative integral should end at total integral."""
        y = jnp.ones(5)
        x = jnp.linspace(0.0, 1.0, 5)
        cumulative = integration.cumulative_trapz(y, x)
        total = integration.trapz(y, x)
        assert jnp.allclose(cumulative[-1], total)


class TestJAXTransforms:
    """Tests for JAX transform compatibility."""

    def test_grad_through_safe_functions(self):
        """Gradients should work through safe_* functions."""

        def f(x):
            return stats.safe_log(stats.safe_exp(x))

        grad_f = jax.grad(f)
        result = grad_f(jnp.array(1.0))
        # d/dx log(exp(x)) = 1
        assert jnp.allclose(result, 1.0)

    def test_vmap_interpolation(self):
        """vmap should work over interpolation queries."""
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([0.0, 1.0, 4.0])
        x_new = jnp.array([[0.5], [1.5]])

        result = jax.vmap(lambda xn: interpolation.interp1d(x, y, xn))(x_new)
        assert result.shape == (2, 1)


class TestCompensatedAccuracy:
    """Accuracy tests for Neumaier compensated summation.

    The whole point of the module is to beat naive float summation under
    catastrophic cancellation. The classic witness is [1e16, 1, -1e16, 1]:
    naive float64 summation loses the small terms and returns 0.0, while
    Neumaier summation recovers the exact answer 2.0.
    """

    def test_catastrophic_cancellation_array(self):
        terms = jnp.array([1e16, 1.0, -1e16, 1.0])
        # Sanity: naive summation does not recover 2.0 (loses the small terms).
        naive = jnp.sum(terms)
        assert not jnp.allclose(naive, 2.0)
        # Neumaier recovers the exact answer.
        result = compensated.compensated_sum_array(terms)
        assert jnp.allclose(result, 2.0)

    def test_catastrophic_cancellation_variadic(self):
        result = compensated.compensated_sum(
            jnp.array(1e16), jnp.array(1.0), jnp.array(-1e16), jnp.array(1.0)
        )
        assert jnp.allclose(result, 2.0)

    def test_compensated_dot_cancellation(self):
        a = jnp.array([1e16, 1.0, -1e16, 1.0])
        b = jnp.array([1.0, 1.0, 1.0, 1.0])
        # Naive dot loses precision and does NOT recover the exact 2.0.
        assert not jnp.allclose(jnp.dot(a, b), 2.0)
        result = compensated.compensated_dot(a, b)
        assert jnp.allclose(result, 2.0)

    def test_vector_sum_cancellation(self):
        vecs = jnp.array([[1e16, 1.0], [1.0, -1e16], [-1e16, 1e16], [1.0, 1.0]])
        result = compensated.compensated_vector_sum(vecs)
        assert jnp.allclose(result, jnp.array([2.0, 2.0]))

    def test_grad_through_compensated_sum(self):
        # Compensated sum is linear in its inputs; gradient should be all-ones.
        def f(terms):
            return compensated.compensated_sum_array(terms)

        g = jax.grad(f)(jnp.array([1e16, 1.0, -1e16, 1.0]))
        assert jnp.allclose(g, jnp.ones(4))


class TestSimpson:
    """Tests for Simpson's rule, including uniform-spacing validation."""

    def test_sin_integral(self):
        # ∫_0^π sin(x) dx = 2.
        x = jnp.linspace(0.0, jnp.pi, 101)
        y = jnp.sin(x)
        result = integration.simpson(y, x)
        assert jnp.allclose(result, 2.0, atol=1e-6)

    def test_polynomial_exact(self):
        # Simpson is exact for cubics: ∫_0^2 x^3 dx = 4.
        x = jnp.linspace(0.0, 2.0, 5)
        y = x**3
        result = integration.simpson(y, x)
        assert jnp.allclose(result, 4.0, atol=1e-10)

    def test_default_dx(self):
        # Without x, dx defaults to 1; ∫ over indices 0..2 of [1,1,1] = 2.
        y = jnp.ones(3)
        result = integration.simpson(y)
        assert jnp.allclose(result, 2.0)

    def test_even_points_raises(self):
        with pytest.raises(ValueError, match="odd number"):
            integration.simpson(jnp.ones(4))

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="odd number"):
            integration.simpson(jnp.ones(1))

    def test_nonuniform_spacing_raises(self):
        # Simpson assumes uniform spacing; a non-uniform x must be rejected
        # (eager/host-side debug check) rather than silently mis-integrate.
        x = jnp.array([0.0, 1.0, 3.0])  # spacings 1.0 and 2.0 differ
        y = x**2
        with pytest.raises(ValueError, match="uniform"):
            integration.simpson(y, x)

    def test_uniform_spacing_ok(self):
        x = jnp.array([0.0, 1.0, 2.0])  # uniform
        y = x**2
        result = integration.simpson(y, x)
        # ∫_0^2 x^2 dx = 8/3.
        assert jnp.allclose(result, 8.0 / 3.0)

    def test_jit_compatible(self):
        # Under jit the eager uniformity check is skipped (tracers); the
        # numerics must still work for a uniform grid.
        x = jnp.linspace(0.0, jnp.pi, 101)
        y = jnp.sin(x)
        result = jax.jit(lambda x, y: integration.simpson(y, x))(x, y)
        assert jnp.allclose(result, 2.0, atol=1e-6)


class TestInterp1dValidation:
    """Monotonic-x validation for interp1d."""

    def test_non_monotonic_raises(self):
        x = jnp.array([0.0, 2.0, 1.0])  # not strictly increasing
        y = jnp.array([0.0, 1.0, 2.0])
        with pytest.raises(ValueError, match="strictly increasing"):
            interpolation.interp1d(x, y, jnp.array(0.5))

    def test_repeated_x_raises(self):
        x = jnp.array([0.0, 1.0, 1.0, 2.0])  # not strictly increasing (tie)
        y = jnp.array([0.0, 1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="strictly increasing"):
            interpolation.interp1d(x, y, jnp.array(0.5))

    def test_monotonic_x_ok(self):
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([0.0, 1.0, 4.0])
        result = interpolation.interp1d(x, y, jnp.array(1.5))
        assert jnp.allclose(result, 2.5)

    def test_jit_skips_check(self):
        # Under jit the eager check is skipped; a valid monotonic grid works.
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([0.0, 1.0, 4.0])
        result = jax.jit(lambda xn: interpolation.interp1d(x, y, xn))(jnp.array(0.5))
        assert jnp.allclose(result, 0.5)
