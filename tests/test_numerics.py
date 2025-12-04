# tests/test_numerics.py
"""
Tests for jaxstro.numerics modules.

Verifies numerical utilities work correctly with JAX transforms.
"""

import math

import jax
import jax.numpy as jnp
import pytest

from jaxstro.numerics import stats, interpolation, rootfinding, integration


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
                lambda x: x**2 - 2.0,
                lambda x: 2.0 * x,
                x0
            )

        root = find_root(jnp.array(1.5))
        assert jnp.allclose(root, jnp.sqrt(2.0), atol=1e-9)

    def test_grad_compatible(self):
        """newton_with_grad should be differentiable w.r.t. initial guess."""
        @jax.jit
        def find_root(x0):
            return rootfinding.newton_with_grad(
                lambda x: x**2 - 2.0,
                lambda x: 2.0 * x,
                x0
            )

        # Gradient should exist and be finite
        grad_fn = jax.grad(find_root)
        g = grad_fn(jnp.array(1.5))
        assert jnp.isfinite(g)


class TestNewton1DAlias:
    """Tests for newton_1d backwards compatibility alias."""

    def test_is_alias(self):
        """newton_1d should be an alias for newton_with_grad."""
        assert rootfinding.newton_1d is rootfinding.newton_with_grad


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
