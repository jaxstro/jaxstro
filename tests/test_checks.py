# tests/test_checks.py
"""
Tests for jaxstro.numerics.checks module.

Verifies numerical validation helpers work correctly with JAX transforms.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxstro.numerics import checks


class TestIsFinite:
    """Tests for is_finite function."""

    def test_finite_values(self):
        """is_finite should return True for finite values."""
        x = jnp.array([1.0, 2.0, 3.0])
        result = checks.is_finite(x)
        assert jnp.all(result)

    def test_nan_detected(self):
        """is_finite should return False for NaN."""
        x = jnp.array([1.0, jnp.nan, 3.0])
        result = checks.is_finite(x)
        expected = jnp.array([True, False, True])
        assert jnp.array_equal(result, expected)

    def test_inf_detected(self):
        """is_finite should return False for inf."""
        x = jnp.array([1.0, jnp.inf, -jnp.inf])
        result = checks.is_finite(x)
        expected = jnp.array([True, False, False])
        assert jnp.array_equal(result, expected)

    def test_jit_compatible(self):
        """is_finite should work under jit."""
        f = jax.jit(checks.is_finite)
        x = jnp.array([1.0, jnp.nan])
        result = f(x)
        assert result[0] and not result[1]


class TestAllFinite:
    """Tests for all_finite function."""

    def test_all_finite_true(self):
        """all_finite should return True when all values finite."""
        x = jnp.array([1.0, 2.0, 3.0])
        assert checks.all_finite(x)

    def test_all_finite_false_nan(self):
        """all_finite should return False when NaN present."""
        x = jnp.array([1.0, jnp.nan, 3.0])
        assert not checks.all_finite(x)

    def test_all_finite_false_inf(self):
        """all_finite should return False when inf present."""
        x = jnp.array([1.0, jnp.inf, 3.0])
        assert not checks.all_finite(x)

    def test_vmap_compatible(self):
        """all_finite should work with vmap."""
        xs = jnp.array([[1.0, 2.0], [jnp.nan, 1.0]])
        result = jax.vmap(checks.all_finite)(xs)
        expected = jnp.array([True, False])
        assert jnp.array_equal(result, expected)


class TestAssertAllFinite:
    """Tests for assert_all_finite function."""

    def test_passes_for_finite(self):
        """assert_all_finite should not raise for finite values."""
        x = jnp.array([1.0, 2.0, 3.0])
        checks.assert_all_finite(x)  # Should not raise

    def test_raises_for_nan(self):
        """assert_all_finite should raise ValueError for NaN."""
        x = jnp.array([1.0, jnp.nan, 3.0])
        with pytest.raises(ValueError, match="non-finite"):
            checks.assert_all_finite(x, name="test_array")

    def test_raises_for_inf(self):
        """assert_all_finite should raise ValueError for inf."""
        x = jnp.array([jnp.inf])
        with pytest.raises(ValueError, match="non-finite"):
            checks.assert_all_finite(x)


class TestIsMonotonic:
    """Tests for is_monotonic functions."""

    def test_strictly_increasing(self):
        """is_monotonic should detect strictly increasing."""
        x = jnp.array([1.0, 2.0, 3.0])
        assert checks.is_monotonic(x, strict=True)

    def test_non_strictly_increasing(self):
        """is_monotonic with strict=False should allow equal values."""
        x = jnp.array([1.0, 2.0, 2.0, 3.0])
        assert not checks.is_monotonic(x, strict=True)
        assert checks.is_monotonic(x, strict=False)

    def test_not_monotonic(self):
        """is_monotonic should return False for non-monotonic."""
        x = jnp.array([1.0, 3.0, 2.0])
        assert not checks.is_monotonic(x, strict=True)
        assert not checks.is_monotonic(x, strict=False)

    def test_decreasing(self):
        """is_monotonic_decreasing should detect decreasing."""
        x = jnp.array([3.0, 2.0, 1.0])
        assert checks.is_monotonic_decreasing(x, strict=True)
        assert not checks.is_monotonic_increasing(x)

    def test_jit_compatible(self):
        """is_monotonic should work under jit."""
        @jax.jit
        def check(x):
            return checks.is_monotonic_increasing(x, strict=True)

        x = jnp.array([1.0, 2.0, 3.0])
        assert check(x)


class TestAssertMonotonic:
    """Tests for assert_monotonic function."""

    def test_passes_for_monotonic(self):
        """assert_monotonic should not raise for monotonic array."""
        x = jnp.array([1.0, 2.0, 3.0])
        checks.assert_monotonic(x)  # Should not raise

    def test_raises_for_non_monotonic(self):
        """assert_monotonic should raise for non-monotonic."""
        x = jnp.array([1.0, 3.0, 2.0])
        with pytest.raises(ValueError, match="not monotonic"):
            checks.assert_monotonic(x)

    def test_raises_for_non_1d(self):
        """assert_monotonic should raise for non-1D input."""
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="must be 1D"):
            checks.assert_monotonic(x)

    def test_decreasing_mode(self):
        """assert_monotonic should support decreasing mode."""
        x = jnp.array([3.0, 2.0, 1.0])
        checks.assert_monotonic(x, decreasing=True)  # Should not raise

        # When checking for increasing on a decreasing array, should fail
        with pytest.raises(ValueError, match="not monotonic increasing"):
            checks.assert_monotonic(x, decreasing=False)


class TestInRange:
    """Tests for in_range function."""

    def test_all_in_range(self):
        """in_range should return True for values in range."""
        x = jnp.array([1.0, 2.0, 3.0])
        result = checks.in_range(x, lo=0.0, hi=5.0)
        assert jnp.all(result)

    def test_out_of_range_lo(self):
        """in_range should detect values below lower bound."""
        x = jnp.array([-1.0, 1.0, 2.0])
        result = checks.in_range(x, lo=0.0)
        expected = jnp.array([False, True, True])
        assert jnp.array_equal(result, expected)

    def test_out_of_range_hi(self):
        """in_range should detect values above upper bound."""
        x = jnp.array([1.0, 5.0, 10.0])
        result = checks.in_range(x, hi=5.0)
        expected = jnp.array([True, True, False])
        assert jnp.array_equal(result, expected)

    def test_exclusive_bounds(self):
        """in_range with inclusive=False should use strict comparison."""
        x = jnp.array([0.0, 0.5, 1.0])
        result = checks.in_range(x, lo=0.0, hi=1.0, inclusive=False)
        expected = jnp.array([False, True, False])
        assert jnp.array_equal(result, expected)

    def test_jit_compatible(self):
        """in_range should work under jit."""
        @jax.jit
        def check(x):
            return checks.in_range(x, lo=0.0, hi=1.0, inclusive=True)

        x = jnp.array([0.0, 0.5, 1.0])
        assert jnp.all(check(x))


class TestAllInRange:
    """Tests for all_in_range function."""

    def test_all_in_range_true(self):
        """all_in_range should return True when all values in range."""
        x = jnp.array([1.0, 2.0, 3.0])
        assert checks.all_in_range(x, lo=0.0, hi=5.0)

    def test_all_in_range_false(self):
        """all_in_range should return False when values out of range."""
        x = jnp.array([1.0, 10.0, 3.0])
        assert not checks.all_in_range(x, lo=0.0, hi=5.0)


class TestAllPositive:
    """Tests for all_positive function."""

    def test_all_positive_true(self):
        """all_positive should return True for positive values."""
        x = jnp.array([1.0, 2.0, 3.0])
        assert checks.all_positive(x)

    def test_all_positive_false_zero(self):
        """all_positive should return False when zero present."""
        x = jnp.array([0.0, 1.0, 2.0])
        assert not checks.all_positive(x)

    def test_all_positive_false_negative(self):
        """all_positive should return False for negative values."""
        x = jnp.array([1.0, -1.0, 2.0])
        assert not checks.all_positive(x)


class TestAllNonNegative:
    """Tests for all_non_negative function."""

    def test_all_non_negative_true(self):
        """all_non_negative should return True for non-negative values."""
        x = jnp.array([0.0, 1.0, 2.0])
        assert checks.all_non_negative(x)

    def test_all_non_negative_false(self):
        """all_non_negative should return False for negative values."""
        x = jnp.array([1.0, -1.0, 2.0])
        assert not checks.all_non_negative(x)


class TestAssertInRange:
    """Tests for assert_in_range function."""

    def test_passes_for_in_range(self):
        """assert_in_range should not raise for values in range."""
        x = jnp.array([1.0, 2.0, 3.0])
        checks.assert_in_range(x, lo=0.0, hi=5.0)  # Should not raise

    def test_raises_for_out_of_range(self):
        """assert_in_range should raise for values out of range."""
        x = jnp.array([1.0, 10.0, 3.0])
        with pytest.raises(ValueError, match="out of range"):
            checks.assert_in_range(x, lo=0.0, hi=5.0, name="test_array")


class TestAssertPositive:
    """Tests for assert_positive function."""

    def test_passes_for_positive(self):
        """assert_positive should not raise for positive values."""
        x = jnp.array([1.0, 2.0, 3.0])
        checks.assert_positive(x)  # Should not raise

    def test_raises_for_zero(self):
        """assert_positive should raise when zero present."""
        x = jnp.array([0.0, 1.0])
        with pytest.raises(ValueError):
            checks.assert_positive(x)


class TestJAXTransforms:
    """Tests for JAX transform compatibility."""

    def test_vmap_all_finite(self):
        """vmap should work with all_finite."""
        xs = jnp.array([[1.0, 2.0], [jnp.nan, 1.0], [3.0, 4.0]])
        result = jax.vmap(checks.all_finite)(xs)
        expected = jnp.array([True, False, True])
        assert jnp.array_equal(result, expected)

    def test_vmap_all_positive(self):
        """vmap should work with all_positive."""
        xs = jnp.array([[1.0, 2.0], [-1.0, 1.0], [3.0, 4.0]])
        result = jax.vmap(checks.all_positive)(xs)
        expected = jnp.array([True, False, True])
        assert jnp.array_equal(result, expected)

    def test_jit_combined_checks(self):
        """Multiple checks should compose under jit."""
        @jax.jit
        def validate(x):
            return checks.all_finite(x) & checks.all_positive(x)

        x_good = jnp.array([1.0, 2.0, 3.0])
        x_nan = jnp.array([1.0, jnp.nan, 3.0])
        x_neg = jnp.array([1.0, -1.0, 3.0])

        assert validate(x_good)
        assert not validate(x_nan)
        assert not validate(x_neg)
