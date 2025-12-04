# tests/test_jaxconfig.py
"""
Tests for jaxstro.jaxconfig module.

Note: These tests verify the module structure and that the function
is callable. Full functional testing of JAX config changes is tricky
because config is global and our conftest.py already enables x64.
"""

import pytest

from jaxstro import jaxconfig


class TestEnableHighPrecision:
    """Tests for enable_high_precision function."""

    def test_callable(self):
        """enable_high_precision should be callable."""
        assert callable(jaxconfig.enable_high_precision)

    def test_returns_none(self):
        """enable_high_precision should return None."""
        result = jaxconfig.enable_high_precision()
        assert result is None

    def test_idempotent(self):
        """Calling enable_high_precision multiple times should be safe."""
        # Should not raise
        jaxconfig.enable_high_precision()
        jaxconfig.enable_high_precision()
        jaxconfig.enable_high_precision()

    def test_x64_enabled(self):
        """After calling, float64 should be available."""
        import jax
        import jax.numpy as jnp

        jaxconfig.enable_high_precision()

        # Create a float64 array
        x = jnp.array(1.0, dtype=jnp.float64)
        assert x.dtype == jnp.float64

    def test_default_dtype_is_float64(self):
        """After calling, default float should be float64."""
        import jax
        import jax.numpy as jnp

        jaxconfig.enable_high_precision()

        # Default dtype should be float64
        x = jnp.array(1.0)
        assert x.dtype == jnp.float64


class TestModuleExports:
    """Tests for module-level exports."""

    def test_has_enable_high_precision(self):
        """Module should export enable_high_precision."""
        assert hasattr(jaxconfig, "enable_high_precision")
