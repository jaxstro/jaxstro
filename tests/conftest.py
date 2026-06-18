# tests/conftest.py
"""
Pytest configuration for jaxstro tests.

Ensures float64 is enabled before any tests run, and auto-applies the
tier marker (unit/integration/validation) to each test from its path.
"""

import pathlib

import pytest

_TIERS = ("unit", "integration", "validation")


def pytest_configure(config):
    """Enable JAX float64 before any tests run."""
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")


def pytest_collection_modifyitems(config, items):
    """Auto-apply the tier marker (unit/integration/validation) from each test's path."""
    for item in items:
        parts = pathlib.Path(str(item.fspath)).parts
        for tier in _TIERS:
            if tier in parts:
                item.add_marker(getattr(pytest.mark, tier))
                break
