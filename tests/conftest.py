# tests/conftest.py
"""
Pytest configuration for jaxstro tests.

Ensures float64 is enabled before any tests run, and auto-applies the
tier marker (unit/integration/validation) to each test from its path.
"""

import pathlib

import pytest

_TIERS = ("unit", "integration", "validation")
_TESTS_ROOT = pathlib.Path(__file__).resolve().parent


def _tier_for(path: pathlib.Path, tests_root: pathlib.Path = _TESTS_ROOT) -> str | None:
    """The tier is the first directory below tests/, never one above the repo."""
    try:
        first = path.relative_to(tests_root).parts[0]
    except (ValueError, IndexError):
        return None
    return first if first in _TIERS else None


def pytest_configure(config):
    """Enable JAX float64 before any tests run."""
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")


def pytest_collection_modifyitems(config, items):
    """Auto-apply the tier marker (unit/integration/validation) from each test's path."""
    for item in items:
        tier = _tier_for(pathlib.Path(str(item.fspath)).resolve())
        if tier is not None:
            item.add_marker(getattr(pytest.mark, tier))
