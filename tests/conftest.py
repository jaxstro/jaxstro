# tests/conftest.py
"""
Pytest configuration for jaxstro tests.

Ensures float64 is enabled before any tests run.
"""



def pytest_configure(config):
    """Enable JAX float64 before any tests run."""
    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
