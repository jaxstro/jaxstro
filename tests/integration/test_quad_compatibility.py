import subprocess
import sys


def test_canonical_and_legacy_paths_match_in_a_clean_process() -> None:
    code = r"""
import jax.numpy as jnp
import jaxstro
from jaxstro.numerics import integration, quadrature

y = jnp.array([1.0, -2.0, 3.0, -4.0, 5.0])
assert jnp.array_equal(
    jaxstro.quad.cumulative_trapezoid(y, dx=0.3),
    integration.cumulative_trapz(y, dx=0.3),
)
assert jaxstro.quad.gauss_hermite_nodes is quadrature.gauss_hermite_nodes
assert jaxstro.numerics.gauss_hermite_nodes is jaxstro.quad.gauss_hermite_nodes
assert jaxstro.quad.gauss_legendre_nodes is quadrature.gauss_legendre_nodes
assert jaxstro.quad.gauss_laguerre_nodes is quadrature.gauss_laguerre_nodes
assert jaxstro.quad.clenshaw_curtis_nodes is quadrature.clenshaw_curtis_nodes
assert jaxstro.quad.gauss_legendre_nodes.__module__ == "jaxstro.quad._recurrence"
assert jaxstro.quad.gauss_laguerre_nodes.__module__ == "jaxstro.quad._recurrence"
assert jaxstro.quad.clenshaw_curtis_nodes.__module__ == "jaxstro.quad._chebyshev"
"""
    subprocess.run([sys.executable, "-c", code], check=True)
