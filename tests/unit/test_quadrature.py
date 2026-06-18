# tests/test_quadrature.py
"""
Tests for jaxstro.numerics.quadrature.

Covers:
- Gauss-Legendre nodes integrate polynomials exactly to degree 2n-1.
- Gauss-Hermite (probabilists') integrate Gaussian moments exactly.
- Hermite-e recurrence matches closed forms.
- ``hermite_coefficients`` matches a known Hermite-e expansion.
- Byte-identical parity vs the progenax ``_gauss_hermite`` construction.
- FD-vs-AD grad-check through ``hermite_coefficients``.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from jaxstro.numerics import quadrature


def _double_factorial(k: int) -> float:
    """(k)!! for odd k -> product of odd integers; (-1)!! = 1."""
    r = 1.0
    while k > 1:
        r *= k
        k -= 2
    return r


class TestGaussLegendre:
    """Gauss-Legendre exactness on [-1, 1]."""

    def test_returns_pair_correct_shape(self):
        nodes, weights = quadrature.gauss_legendre_nodes(8)
        assert nodes.shape == (8,)
        assert weights.shape == (8,)

    def test_weights_sum_to_interval_length(self):
        """sum of GL weights on [-1,1] equals 2."""
        _, weights = quadrature.gauss_legendre_nodes(16)
        assert jnp.allclose(weights.sum(), 2.0)

    def test_exact_to_degree_2n_minus_1(self):
        """n-point GL integrates x^k exactly for k <= 2n-1."""
        n = 5
        nodes, weights = quadrature.gauss_legendre_nodes(n)
        for k in range(2 * n):  # 0 .. 2n-1
            approx = jnp.sum(weights * nodes**k)
            # exact integral of x^k on [-1,1]
            exact = 0.0 if (k % 2 == 1) else 2.0 / (k + 1)
            assert jnp.allclose(approx, exact, atol=1e-12), f"k={k}"

    def test_fails_above_degree_2n_minus_1(self):
        """Sanity: degree 2n is NOT exact (guards against trivially-passing test)."""
        n = 3
        nodes, weights = quadrature.gauss_legendre_nodes(n)
        approx = jnp.sum(weights * nodes ** (2 * n))
        exact = 2.0 / (2 * n + 1)
        assert not jnp.allclose(approx, exact, atol=1e-12)


class TestGaussHermite:
    """Gauss-Hermite (probabilists') moments under N(0,1)."""

    def test_returns_pair_correct_shape(self):
        nodes, weights = quadrature.gauss_hermite_nodes(32)
        assert nodes.shape == (32,)
        assert weights.shape == (32,)

    def test_weights_normalized(self):
        """Probabilists' weights sum to 1 (expectation under N(0,1))."""
        _, weights = quadrature.gauss_hermite_nodes(64)
        assert jnp.allclose(weights.sum(), 1.0)

    def test_gaussian_moments_exact(self):
        """E[g^{2m}] = (2m-1)!! under N(0,1); odd moments vanish."""
        n = 32
        nodes, weights = quadrature.gauss_hermite_nodes(n)
        for m in range(1, 6):
            even = jnp.sum(weights * nodes ** (2 * m))
            assert jnp.allclose(even, _double_factorial(2 * m - 1), atol=1e-9), m
            odd = jnp.sum(weights * nodes ** (2 * m - 1))
            assert jnp.allclose(odd, 0.0, atol=1e-12), m


class TestHermiteEBasis:
    """Probabilists' Hermite He_n recurrence."""

    def test_closed_forms(self):
        """He_0..He_4 match closed forms at sample points."""
        g = jnp.array([-1.3, 0.0, 0.7, 2.1])
        he = quadrature.hermite_e_basis(g, 4)
        assert he.shape == (5, g.shape[0])
        He0 = jnp.ones_like(g)
        He1 = g
        He2 = g**2 - 1.0
        He3 = g**3 - 3.0 * g
        He4 = g**4 - 6.0 * g**2 + 3.0
        assert jnp.allclose(he[0], He0)
        assert jnp.allclose(he[1], He1)
        assert jnp.allclose(he[2], He2)
        assert jnp.allclose(he[3], He3)
        assert jnp.allclose(he[4], He4)

    def test_orthogonality_under_gh(self):
        """<He_i He_j>_{N(0,1)} = i! delta_ij via GH quadrature."""
        g, w = quadrature.gauss_hermite_nodes(48)
        he = quadrature.hermite_e_basis(g, 5)
        for i in range(6):
            for j in range(6):
                val = jnp.sum(w * he[i] * he[j])
                expected = math.factorial(i) if i == j else 0.0
                assert jnp.allclose(val, expected, atol=1e-8), (i, j)


class TestHermiteCoefficients:
    """Hermite-e expansion coefficients via GH quadrature."""

    def test_known_expansion(self):
        """For map(g) = He_2(g) + 3 He_3(g), coefficients are c_2=1, c_3=3."""

        def map_fn(g):
            He2 = g**2 - 1.0
            He3 = g**3 - 3.0 * g
            return He2 + 3.0 * He3

        c = quadrature.hermite_coefficients(map_fn, n_max=4, n_quad=64)
        # c_n = <map He_n>; with He orthogonal: c_n = (coeff of He_n) * n!
        assert jnp.allclose(c[0], 0.0, atol=1e-9)
        assert jnp.allclose(c[1], 0.0, atol=1e-9)
        assert jnp.allclose(c[2], 1.0 * math.factorial(2), atol=1e-8)
        assert jnp.allclose(c[3], 3.0 * math.factorial(3), atol=1e-8)
        assert jnp.allclose(c[4], 0.0, atol=1e-8)

    def test_mean_term(self):
        """c_0 = <map_fn> (the mean under N(0,1))."""
        c = quadrature.hermite_coefficients(lambda g: g**2, n_max=2, n_quad=32)
        assert jnp.allclose(c[0], 1.0, atol=1e-9)  # E[g^2] = 1


class TestParityWithProgenax:
    """Byte-identical to the progenax ``_gauss_hermite`` construction."""

    def test_gauss_hermite_nodes_byte_identical(self):
        """Same numpy call as progenax: hermgauss + sqrt(2) transform."""
        for n in (32, 64, 128, 256):
            x, w = np.polynomial.hermite.hermgauss(n)
            g_prog = jnp.asarray(np.sqrt(2.0) * x)
            w_prog = jnp.asarray(w / np.sqrt(np.pi))
            g_jax, w_jax = quadrature.gauss_hermite_nodes(n)
            assert jnp.array_equal(g_jax, g_prog), n
            assert jnp.array_equal(w_jax, w_prog), n


class TestGradient:
    """FD-vs-AD grad-check: gradient flows through integrand values, not nodes."""

    def test_hermite_coefficients_differentiable(self):
        """d c_n / d theta where theta is captured inside map_fn."""

        def coeff_sum(theta):
            # A smooth parameterized map; grad must flow through values.
            c = quadrature.hermite_coefficients(
                lambda g: jnp.exp(theta * g) - 1.0, n_max=4, n_quad=128
            )
            return jnp.sum(c**2)

        theta0 = 0.3
        ad = jax.grad(coeff_sum)(theta0)
        eps = 1e-6
        fd = (coeff_sum(theta0 + eps) - coeff_sum(theta0 - eps)) / (2 * eps)
        assert jnp.allclose(ad, fd, rtol=1e-5, atol=1e-6), (ad, fd)
