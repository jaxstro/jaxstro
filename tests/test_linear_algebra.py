# tests/test_linear_algebra.py
"""
Tests for jaxstro.numerics.linear_algebra.

Covers norm2, project_onto, and condition_number, including the
degenerate b·b == 0 case for project_onto (must be finite, not NaN).
"""

import jax
import jax.numpy as jnp

from jaxstro.numerics import linear_algebra as la


class TestNorm2:
    """Tests for norm2 (Euclidean / ell-2 norm)."""

    def test_vector_norm(self):
        x = jnp.array([3.0, 4.0])
        assert jnp.allclose(la.norm2(x), 5.0)

    def test_matches_jnp_linalg(self):
        x = jnp.array([1.0, -2.0, 2.0, 4.0])
        assert jnp.allclose(la.norm2(x), jnp.linalg.norm(x, ord=2))

    def test_axis_and_keepdims(self):
        x = jnp.array([[3.0, 4.0], [5.0, 12.0]])
        result = la.norm2(x, axis=1, keepdims=True)
        assert result.shape == (2, 1)
        assert jnp.allclose(result[:, 0], jnp.array([5.0, 13.0]))

    def test_jit_and_grad(self):
        f = jax.jit(lambda x: la.norm2(x))
        x = jnp.array([3.0, 4.0])
        assert jnp.allclose(f(x), 5.0)
        # d|x|/dx = x / |x|
        g = jax.grad(lambda x: la.norm2(x))(x)
        assert jnp.allclose(g, x / 5.0)


class TestProjectOnto:
    """Tests for project_onto, including the degenerate b == 0 case."""

    def test_analytic_projection(self):
        # Project a onto b; b along x-axis -> keep only x-component.
        a = jnp.array([3.0, 4.0])
        b = jnp.array([2.0, 0.0])
        result = la.project_onto(a, b)
        assert jnp.allclose(result, jnp.array([3.0, 0.0]))

    def test_projection_parallel(self):
        # Projecting a vector onto a parallel vector returns the vector itself.
        a = jnp.array([1.0, 2.0, 2.0])
        b = jnp.array([2.0, 4.0, 4.0])  # b = 2a
        result = la.project_onto(a, b)
        assert jnp.allclose(result, a)

    def test_projection_orthogonal(self):
        # Projecting onto an orthogonal vector returns zero.
        a = jnp.array([1.0, 0.0])
        b = jnp.array([0.0, 5.0])
        result = la.project_onto(a, b)
        assert jnp.allclose(result, jnp.zeros(2))

    def test_default_eps_zero_b_is_finite(self):
        """Degenerate case b == 0 with default eps must be finite (not NaN)."""
        a = jnp.array([3.0, 4.0])
        b = jnp.zeros(2)
        result = la.project_onto(a, b)  # default eps
        assert jnp.all(jnp.isfinite(result))
        # Projection onto the zero subspace is the zero vector.
        assert jnp.allclose(result, jnp.zeros(2))

    def test_explicit_eps_zero_b_is_finite(self):
        """Even with eps=0.0 explicitly, b == 0 must not produce NaN."""
        a = jnp.array([1.0, 2.0, -3.0])
        b = jnp.zeros(3)
        result = la.project_onto(a, b, eps=0.0)
        assert jnp.all(jnp.isfinite(result))
        assert jnp.allclose(result, jnp.zeros(3))

    def test_normal_inputs_unchanged_by_fix(self):
        """Non-degenerate projection equals the exact analytic formula."""
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([4.0, -1.0, 2.0])
        scale = jnp.dot(a, b) / jnp.dot(b, b)
        expected = scale * b
        result = la.project_onto(a, b, eps=0.0)
        assert jnp.allclose(result, expected)

    def test_batched_axis(self):
        a = jnp.array([[3.0, 4.0], [1.0, 0.0]])
        b = jnp.array([[2.0, 0.0], [0.0, 5.0]])
        result = la.project_onto(a, b, axis=-1)
        expected = jnp.array([[3.0, 0.0], [0.0, 0.0]])
        assert jnp.allclose(result, expected)

    def test_jit_and_grad(self):
        f = jax.jit(lambda a, b: la.project_onto(a, b))
        a = jnp.array([3.0, 4.0])
        b = jnp.array([2.0, 0.0])
        assert jnp.allclose(f(a, b), jnp.array([3.0, 0.0]))
        g = jax.grad(lambda a: jnp.sum(la.project_onto(a, b)))(a)
        assert jnp.all(jnp.isfinite(g))


class TestConditionNumber:
    """Tests for condition_number (2-norm condition number)."""

    def test_diagonal_matrix(self):
        # Diagonal -> singular values are |diag|; cond = max/min.
        A = jnp.diag(jnp.array([4.0, 1.0, 2.0]))
        assert jnp.allclose(la.condition_number(A), 4.0)

    def test_identity_is_one(self):
        A = jnp.eye(3)
        assert jnp.allclose(la.condition_number(A), 1.0)

    def test_exact_zero_singular_value_sentinel(self):
        # Structurally rank-deficient: singular values are exactly {1, 0}.
        # The s_min == 0 guard replaces s_min by +inf in the denominator, so the
        # condition number returns 0.0 as a "singular / undefined" sentinel
        # (documented behavior; avoids NaN).
        A = jnp.array([[1.0, 0.0], [0.0, 0.0]])
        cond = la.condition_number(A)
        assert jnp.isfinite(cond)
        assert cond == 0.0

    def test_near_singular_matrix_is_huge(self):
        # Numerically rank-deficient: float SVD gives a tiny (not exactly 0)
        # smallest singular value, so the condition number is huge but finite.
        A = jnp.array([[1.0, 2.0], [2.0, 4.0]])
        cond = la.condition_number(A)
        assert jnp.isfinite(cond)
        assert cond > 1e12

    def test_known_matrix(self):
        # 2x2 with known singular values: A = diag(10, 0.5) rotated keeps sv.
        A = jnp.array([[10.0, 0.0], [0.0, 0.5]])
        assert jnp.allclose(la.condition_number(A), 20.0)

    def test_jit_compatible(self):
        A = jnp.diag(jnp.array([3.0, 1.0]))
        assert jnp.allclose(jax.jit(la.condition_number)(A), 3.0)
