# tests/test_linear_algebra.py
"""
Tests for jaxstro.numerics.linear_algebra.

Covers norm2, project_onto, and condition_number, including the
degenerate b·b == 0 case for project_onto (must be finite, not NaN).
"""

import jax
import jax.numpy as jnp
import pytest

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

    def test_exact_zero_singular_value_is_inf(self):
        # Structurally rank-deficient: singular values are exactly {1, 0}.
        # A singular matrix has a mathematically infinite condition number, so the
        # s_min == 0 case returns +inf (matching numpy.linalg.cond) rather than a
        # surprising 0.0 — a caller guarding `cond > threshold` then correctly
        # rejects it. The result is never NaN (guarded for the zero matrix too).
        A = jnp.array([[1.0, 0.0], [0.0, 0.0]])
        cond = la.condition_number(A)
        assert jnp.isinf(cond)
        assert cond > 0.0

    def test_zero_matrix_condition_is_inf_not_nan(self):
        # Zero matrix: s_max == s_min == 0; must not produce 0/0 = NaN.
        cond = la.condition_number(jnp.zeros((3, 3)))
        assert jnp.isinf(cond)
        assert not jnp.isnan(cond)

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


class TestWeightedLeastSquares:
    """Tests for weighted least-squares coefficient fitting."""

    def test_unweighted_matches_known_line(self):
        x = jnp.array([0.0, 1.0, 2.0, 3.0])
        design = jnp.stack([jnp.ones_like(x), x], axis=1)
        y = 2.0 + 3.0 * x
        coeffs = la.weighted_lstsq(design, y)
        assert jnp.allclose(coeffs, jnp.array([2.0, 3.0]), atol=1e-12)

    def test_weights_downweight_outlier(self):
        x = jnp.array([0.0, 1.0, 2.0, 3.0])
        design = jnp.stack([jnp.ones_like(x), x], axis=1)
        y = jnp.array([1.0, 3.0, 5.0, 100.0])
        weights = jnp.array([1.0, 1.0, 1.0, 0.0])
        coeffs = la.weighted_lstsq(design, y, weights=weights)
        assert jnp.allclose(coeffs, jnp.array([1.0, 2.0]), atol=1e-12)

    def test_vector_valued_response(self):
        x = jnp.array([0.0, 1.0, 2.0, 3.0])
        design = jnp.stack([jnp.ones_like(x), x], axis=1)
        y = jnp.stack([1.0 + x, 2.0 - 0.5 * x], axis=1)
        coeffs = la.weighted_lstsq(design, y)
        expected = jnp.array([[1.0, 2.0], [1.0, -0.5]])
        assert jnp.allclose(coeffs, expected, atol=1e-12)

    def test_jit_and_grad_compatible(self):
        design = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])

        @jax.jit
        def loss(y):
            coeffs = la.weighted_lstsq(design, y)
            return jnp.sum(coeffs)

        y = jnp.array([1.0, 3.0, 5.0])
        assert jnp.isfinite(loss(y))
        assert jnp.all(jnp.isfinite(jax.grad(loss)(y)))

    def test_rejects_invalid_shapes_eagerly(self):
        with pytest.raises(ValueError, match="2D"):
            la.weighted_lstsq(jnp.ones(3), jnp.ones(3))
        with pytest.raises(ValueError, match="same number of samples"):
            la.weighted_lstsq(jnp.ones((3, 2)), jnp.ones(4))
        with pytest.raises(ValueError, match="weights"):
            la.weighted_lstsq(jnp.ones((3, 2)), jnp.ones(3), weights=jnp.ones(4))


class TestSolveWrappers:
    """Tests for QR and SVD linear solve wrappers."""

    def test_qr_solve_matches_square_solution(self):
        A = jnp.array([[3.0, 1.0], [1.0, 2.0]])
        b = jnp.array([9.0, 8.0])
        result = la.qr_solve(A, b)
        assert jnp.allclose(result, jnp.linalg.solve(A, b), atol=1e-12)

    def test_qr_solve_handles_tall_least_squares(self):
        A = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        b = jnp.array([1.0, 3.0, 5.0, 7.0])
        result = la.qr_solve(A, b)
        assert jnp.allclose(result, jnp.array([1.0, 2.0]), atol=1e-12)

    def test_svd_solve_drops_truncated_direction(self):
        A = jnp.diag(jnp.array([2.0, 1e-12]))
        b = jnp.array([4.0, 1.0])
        result = la.svd_solve(A, b, rcond=1e-8)
        assert jnp.allclose(result, jnp.array([2.0, 0.0]))

    def test_svd_solve_matches_pseudoinverse_for_rank_deficient_matrix(self):
        A = jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        b = jnp.array([2.0, 4.0, 6.0])
        result = la.svd_solve(A, b)
        assert jnp.allclose(A @ result, b, atol=1e-10)
        assert jnp.allclose(result[0], result[1], atol=1e-10)


class TestCovarianceCorrelation:
    """Tests for covariance and correlation helpers."""

    def test_covariance_matrix_matches_centered_formula(self):
        samples = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 8.0]])
        result = la.covariance_matrix(samples, ddof=1)
        centered = samples - jnp.mean(samples, axis=0)
        expected = centered.T @ centered / 2.0
        assert jnp.allclose(result, expected)

    def test_weighted_covariance_uses_supplied_weights(self):
        samples = jnp.array([[0.0, 0.0], [2.0, 0.0], [4.0, 2.0]])
        weights = jnp.array([1.0, 2.0, 1.0])
        result = la.covariance_matrix(samples, weights=weights, ddof=0)
        mean = jnp.sum(samples * weights[:, None], axis=0) / jnp.sum(weights)
        centered = samples - mean
        expected = (centered * weights[:, None]).T @ centered / jnp.sum(weights)
        assert jnp.allclose(result, expected)

    def test_correlation_matrix_has_unit_diagonal(self):
        samples = jnp.array([[1.0, 1.0], [2.0, 3.0], [4.0, 5.0], [8.0, 7.0]])
        corr = la.correlation_matrix(samples)
        assert jnp.allclose(jnp.diag(corr), jnp.ones(2))
        assert jnp.all(jnp.abs(corr) <= 1.0 + 1e-12)

    def test_correlation_from_covariance_handles_zero_variance(self):
        cov = jnp.array([[4.0, 0.0], [0.0, 0.0]])
        corr = la.correlation_from_covariance(cov)
        assert jnp.all(jnp.isfinite(corr))
        assert jnp.allclose(corr, jnp.array([[1.0, 0.0], [0.0, 0.0]]))


class TestPositiveDefiniteJitter:
    """Tests for positive-definite checks and jitter utilities."""

    def test_is_positive_definite(self):
        assert bool(la.is_positive_definite(jnp.array([[2.0, 0.2], [0.2, 1.0]])))
        assert not bool(la.is_positive_definite(jnp.array([[1.0, 0.0], [0.0, -1.0]])))

    def test_add_diagonal_jitter_only_touches_diagonal(self):
        A = jnp.array([[1.0, 0.2], [0.2, 3.0]])
        result = la.add_diagonal_jitter(A, 0.5)
        assert jnp.allclose(result, jnp.array([[1.5, 0.2], [0.2, 3.5]]))

    def test_positive_definite_jitter_finds_small_enough_diagonal_shift(self):
        A = jnp.array([[0.0, 0.0], [0.0, 2.0]])
        shifted, jitter, success = la.positive_definite_jitter(
            A, initial_jitter=1e-6, growth=10.0, max_steps=4
        )
        assert bool(success)
        assert jitter == pytest.approx(1e-6)
        assert bool(la.is_positive_definite(shifted))

    def test_positive_definite_jitter_preserves_already_pd_matrix(self):
        A = jnp.array([[2.0, 0.1], [0.1, 1.0]])
        shifted, jitter, success = la.positive_definite_jitter(A)
        assert bool(success)
        assert jitter == pytest.approx(0.0)
        assert jnp.allclose(shifted, A)
