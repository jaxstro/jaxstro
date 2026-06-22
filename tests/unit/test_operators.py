"""Tests for small matrix-free linear operators."""

import jax
import jax.numpy as jnp

from jaxstro.numerics import operators


class TestPrimitiveOperators:
    """Tests for dense and diagonal operators."""

    def test_dense_operator_matches_matrix_products(self):
        matrix = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        op = operators.DenseOperator(matrix)
        x = jnp.array([0.5, -1.0])
        y = jnp.array([2.0, -3.0])
        assert op.shape == matrix.shape
        assert jnp.allclose(op.matvec(x), matrix @ x)
        assert jnp.allclose(op.rmatvec(y), matrix.T @ y)
        assert jnp.allclose(op.to_dense(), matrix)

    def test_diagonal_operator_matches_dense_diagonal(self):
        diagonal = jnp.array([1.0, -2.0, 3.0])
        op = operators.DiagonalOperator(diagonal)
        x = jnp.array([4.0, 5.0, 6.0])
        dense = jnp.diag(diagonal)
        assert op.shape == (3, 3)
        assert jnp.allclose(op.matvec(x), dense @ x)
        assert jnp.allclose(op.rmatvec(x), dense.T @ x)
        assert jnp.allclose(op.to_dense(), dense)


class TestOperatorComposition:
    """Tests for scaled, sum, product, transpose, and block operators."""

    def test_scaled_sum_and_product_match_dense_algebra(self):
        a = jnp.array([[1.0, 2.0], [0.0, -1.0]])
        b = jnp.array([[3.0, 0.0], [2.0, 1.0]])
        op_a = operators.DenseOperator(a)
        op_b = operators.DenseOperator(b)
        x = jnp.array([0.25, -0.5])

        scaled = operators.scale(op_a, 2.5)
        summed = operators.add(op_a, op_b)
        product = operators.compose(op_a, op_b)

        assert jnp.allclose(scaled.matvec(x), (2.5 * a) @ x)
        assert jnp.allclose(summed.matvec(x), (a + b) @ x)
        assert jnp.allclose(product.matvec(x), (a @ b) @ x)
        assert jnp.allclose(product.rmatvec(x), (a @ b).T @ x)

    def test_transpose_operator_swaps_matvec_and_rmatvec(self):
        matrix = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        op = operators.DenseOperator(matrix)
        transposed = operators.transpose(op)
        x = jnp.array([1.0, -1.0])
        y = jnp.array([0.5, 1.5, -2.0])
        assert transposed.shape == (3, 2)
        assert jnp.allclose(transposed.matvec(x), matrix.T @ x)
        assert jnp.allclose(transposed.rmatvec(y), matrix @ y)
        assert jnp.allclose(transposed.to_dense(), matrix.T)

    def test_block_diagonal_operator_matches_dense_block_matrix(self):
        left = operators.DenseOperator(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
        right = operators.DiagonalOperator(jnp.array([5.0]))
        op = operators.block_diag(left, right)
        dense = jnp.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
        x = jnp.array([1.0, -1.0, 2.0])
        assert op.shape == dense.shape
        assert jnp.allclose(op.matvec(x), dense @ x)
        assert jnp.allclose(op.rmatvec(x), dense.T @ x)
        assert jnp.allclose(op.to_dense(), dense)


class TestOperatorJAXTransforms:
    """Tests for PyTree and transform behavior."""

    def test_operator_is_pytree_and_jit_compatible(self):
        op = operators.scale(
            operators.DenseOperator(jnp.array([[2.0, 0.0], [0.0, 3.0]])),
            0.5,
        )
        leaves = jax.tree_util.tree_leaves(op)
        assert leaves

        @jax.jit
        def apply(operator, x):
            return operator.matvec(x)

        x = jnp.array([4.0, 6.0])
        assert jnp.allclose(apply(op, x), jnp.array([4.0, 9.0]))

    def test_operator_matvec_supports_vmap_over_vectors(self):
        op = operators.DenseOperator(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
        xs = jnp.array([[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]])
        result = jax.vmap(op.matvec)(xs)
        expected = xs @ op.to_dense().T
        assert jnp.allclose(result, expected)
