"""Block tridiagonal solves against the dense solve of the same matrix.

The reference is ``jnp.linalg.solve`` on the assembled dense matrix: an independent
algorithm (LU with partial pivoting) on the same system. Matrices are random and block
diagonally dominant, the class the elimination is stable for, with nonzero off-diagonal
blocks so the coupling is exercised (a block-diagonal matrix would pass a solver that
ignored the bands).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstro.numerics.block_tridiagonal import (
    block_tridiagonal_solve,
    block_tridiagonal_solve_transposed,
)

jax.config.update("jax_enable_x64", True)


def _bands(n, b, seed):
    rng = np.random.default_rng(seed)
    lower, upper = rng.normal(size=(n, b, b)), rng.normal(size=(n, b, b))
    diag = rng.normal(size=(n, b, b)) + 4.0 * b * np.eye(b)
    lower[0] = 0.0
    upper[-1] = 0.0
    return jnp.asarray(lower), jnp.asarray(diag), jnp.asarray(upper)


def _dense(lower, diag, upper):
    n, b = diag.shape[0], diag.shape[1]
    m = jnp.zeros((n * b, n * b))
    for j in range(n):                       # test assembly of the reference, not traced
        m = m.at[j * b:(j + 1) * b, j * b:(j + 1) * b].set(diag[j])
        if j > 0:
            m = m.at[j * b:(j + 1) * b, (j - 1) * b:j * b].set(lower[j])
        if j < n - 1:
            m = m.at[j * b:(j + 1) * b, (j + 1) * b:(j + 2) * b].set(upper[j])
    return m


@pytest.mark.parametrize("b", [1, 2, 4])
@pytest.mark.parametrize("k", [None, 3])
def test_matches_the_dense_solve(b, k):
    n = 9
    lower, diag, upper = _bands(n, b, seed=b)
    rhs = jnp.asarray(np.random.default_rng(7).normal(size=(n, b) if k is None else (n, b, k)))
    x = block_tridiagonal_solve(lower, diag, upper, rhs)
    dense = _dense(lower, diag, upper)
    ref = jnp.linalg.solve(dense, rhs.reshape(n * b, -1)).reshape(x.shape)
    off = float(jnp.max(jnp.abs(lower))) + float(jnp.max(jnp.abs(upper)))
    assert off > 1.0                                      # excited: the bands couple cells
    assert jnp.allclose(x, ref, rtol=1e-12, atol=1e-13), float(jnp.max(jnp.abs(x - ref)))
    xt = block_tridiagonal_solve_transposed(lower, diag, upper, rhs)
    ref_t = jnp.linalg.solve(dense.T, rhs.reshape(n * b, -1)).reshape(x.shape)
    assert jnp.allclose(xt, ref_t, rtol=1e-12, atol=1e-13)


def test_the_implicit_adjoint_matches_differentiating_the_dense_solve():
    """Every band and the right-hand side, through a scalar loss of the solution."""
    n, b = 7, 2
    lower, diag, upper = _bands(n, b, seed=3)
    rhs = jnp.asarray(np.random.default_rng(4).normal(size=(n, b)))
    weights = jnp.asarray(np.random.default_rng(5).normal(size=(n, b)))

    def loss(solve, *args):
        return jnp.sum(weights * solve(*args) ** 2)

    def dense_solve(lower, diag, upper, rhs):
        return jnp.linalg.solve(_dense(lower, diag, upper), rhs.reshape(-1)).reshape(n, b)

    ours = jax.grad(lambda *a: loss(block_tridiagonal_solve, *a), argnums=(0, 1, 2, 3))(
        lower, diag, upper, rhs)
    ref = jax.grad(lambda *a: loss(dense_solve, *a), argnums=(0, 1, 2, 3))(
        lower, diag, upper, rhs)
    # The bands outside the matrix carry no derivative in either form.
    ref = (ref[0].at[0].set(0.0), ref[1], ref[2].at[-1].set(0.0), ref[3])
    for g, r in zip(ours, ref):
        assert float(jnp.max(jnp.abs(r))) > 1e-3        # excited
        assert jnp.allclose(g, r, rtol=1e-10, atol=1e-12), float(jnp.max(jnp.abs(g - r)))


def test_jit_and_vmap():
    n, b = 6, 2
    lower, diag, upper = _bands(n, b, seed=11)
    rhs = jnp.asarray(np.random.default_rng(12).normal(size=(4, n, b)))
    batched = jax.jit(jax.vmap(block_tridiagonal_solve, in_axes=(None, None, None, 0)))(
        lower, diag, upper, rhs)
    single = jnp.stack([block_tridiagonal_solve(lower, diag, upper, r) for r in rhs])
    assert jnp.allclose(batched, single, rtol=1e-13, atol=1e-14)
