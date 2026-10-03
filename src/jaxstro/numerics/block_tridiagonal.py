# src/jaxstro/numerics/block_tridiagonal.py

"""
Block tridiagonal linear solves (block Thomas elimination) for JAX.

Systems of the form

    L_j x_{j-1} + D_j x_j + U_j x_{j+1} = b_j,    j = 0 .. n-1,

with ``b x b`` blocks: the Jacobian of any 1D problem whose cells couple only to their
neighbours (a Henyey stellar-structure step, an implicit Lagrangian hydro step, a radiation
diffusion operator with several fields per cell). O(n b^3) work in two ``lax.scan`` sweeps.

The elimination solves each ``b x b`` block system with ``jnp.linalg.solve`` and never
forms an inverse. It does not pivot ACROSS blocks, so it is stable for block diagonally
dominant matrices (the usual case for implicit discretisations) and for the stellar
structure Jacobians stellax measured; for other matrices the caller owns the check.

Reverse mode is the implicit adjoint, not differentiation through the sweeps: with
``x = M^{-1} b`` and cotangent ``x_bar``, solve ``M^T lam = x_bar``; then ``b_bar = lam`` and
the band cotangents are ``L_bar_j = -lam_j x_{j-1}^T``, ``D_bar_j = -lam_j x_j^T``,
``U_bar_j = -lam_j x_{j+1}^T``. One transposed block solve, the cost of the forward one.

Ported from stellax's Henyey solver (``stellax/solver/block_thomas.py``, Henyey et al. 1964;
Press et al. 2007, Numerical Recipes 2.4) and generalised to any block size and to several
right-hand sides, for hydrax's implicit relaxation step (2026-10-03, Anna).

Conventions: ``lower``, ``diag`` and ``upper`` are each ``(n, b, b)``; ``lower[0]`` and
``upper[n-1]`` lie outside the matrix and are ignored. ``rhs`` is ``(n, b)`` or
``(n, b, k)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = ["block_tridiagonal_solve", "block_tridiagonal_solve_transposed"]


def _sweep(lower, diag, upper, rhs):
    """Forward elimination and back substitution; ``rhs`` is ``(n, b, k)``."""
    b, k = diag.shape[-1], rhs.shape[-1]

    def forward(carry, row):
        c_prev, d_prev = carry
        l_j, d_j, u_j, f_j = row
        schur = d_j - l_j @ c_prev
        packed = jnp.linalg.solve(schur, jnp.concatenate([u_j, f_j - l_j @ d_prev], axis=1))
        c_j, d_j_out = packed[:, :b], packed[:, b:]
        return (c_j, d_j_out), (c_j, d_j_out)

    start = (jnp.zeros((b, b), diag.dtype), jnp.zeros((b, k), diag.dtype))
    lower = lower.at[0].set(0.0)
    upper = upper.at[-1].set(0.0)
    _, (c, d) = jax.lax.scan(forward, start, (lower, diag, upper, rhs))

    def backward(x_next, row):
        c_j, d_j = row
        x_j = d_j - c_j @ x_next
        return x_j, x_j

    _, x = jax.lax.scan(backward, jnp.zeros((b, k), diag.dtype), (c, d), reverse=True)
    return x


def _transpose_bands(lower, diag, upper):
    """The bands of ``M^T``: row ``j`` of ``M^T`` holds ``U_{j-1}^T``, ``D_j^T``, ``L_{j+1}^T``."""
    t = lambda a: jnp.swapaxes(a, -1, -2)  # noqa: E731
    zero = jnp.zeros_like(diag[:1])
    lower_t = jnp.concatenate([zero, t(upper[:-1])], axis=0)
    upper_t = jnp.concatenate([t(lower[1:]), zero], axis=0)
    return lower_t, t(diag), upper_t


@jax.custom_vjp
def _solve(lower, diag, upper, rhs):
    return _sweep(lower, diag, upper, rhs)


def _solve_fwd(lower, diag, upper, rhs):
    x = _sweep(lower, diag, upper, rhs)
    return x, (lower, diag, upper, x)


def _solve_bwd(residuals, x_bar):
    lower, diag, upper, x = residuals
    lam = _sweep(*_transpose_bands(lower, diag, upper), x_bar)
    outer = lambda a, c: -jnp.einsum("nik,njk->nij", a, c)  # noqa: E731
    zero = jnp.zeros_like(x[:1])
    x_prev = jnp.concatenate([zero, x[:-1]], axis=0)
    x_next = jnp.concatenate([x[1:], zero], axis=0)
    lower_bar = outer(lam, x_prev).at[0].set(0.0)
    upper_bar = outer(lam, x_next).at[-1].set(0.0)
    return lower_bar, outer(lam, x), upper_bar, lam


_solve.defvjp(_solve_fwd, _solve_bwd)


def block_tridiagonal_solve(
    lower: Float[Array, "n b b"],
    diag: Float[Array, "n b b"],
    upper: Float[Array, "n b b"],
    rhs: Float[Array, "n b ..."],
) -> Float[Array, "n b ..."]:
    """Solve ``L_j x_{j-1} + D_j x_j + U_j x_{j+1} = rhs_j`` (module docstring).

    Differentiable in all four arguments by the implicit adjoint (one transposed solve).
    """
    vector = rhs.ndim == 2
    x = _solve(lower, diag, upper, rhs[..., None] if vector else rhs)
    return x[..., 0] if vector else x


def block_tridiagonal_solve_transposed(
    lower: Float[Array, "n b b"],
    diag: Float[Array, "n b b"],
    upper: Float[Array, "n b b"],
    rhs: Float[Array, "n b ..."],
) -> Float[Array, "n b ..."]:
    """Solve ``M^T x = rhs`` for the ``M`` the same three bands describe."""
    return block_tridiagonal_solve(*_transpose_bands(lower, diag, upper), rhs)
