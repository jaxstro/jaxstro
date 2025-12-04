# src/jaxstro/numerics/linear_algebra.py
"""
Small linear algebra helpers for JAX.

These are convenience wrappers around jax.numpy.linalg and related
operations, not a replacement for a full LA library. For serious
linear solves, eigenproblems, etc., use lineax or JAX's linalg.
"""

import jax
import jax.numpy as jnp

from .types import Array


@jax.jit
def norm2(
    x: Array,
    axis: int | None = None,
    keepdims: bool = False,
) -> Array:
    """
    Euclidean (ℓ2) norm of an array.
    """
    return jnp.linalg.norm(x, ord=2, axis=axis, keepdims=keepdims)


@jax.jit
def project_onto(
    a: Array,
    b: Array,
    *,
    axis: int = -1,
    eps: float = 0.0,
) -> Array:
    """
    Project vector a onto vector b along a given axis.

    Computes: proj_b(a) = (a·b / (b·b + eps)) * b
    """
    num = jnp.sum(a * b, axis=axis, keepdims=True)
    den = jnp.sum(b * b, axis=axis, keepdims=True) + eps
    scale = num / den
    return scale * b


@jax.jit
def condition_number(A: Array) -> Array:
    """
    2-norm condition number of a matrix (or batch of matrices).
    """
    s = jnp.linalg.svd(A, compute_uv=False)
    s_max = jnp.max(s, axis=-1)
    s_min = jnp.min(s, axis=-1)
    s_min_safe = jnp.where(s_min == 0.0, jnp.inf, s_min)
    return s_max / s_min_safe


__all__ = ["norm2", "project_onto", "condition_number"]
