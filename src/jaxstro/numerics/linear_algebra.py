# src/jaxstro/numerics/linear_algebra.py
"""
Small linear algebra helpers for JAX.

These are convenience wrappers around jax.numpy.linalg and related
operations, not a replacement for a full LA library. For serious
linear solves, eigenproblems, etc., use lineax or JAX's linalg.
"""

from functools import partial

import jax
import jax.numpy as jnp

from .types import Array


@partial(jax.jit, static_argnames=("axis", "keepdims"))
def norm2(
    x: Array,
    axis: int | None = None,
    keepdims: bool = False,
) -> Array:
    """
    Euclidean (ℓ2) norm of an array.

    ``axis`` and ``keepdims`` are static (they shape the output), so this is
    JIT-safe when called as ``norm2(x, axis=1)``.
    """
    return jnp.linalg.norm(x, ord=2, axis=axis, keepdims=keepdims)


@partial(jax.jit, static_argnames=("axis",))
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

    Degenerate case (b·b + eps == 0, i.e. projecting onto the zero vector with
    no regularization): the projection onto the zero subspace is the zero vector.
    The denominator is guarded with ``jnp.where`` so the result is 0 (finite),
    not NaN from a 0/0 division. Because ``b == 0`` in that case, ``scale * b``
    is exactly 0 regardless of the guarded ``scale`` value, so non-degenerate
    projections (den != 0) are bit-for-bit unchanged by the guard.
    """
    num = jnp.sum(a * b, axis=axis, keepdims=True)
    den = jnp.sum(b * b, axis=axis, keepdims=True) + eps
    # Guard the 0/0 division: replace a zero denominator by 1 so ``scale`` is
    # finite. Where den == 0 we necessarily have b == 0 (and eps == 0), so the
    # final ``scale * b`` is 0 — the correct projection onto the zero subspace.
    den_safe = jnp.where(den == 0.0, 1.0, den)
    scale = num / den_safe
    return scale * b


@jax.jit
def condition_number(A: Array) -> Array:
    """
    2-norm condition number of a matrix (or batch of matrices).

    Defined as ``sigma_max / sigma_min`` from the singular value decomposition.
    A rank-deficient matrix has ``sigma_min == 0`` (a mathematically infinite
    condition number). To keep the output free of NaN, an exactly-zero smallest
    singular value is replaced by ``+inf`` in the denominator, so the returned
    value is ``0.0`` as a sentinel for "singular / undefined condition number".
    Note this is only triggered by an *exact* float zero; a merely
    near-singular matrix returns a finite, very large value (see tests).

    Not differentiable at coincident singular values: the SVD's singular values
    have non-smooth (and, at exact degeneracy, undefined) derivatives where two
    singular values coincide, so ``jax.grad(condition_number)`` is unreliable
    there. Use this as a diagnostic, not inside a differentiated objective.
    """
    s = jnp.linalg.svd(A, compute_uv=False)
    s_max = jnp.max(s, axis=-1)
    s_min = jnp.min(s, axis=-1)
    s_min_safe = jnp.where(s_min == 0.0, jnp.inf, s_min)
    return s_max / s_min_safe


__all__ = ["norm2", "project_onto", "condition_number"]
