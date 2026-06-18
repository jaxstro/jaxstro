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
from jaxtyping import Array, Float


@partial(jax.jit, static_argnames=("axis", "keepdims"))
def norm2(
    x: Float[Array, "..."],
    axis: int | None = None,
    keepdims: bool = False,
) -> Float[Array, "..."]:
    """
    Euclidean (ℓ2) norm of an array.

    ``axis`` and ``keepdims`` are static (they shape the output), so this is
    JIT-safe when called as ``norm2(x, axis=1)``.
    """
    return jnp.linalg.norm(x, ord=2, axis=axis, keepdims=keepdims)


@partial(jax.jit, static_argnames=("axis",))
def project_onto(
    a: Float[Array, "..."],
    b: Float[Array, "..."],
    *,
    axis: int = -1,
    eps: float = 0.0,
) -> Float[Array, "..."]:
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
def condition_number(A: Float[Array, "... n n"]) -> Float[Array, "..."]:
    """
    2-norm condition number of a matrix (or batch of matrices).

    Defined as ``sigma_max / sigma_min`` from the singular value decomposition.
    A rank-deficient matrix has ``sigma_min == 0`` (a mathematically infinite
    condition number), so an exactly-zero smallest singular value returns
    ``+inf`` (matching ``numpy.linalg.cond``) — a caller guarding
    ``cond > threshold`` then correctly rejects a singular matrix. The result is
    never NaN: the zero matrix (``sigma_max == sigma_min == 0``) also returns
    ``+inf`` rather than ``0/0``. Note ``+inf`` is only triggered by an *exact*
    float zero; a merely near-singular matrix returns a finite, very large
    value (see tests).

    Not differentiable at coincident singular values: the SVD's singular values
    have non-smooth (and, at exact degeneracy, undefined) derivatives where two
    singular values coincide, so ``jax.grad(condition_number)`` is unreliable
    there. Use this as a diagnostic, not inside a differentiated objective.
    """
    s = jnp.linalg.svd(A, compute_uv=False)
    s_max = jnp.max(s, axis=-1)
    s_min = jnp.min(s, axis=-1)
    # Double-where: avoid 0/0 -> NaN at s_min == 0 (incl. the zero matrix),
    # then map the singular case to +inf (infinite condition number).
    singular = s_min == 0.0
    s_min_safe = jnp.where(singular, 1.0, s_min)
    return jnp.where(singular, jnp.inf, s_max / s_min_safe)


__all__ = ["norm2", "project_onto", "condition_number"]
