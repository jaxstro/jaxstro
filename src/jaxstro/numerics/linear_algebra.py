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

from .checks import try_concrete_bool


def _raise_if_concrete_false(predicate, message: str) -> None:
    """Raise eagerly when a validation predicate is concrete and false."""
    result = try_concrete_bool(jnp.asarray(predicate))
    if result is False:
        raise ValueError(message)


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


def weighted_lstsq(
    design: Float[Array, " n p"],
    y: Float[Array, " n ..."],
    weights: Float[Array, " n"] | None = None,
    rcond: float | None = None,
) -> Float[Array, " p ..."]:
    """
    Solve an ordinary or weighted least-squares problem.

    ``design`` is a two-dimensional sample-by-feature matrix. ``y`` may be a
    one-dimensional response or a matrix of vector-valued responses. If
    supplied, ``weights`` are nonnegative per-sample weights applied as
    ``sqrt(weights)`` to both the design matrix and response.
    """
    design = jnp.asarray(design)
    y = jnp.asarray(y)
    if design.ndim != 2:
        raise ValueError("design must be a 2D array")
    if y.ndim < 1:
        raise ValueError("y must have at least one dimension")
    if design.shape[0] != y.shape[0]:
        raise ValueError("design and y must have the same number of samples")

    if weights is None:
        lhs = design
        rhs = y
    else:
        weights = jnp.asarray(weights)
        if weights.ndim != 1 or weights.shape[0] != design.shape[0]:
            raise ValueError("weights must be a 1D array matching the samples")
        _raise_if_concrete_false(jnp.all(weights >= 0.0), "weights must be nonnegative")
        sqrt_w = jnp.sqrt(weights)
        lhs = design * sqrt_w[:, None]
        if y.ndim == 1:
            rhs = y * sqrt_w
        else:
            rhs = y * sqrt_w.reshape((sqrt_w.shape[0],) + (1,) * (y.ndim - 1))

    coeffs, _, _, _ = jnp.linalg.lstsq(lhs, rhs, rcond=rcond)
    return coeffs


def qr_solve(
    A: Float[Array, " m n"],
    b: Float[Array, " m ..."],
) -> Float[Array, " n ..."]:
    """
    Solve a square or overdetermined full-rank system using reduced QR.

    For tall matrices this returns the least-squares solution of ``A x ~= b``.
    """
    A = jnp.asarray(A)
    b = jnp.asarray(b)
    if A.ndim != 2:
        raise ValueError("A must be a 2D array")
    if b.ndim < 1 or b.shape[0] != A.shape[0]:
        raise ValueError("b must have the same leading dimension as A")
    if A.shape[0] < A.shape[1]:
        raise ValueError("qr_solve requires rows >= columns")

    q, r = jnp.linalg.qr(A, mode="reduced")
    rhs = q.T @ b
    return jnp.linalg.solve(r, rhs)


def svd_solve(
    A: Float[Array, " m n"],
    b: Float[Array, " m ..."],
    rcond: float | None = None,
) -> Float[Array, " n ..."]:
    """
    Solve ``A x ~= b`` through a truncated SVD pseudoinverse.

    Singular values at or below ``rcond * max(s)`` are discarded, making the
    truncation policy explicit for ill-conditioned problems.
    """
    A = jnp.asarray(A)
    b = jnp.asarray(b)
    if A.ndim != 2:
        raise ValueError("A must be a 2D array")
    if b.ndim < 1 or b.shape[0] != A.shape[0]:
        raise ValueError("b must have the same leading dimension as A")

    u, s, vh = jnp.linalg.svd(A, full_matrices=False)
    if rcond is None:
        rcond = float(jnp.finfo(A.dtype).eps * max(A.shape))
    cutoff = rcond * jnp.max(s)
    s_inv = jnp.where(s > cutoff, 1.0 / s, 0.0)
    rhs = u.T @ b
    if rhs.ndim == 1:
        scaled = s_inv * rhs
    else:
        scaled = s_inv.reshape((s_inv.shape[0],) + (1,) * (rhs.ndim - 1)) * rhs
    return vh.T @ scaled


@partial(jax.jit, static_argnames=("rowvar", "ddof"))
def covariance_matrix(
    samples: Float[Array, " n p"],
    weights: Float[Array, " n"] | None = None,
    *,
    rowvar: bool = False,
    ddof: int = 1,
) -> Float[Array, " p p"]:
    """
    Covariance matrix for sample rows by default.

    With ``rowvar=False`` (default), rows are observations and columns are
    variables. With ``rowvar=True``, rows are variables and columns are
    observations. Optional weights are interpreted as per-observation weights.
    """
    samples = jnp.asarray(samples)
    if samples.ndim != 2:
        raise ValueError("samples must be a 2D array")
    data = samples if rowvar else samples.T
    n_obs = data.shape[1]

    if weights is None:
        mean = jnp.mean(data, axis=1, keepdims=True)
        centered = data - mean
        denom = jnp.asarray(n_obs - ddof, dtype=data.dtype)
        denom_safe = jnp.where(denom == 0.0, 1.0, denom)
        return centered @ centered.T / denom_safe

    weights = jnp.asarray(weights)
    if weights.ndim != 1 or weights.shape[0] != n_obs:
        raise ValueError("weights must be a 1D array matching observations")
    _raise_if_concrete_false(jnp.all(weights >= 0.0), "weights must be nonnegative")
    w_sum = jnp.sum(weights)
    w_sum_safe = jnp.where(w_sum == 0.0, 1.0, w_sum)
    mean = jnp.sum(data * weights[None, :], axis=1, keepdims=True) / w_sum_safe
    centered = data - mean
    denom = w_sum - ddof
    denom_safe = jnp.where(denom == 0.0, 1.0, denom)
    return (centered * weights[None, :]) @ centered.T / denom_safe


@jax.jit
def correlation_from_covariance(
    covariance: Float[Array, " n n"],
) -> Float[Array, " n n"]:
    """
    Convert a covariance matrix to a correlation matrix with zero-variance guards.
    """
    covariance = jnp.asarray(covariance)
    variance = jnp.diag(covariance)
    scale = jnp.sqrt(jnp.outer(variance, variance))
    scale_safe = jnp.where(scale == 0.0, 1.0, scale)
    corr = covariance / scale_safe
    return jnp.where(scale == 0.0, 0.0, corr)


@partial(jax.jit, static_argnames=("rowvar", "ddof"))
def correlation_matrix(
    samples: Float[Array, " n p"],
    weights: Float[Array, " n"] | None = None,
    *,
    rowvar: bool = False,
    ddof: int = 1,
) -> Float[Array, " p p"]:
    """Correlation matrix for sample rows by default."""
    return correlation_from_covariance(
        covariance_matrix(samples, weights=weights, rowvar=rowvar, ddof=ddof)
    )


@jax.jit
def is_positive_definite(
    A: Float[Array, " n n"],
    *,
    tol: float = 0.0,
) -> Array:
    """Return whether the symmetrized matrix is positive definite."""
    A = jnp.asarray(A)
    sym = 0.5 * (A + jnp.swapaxes(A, -1, -2))
    return jnp.all(jnp.linalg.eigvalsh(sym) > tol, axis=-1)


@jax.jit
def add_diagonal_jitter(
    A: Float[Array, "... n n"],
    jitter: float | Float[Array, ""],
) -> Float[Array, "... n n"]:
    """Add ``jitter`` to the diagonal of a square matrix or matrix batch."""
    A = jnp.asarray(A)
    eye = jnp.eye(A.shape[-1], dtype=A.dtype)
    return A + jnp.asarray(jitter, dtype=A.dtype) * eye


def positive_definite_jitter(
    A: Float[Array, " n n"],
    *,
    initial_jitter: float = 1e-12,
    growth: float = 10.0,
    max_steps: int = 8,
    tol: float = 0.0,
) -> tuple[Float[Array, " n n"], Float[Array, ""], Array]:
    """
    Find a diagonal jitter that makes a symmetric matrix positive definite.

    Returns ``(A + jitter I, jitter, success)``. Already-positive-definite
    matrices return zero jitter.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if initial_jitter <= 0.0:
        raise ValueError("initial_jitter must be positive")
    if growth <= 1.0:
        raise ValueError("growth must be greater than 1")

    A = jnp.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D matrix")

    dtype = A.dtype
    zero = jnp.asarray(0.0, dtype=dtype)
    steps = jnp.arange(max_steps, dtype=dtype)
    success0 = is_positive_definite(A, tol=tol)
    init = (zero, success0)

    def scan_step(carry, k):
        jitter, success = carry
        candidate = jnp.asarray(initial_jitter, dtype=dtype) * jnp.power(
            jnp.asarray(growth, dtype=dtype), k
        )
        candidate_success = is_positive_definite(
            add_diagonal_jitter(A, candidate), tol=tol
        )
        take = (~success) & candidate_success
        jitter_new = jnp.where(take, candidate, jitter)
        success_new = success | candidate_success
        return (jitter_new, success_new), None

    (jitter, success), _ = jax.lax.scan(scan_step, init, steps)
    return add_diagonal_jitter(A, jitter), jitter, success


__all__ = [
    "norm2",
    "project_onto",
    "condition_number",
    "weighted_lstsq",
    "qr_solve",
    "svd_solve",
    "covariance_matrix",
    "correlation_from_covariance",
    "correlation_matrix",
    "is_positive_definite",
    "add_diagonal_jitter",
    "positive_definite_jitter",
]
