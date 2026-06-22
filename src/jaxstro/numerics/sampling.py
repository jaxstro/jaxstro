# src/jaxstro/numerics/sampling.py
"""
Differentiable sampling primitives for JAX.

These helpers turn an (unnormalized) weight tabulated on a grid into draws via
the inverse-CDF (a.k.a. percent-point function / PPF) method: build the
cumulative distribution, normalize it, and invert it at a uniform deviate. The
kernels are fully differentiable (no data-dependent shapes, no ``argmax``/
``argsort``) so gradients flow through both the tabulated weights and the
uniform deviate.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .integration import cumulative_trapz

__all__ = ["inverse_cdf_draw", "stratified_uniform"]


def inverse_cdf_draw(
    weight: Float[Array, " n"],
    grid: Float[Array, " n"],
    unif: Float[Array, ""],
    reg: float = 1e-30,
) -> Float[Array, ""]:
    r"""Differentiable inverse-CDF draw from an unnormalized weight on a uniform grid.

    Algorithm (inverse-CDF / PPF sampling)
    --------------------------------------
    1. Build the cumulative distribution of ``weight`` over ``grid`` with the
       trapezoid rule, assuming **uniform** spacing inferred from the first cell
       (``dx = grid[1] - grid[0]``). This reuses
       :func:`jaxstro.numerics.integration.cumulative_trapz` (dx-outside ordering)
       so the CDF is the ecosystem's single source of truth.
    2. Normalize by the total, ``cdf / (cdf[-1] + reg)``. The ``+reg`` guard
       (default ``reg = 1e-30``) keeps the normalization finite at **zero total
       weight**: instead of ``0/0 -> NaN`` it yields an all-zero CDF, and the
       interpolation below clamps the draw to ``grid[-1]`` (finite). Callers MUST
       therefore keep their own bound guard (e.g. ``where(W > 1e-6, u, 0.0)``)
       when the total weight may vanish.
    3. Invert the CDF at ``unif`` with :func:`jnp.interp` (the quantile / PPF).

    Differentiability
    -----------------
    Fully differentiable w.r.t. both ``weight`` and ``unif``; no data-dependent
    shapes. The ``+reg`` guard is negligible (``1e-30``) at any normal nonzero
    total weight, so it does not perturb gradients there. Returns a scalar draw;
    use :func:`jax.vmap` to draw many independent samples.

    Parameters
    ----------
    weight
        Unnormalized, nonnegative weights tabulated on ``grid`` (shape ``(n,)``).
    grid
        Uniformly spaced abscissae (shape ``(n,)``); spacing is taken from the
        first cell.
    unif
        Scalar uniform deviate in ``[0, 1]`` to invert.
    reg
        Small additive guard on the normalization denominator (default
        ``1e-30``) keeping it finite at zero total weight.

    Returns
    -------
    Float[Array, ""]
        Scalar draw in ``[grid[0], grid[-1]]``.
    """
    dx = grid[1] - grid[0]
    cdf = cumulative_trapz(weight, dx=dx)
    cdf = cdf / (cdf[-1] + reg)
    return jnp.interp(unif, cdf, grid)


@partial(jax.jit, static_argnames=("n",))
def stratified_uniform(
    key: Array,
    n: int,
    *,
    minval: float = 0.0,
    maxval: float = 1.0,
) -> Float[Array, " n"]:
    """
    Draw one uniform sample from each of ``n`` equal-width strata.

    The output has deterministic shape ``(n,)`` and is ordered by stratum. Use
    ``jax.random.permutation`` at the call site if randomized order is needed.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    u = jax.random.uniform(key, shape=(n,), minval=0.0, maxval=1.0)
    unit = (jnp.arange(n, dtype=u.dtype) + u) / n
    return minval + (maxval - minval) * unit
