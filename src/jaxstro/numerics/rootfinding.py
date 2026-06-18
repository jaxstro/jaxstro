# src/jaxstro/numerics/rootfinding.py
"""
Simple 1D root-finding helpers for JAX.

These are minimal, JAX-native solvers intended for scalar monotone
functions (bisection) or well-behaved functions with good initial
guesses (Newton). For large systems or more sophisticated solves,
prefer optimistix / lineax.

All functions use lax.scan internally with fixed iteration counts,
making them fully compatible with jit, vmap, and grad.

Usage:
    @jax.jit
    def solve(a, b):
        return bisect(lambda x: x**2 - 2.0, a, b)

    # Works with vmap
    jax.vmap(solve)(a_arr, b_arr)

    # Works with grad (differentiating through the solver)
    jax.grad(lambda a: solve(a, 2.0))(1.0)
"""

from typing import Optional

import jax
import jax.lax as lax
import jax.numpy as jnp

from .types import Array, ScalarFn


def bisect(
    f: ScalarFn,
    a: Array,
    b: Array,
    max_steps: int = 50,
) -> Array:
    """
    Bisection method for a scalar root of f in [a, b].

    Parameters
    ----------
    f : callable
        Scalar function f(x) -> scalar.
    a : Array
        Left bracket (f(a) and f(b) should have opposite signs).
    b : Array
        Right bracket.
    max_steps : int
        Number of bisection iterations. Default 50 gives ~15 decimal
        digits of precision.

    Returns
    -------
    Array
        Approximate root of f.

    Notes
    -----
    Uses lax.scan with fixed iteration count for full compatibility
    with jit, vmap, and grad. Does not terminate early.

    Example
    -------
    >>> @jax.jit
    ... def find_sqrt2(a, b):
    ...     return bisect(lambda x: x**2 - 2.0, a, b)
    >>> find_sqrt2(1.0, 2.0)  # Returns ~1.414...
    """
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    fa = f(a)
    fb = f(b)

    def step(carry, _):
        a_, b_, fa_, fb_ = carry
        mid = 0.5 * (a_ + b_)
        fm = f(mid)

        # Branchless selection of which half contains the root
        left_has_root = jnp.sign(fa_) * jnp.sign(fm) <= 0

        a_new = jnp.where(left_has_root, a_, mid)
        fa_new = jnp.where(left_has_root, fa_, fm)
        b_new = jnp.where(left_has_root, mid, b_)
        fb_new = jnp.where(left_has_root, fm, fb_)

        return (a_new, b_new, fa_new, fb_new), None

    init_carry = (a, b, fa, fb)
    (a_f, b_f, _, _), _ = lax.scan(step, init_carry, None, length=max_steps)
    return 0.5 * (a_f + b_f)


def newton(
    f: ScalarFn,
    x0: Array,
    max_steps: int = 30,
) -> Array:
    """
    Newton's method with automatic derivative computation.

    Parameters
    ----------
    f : callable
        Scalar function f(x) -> scalar. Must be differentiable.
    x0 : Array
        Initial guess.
    max_steps : int
        Number of Newton iterations.

    Returns
    -------
    Array
        Approximate root of f.

    Notes
    -----
    Computes the derivative automatically via jax.grad(f). Uses
    lax.scan with fixed iteration count for full compatibility
    with jit, vmap, and grad.

    For custom derivatives (e.g., when the automatic derivative is
    expensive or you have an analytical form), use newton_with_grad.

    Example
    -------
    >>> @jax.jit
    ... def find_sqrt2(x0):
    ...     return newton(lambda x: x**2 - 2.0, x0)
    >>> find_sqrt2(1.5)  # Returns ~1.414...
    """
    x0 = jnp.asarray(x0)
    df = jax.grad(f)

    def step(x, _):
        fx = f(x)
        dfx = df(x)
        # Safe division to avoid NaN if derivative is zero
        dfx_safe = jnp.where(dfx == 0.0, 1.0, dfx)
        x_new = x - fx / dfx_safe
        return x_new, None

    x_f, _ = lax.scan(step, x0, None, length=max_steps)
    return x_f


def newton_with_grad(
    f: ScalarFn,
    df: ScalarFn,
    x0: Array,
    max_steps: int = 30,
) -> Array:
    """
    Newton's method with user-provided derivative.

    Parameters
    ----------
    f : callable
        Scalar function f(x) -> scalar.
    df : callable
        Derivative df(x) -> scalar.
    x0 : Array
        Initial guess.
    max_steps : int
        Number of Newton iterations.

    Returns
    -------
    Array
        Approximate root of f.

    Notes
    -----
    Use this when you have an analytical derivative that's faster
    than automatic differentiation, or when the function is not
    directly differentiable by JAX.

    Uses lax.scan with fixed iteration count for full compatibility
    with jit, vmap, and grad.

    Example
    -------
    >>> @jax.jit
    ... def find_sqrt2(x0):
    ...     return newton_with_grad(
    ...         lambda x: x**2 - 2.0,
    ...         lambda x: 2.0 * x,
    ...         x0
    ...     )
    >>> find_sqrt2(1.5)  # Returns ~1.414...
    """
    x0 = jnp.asarray(x0)

    def step(x, _):
        fx = f(x)
        dfx = df(x)
        # Safe division to avoid NaN if derivative is zero
        dfx_safe = jnp.where(dfx == 0.0, 1.0, dfx)
        x_new = x - fx / dfx_safe
        return x_new, None

    x_f, _ = lax.scan(step, x0, None, length=max_steps)
    return x_f


def newton_ppf(
    u: Array,
    cdf: ScalarFn,
    x0: Array,
    lo: Array,
    hi: Array,
    pdf: Optional[ScalarFn] = None,
    n_iter: int = 20,
    pdf_floor: float = 1e-30,
) -> Array:
    r"""
    Generic percent-point-function (PPF / inverse-CDF) solver via Newton.

    Solves ``cdf(x) = u`` for ``x`` using a fixed number of Newton
    iterations. This is the inverse-CDF inversion at the heart of
    reparameterized sampling (e.g. IMF sampling): given a uniform draw
    ``u`` in ``(0, 1)``, it returns the quantile ``x = F^{-1}(u)``.

    Algorithm
    ---------
    Newton's method applied to the residual ``g(x) = cdf(x) - u``.
    Since ``g'(x) = cdf'(x) = pdf(x)``, each step is::

        x_{k+1} = x_k - (cdf(x_k) - u) / pdf(x_k)

    The PDF (CDF derivative) is supplied via ``pdf`` if available, else
    obtained automatically as ``jax.grad(cdf)`` (vmapped to handle array
    ``x``). The update is clipped to ``[lo, hi]`` after every step so the
    iterate never leaves the support. A small ``pdf_floor`` guards the
    division when the density is near zero (flat-CDF regions), preventing
    NaNs without biasing well-conditioned steps.

    This is deliberately decoupled from any specific distribution: callers
    pass their own ``cdf`` (closing over distribution parameters), an
    initial guess ``x0``, and the support bounds. A good ``x0`` (e.g. a
    closed-form approximate inverse) reduces the iterations needed.

    Parameters
    ----------
    u : Array
        Target CDF value(s) in ``(0, 1)``. Shape broadcasts with ``x0``.
    cdf : callable
        Cumulative distribution function ``cdf(x) -> F(x)``. Must be
        differentiable if ``pdf`` is not supplied.
    x0 : Array
        Initial guess for the quantile(s), same shape as ``u``.
    lo, hi : Array
        Lower/upper bounds of the support; iterates are clipped to
        ``[lo, hi]``.
    pdf : callable, optional
        Density ``pdf(x) = cdf'(x)``. If ``None`` (default), computed via
        automatic differentiation of ``cdf``.
    n_iter : int
        Number of Newton iterations (fixed; NOT convergence-based). Default
        20 gives ample precision for smooth unimodal CDFs from a reasonable
        guess.
    pdf_floor : float
        Additive floor on the density in the denominator (default 1e-30)
        guarding against division by ~0 in flat-CDF regions.

    Returns
    -------
    Array
        Quantile(s) ``x`` such that ``cdf(x) ~= u``.

    Notes
    -----
    Fully compatible with ``jit``, ``vmap``, and ``grad``: uses ``lax.scan``
    with a fixed iteration count (no ``while_loop``), additive (not
    branching) safe division, and ``clip`` for bounds. Gradients flow
    through every iteration, so the result is differentiable BOTH w.r.t.
    ``u`` AND w.r.t. any distribution parameters captured inside ``cdf``
    (or ``pdf``) — verified by FD-vs-AD grad-checks against the analytic
    exponential PPF.

    Example
    -------
    >>> import jax.numpy as jnp
    >>> # Exponential: F(x) = 1 - exp(-lam x), true PPF = -ln(1-u)/lam
    >>> lam = 2.0
    >>> u = jnp.array([0.1, 0.5, 0.9])
    >>> x = newton_ppf(u, lambda x: 1.0 - jnp.exp(-lam * x),
    ...                x0=jnp.ones_like(u), lo=0.0, hi=100.0)
    >>> jnp.allclose(x, -jnp.log1p(-u) / lam, atol=1e-6)
    Array(True, dtype=bool)
    """
    u = jnp.asarray(u)
    x0 = jnp.asarray(x0)
    lo = jnp.asarray(lo)
    hi = jnp.asarray(hi)

    if pdf is None:
        # Elementwise derivative of the (possibly vectorized) CDF. vmap over
        # a raveled view so jax.grad sees a scalar->scalar map, then restore
        # shape. This keeps the AD path correct for array-valued x.
        grad_scalar = jax.grad(lambda xi: jnp.reshape(cdf(xi), ()))

        def pdf_fn(x):
            x_arr = jnp.asarray(x)
            flat = jax.vmap(grad_scalar)(x_arr.ravel())
            return flat.reshape(x_arr.shape)
    else:
        pdf_fn = pdf

    def step(x, _):
        residual = cdf(x) - u
        dens = pdf_fn(x)
        x_new = x - residual / (dens + pdf_floor)
        x_new = jnp.clip(x_new, lo, hi)
        return x_new, None

    x_f, _ = lax.scan(step, x0, None, length=n_iter)
    return x_f


# Keep newton_1d as alias for backwards compatibility
newton_1d = newton_with_grad

__all__ = ["bisect", "newton", "newton_with_grad", "newton_1d", "newton_ppf"]
