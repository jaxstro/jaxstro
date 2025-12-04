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

import jax
import jax.numpy as jnp
import jax.lax as lax

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


# Keep newton_1d as alias for backwards compatibility
newton_1d = newton_with_grad

__all__ = ["bisect", "newton", "newton_with_grad", "newton_1d"]
