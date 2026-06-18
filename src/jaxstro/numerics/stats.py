# src/jaxstro/numerics/stats.py

"""
Small numerical-statistics helpers for JAX.

These are tiny wrappers and utilities for building log-likelihoods,
losses, and probability models, without depending on a heavy stats
package. For full probabilistic modeling, use numpyro / blackjax /
MCX or your preferred stack.
"""

from typing import Optional

import jax
import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


@jax.jit
def safe_log(x: Float[Array, "..."], eps: float = 1e-30) -> Float[Array, "..."]:
    """
    Compute log(x) with clipping away from zero.

    Parameters
    ----------
    x : ndarray
        Input values.
    eps : float, optional
        Minimum value allowed before taking log.

    Returns
    -------
    ndarray
        log(clipped x).
    """
    return jnp.log(jnp.clip(x, min=eps))


@jax.jit
def logsumexp(
    x: Float[Array, "..."],
    axis: Optional[int] = None,
    keepdims: bool = False,
) -> Float[Array, "..."]:
    """
    Stable log(sum(exp(x))) along a given axis.

    Thin wrapper around jax.nn.logsumexp with a slightly more
    NumPy-like signature.

    Parameters
    ----------
    x : ndarray
        Input array.
    axis : int or None, optional
        Axis over which to reduce. If None, reduce over all dims.
    keepdims : bool, optional
        If True, keep reduced dimensions with size 1.

    Returns
    -------
    ndarray
        logsumexp result.
    """
    return jnn.logsumexp(x, axis=axis, keepdims=keepdims)


@jax.jit
def gaussian_logpdf(
    x: Float[Array, "..."],
    mu: Float[Array, "..."],
    sigma: Float[Array, "..."],
) -> Float[Array, "..."]:
    """
    Log PDF of a univariate normal N(mu, sigma^2).

    Parameters
    ----------
    x : ndarray
        Evaluation points.
    mu : ndarray
        Mean(s), broadcastable to x.
    sigma : ndarray
        Standard deviation(s), broadcastable to x.

    Returns
    -------
    ndarray
        log p(x | mu, sigma).
    """
    var = sigma**2
    norm_const = -0.5 * jnp.log(2.0 * jnp.pi * var)
    return norm_const - 0.5 * (x - mu) ** 2 / var


@jax.jit
def gaussian_loglikelihood(
    data: Float[Array, "..."],
    mu: Float[Array, "..."],
    sigma: Float[Array, "..."],
    axis: Optional[int] = None,
) -> Float[Array, "..."]:
    """
    Gaussian log-likelihood for independent observations.

    Parameters
    ----------
    data : ndarray
        Observations.
    mu : ndarray
        Mean(s), broadcastable to data.
    sigma : ndarray
        Standard deviation(s), broadcastable to data.
    axis : int or None, optional
        Axis over which to sum the log-likelihood. If None,
        sum over all axes.

    Returns
    -------
    ndarray
        Total log-likelihood.
    """
    logpdf = gaussian_logpdf(data, mu, sigma)
    if axis is None:
        return jnp.sum(logpdf)
    return jnp.sum(logpdf, axis=axis)


@jax.jit
def stable_log1p(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Stable log(1 + x) for small |x|.

    Thin wrapper around jax.numpy.log1p.
    """
    return jnp.log1p(x)


@jax.jit
def stable_expm1(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Stable exp(x) - 1 for small |x|.

    Thin wrapper around jax.numpy.expm1.
    """
    return jnp.expm1(x)


@jax.jit
def safe_exp(ln_x: Float[Array, "..."], max_exp: float = 100.0) -> Float[Array, "..."]:
    """
    Numerically safe exponential with clipping to prevent overflow.

    Parameters
    ----------
    ln_x : ndarray
        Log of input value or array.
    max_exp : float, optional
        Maximum exponent (default: 100.0).

    Returns
    -------
    ndarray
        exp(min(ln_x, max_exp)).
    """
    return jnp.exp(jnp.minimum(ln_x, max_exp))


@jax.jit
def safe_div(
    numerator: Float[Array, "..."],
    denominator: Float[Array, "..."],
    epsilon: float = 1e-100,
) -> Float[Array, "..."]:
    """
    Numerically safe division with small epsilon to prevent division by zero.

    Parameters
    ----------
    numerator : ndarray
        Numerator value or array.
    denominator : ndarray
        Denominator value or array.
    epsilon : float, optional
        Small value to add to denominator (default: 1e-100).

    Returns
    -------
    ndarray
        numerator / (denominator + epsilon).

    Notes
    -----
    - Prevents division by zero without branching.
    - Maintains differentiability for JAX autodiff.
    - Epsilon chosen small enough to not affect normal values.
    """
    return numerator / (denominator + epsilon)


@jax.jit
def relative_error(
    x_new: Float[Array, "..."],
    x_old: Float[Array, "..."],
    floor: float = 1e-100,
) -> Float[Array, "..."]:
    """
    Compute relative error between two values.

    Parameters
    ----------
    x_new : ndarray
        New value or array.
    x_old : ndarray
        Old value or array.
    floor : float, optional
        Minimum denominator to prevent division by zero.

    Returns
    -------
    ndarray
        |x_new - x_old| / max(|x_old|, floor).

    Notes
    -----
    - Always non-negative.
    - Returns value in [0, inf).
    - Floor prevents division by zero.
    """
    return jnp.abs(x_new - x_old) / jnp.maximum(jnp.abs(x_old), floor)


@jax.jit
def check_convergence(
    x_new: Float[Array, "..."],
    x_old: Float[Array, "..."],
    tol: float = 1e-6,
) -> Bool[Array, ""]:
    """
    Check if iteration has converged.

    Parameters
    ----------
    x_new : ndarray
        New value or array.
    x_old : ndarray
        Old value or array.
    tol : float, optional
        Convergence tolerance (default: 1e-6).

    Returns
    -------
    ndarray (bool)
        True if max(relative_error) < tol, False otherwise.

    Notes
    -----
    - Uses relative error, not absolute.
    - Applies to worst (maximum) error across array.
    - Returns JAX boolean (use .item() for Python bool).
    """
    rel_err = relative_error(x_new, x_old)
    return jnp.max(rel_err) < tol


__all__ = [
    "safe_log",
    "safe_exp",
    "safe_div",
    "logsumexp",
    "gaussian_logpdf",
    "gaussian_loglikelihood",
    "stable_log1p",
    "stable_expm1",
    "relative_error",
    "check_convergence",
]
