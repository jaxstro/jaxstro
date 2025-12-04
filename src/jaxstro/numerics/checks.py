# src/jaxstro/numerics/checks.py
"""
Common numerical validation helpers for jaxstro.

These functions provide small, JAX-friendly checks for:
    - finiteness (no NaNs / infs),
    - monotonicity,
    - value ranges.

They are intended for use in library code and tests to catch
silent numerical issues early, while remaining JIT-friendly
for the pure predicate-style functions.

Design:
    - Pure checks (is_* / all_* / in_range) are JAX JIT compatible.
    - assert_* helpers are plain Python and meant for eager/tests.
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp

from .types import Array


# ---------------------------------------------------------------------------
# Finiteness
# ---------------------------------------------------------------------------

@jax.jit
def is_finite(x: Array) -> Array:
    """
    Elementwise finiteness check (no NaNs, no +/-inf).

    Parameters
    ----------
    x : ndarray
        Input array.

    Returns
    -------
    ndarray of bool
        True where x is finite.
    """
    return jnp.isfinite(x)


@jax.jit
def all_finite(x: Array) -> Array:
    """
    Check that all elements of x are finite.

    Parameters
    ----------
    x : ndarray
        Input array.

    Returns
    -------
    scalar boolean ndarray
        True if all elements are finite, False otherwise.
    """
    return jnp.all(jnp.isfinite(x))


def assert_all_finite(x: Array, name: str = "x") -> None:
    """
    Raise a ValueError if any element of x is NaN or +/-inf.

    Intended for eager code, tests, and debug-mode guards.

    Parameters
    ----------
    x : ndarray
        Input array.
    name : str, optional
        Name to display in error messages.
    """
    finite_mask = jnp.isfinite(x)
    if bool(jnp.all(finite_mask)):
        return

    # Build a small diagnostic: count and a sample index
    n_total = x.size
    n_bad = int(n_total - int(jnp.sum(finite_mask)))
    # first bad index in flattened array
    bad_indices = jnp.argwhere(~finite_mask)
    sample_idx = tuple(map(int, bad_indices[0])) if bad_indices.size > 0 else None

    raise ValueError(
        f"{name} contains {n_bad} non-finite values out of {n_total} "
        f"(e.g. at index {sample_idx})."
    )


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("strict",))
def is_monotonic_increasing(x: Array, *, strict: bool = True) -> Array:
    """
    Check whether a 1D array is monotonic increasing.

    Parameters
    ----------
    x : ndarray, shape (N,)
        Input array.
    strict : bool, optional
        If True, require strictly increasing (x[i+1] > x[i]).
        If False, allow equal values (x[i+1] >= x[i]).

    Returns
    -------
    scalar boolean ndarray
        True if x is monotonic increasing along its only axis.

    Notes
    -----
    This is a predicate, not an assert; it never raises inside JAX.
    Assumes input is 1D; behavior undefined for higher dimensions.
    """
    dx = jnp.diff(x)
    if strict:
        return jnp.all(dx > 0)
    else:
        return jnp.all(dx >= 0)


@partial(jax.jit, static_argnames=("strict",))
def is_monotonic_decreasing(x: Array, *, strict: bool = True) -> Array:
    """
    Check whether a 1D array is monotonic decreasing.

    Parameters
    ----------
    x : ndarray, shape (N,)
        Input array.
    strict : bool, optional
        If True, require strictly decreasing (x[i+1] < x[i]).
        If False, allow equal values (x[i+1] <= x[i]).

    Returns
    -------
    scalar boolean ndarray
        True if x is monotonic decreasing along its only axis.
    """
    dx = jnp.diff(x)
    if strict:
        return jnp.all(dx < 0)
    else:
        return jnp.all(dx <= 0)


# Alias for backwards compatibility
is_monotonic = is_monotonic_increasing


def assert_monotonic(
    x: Array,
    *,
    strict: bool = True,
    decreasing: bool = False,
    name: str = "x",
) -> None:
    """
    Raise if x is not monotonic.

    Parameters
    ----------
    x : ndarray, shape (N,)
        Input grid.
    strict : bool, optional
        If True, require strictly monotonic.
        If False, allow equal values.
    decreasing : bool, optional
        If True, check for decreasing. If False (default), check increasing.
    name : str, optional
        Name to display in error messages.
    """
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D for monotonicity check, got ndim={x.ndim}")

    if decreasing:
        mono = bool(is_monotonic_decreasing(x, strict=strict))
        direction = "decreasing"
        relation = "<" if strict else "<="
    else:
        mono = bool(is_monotonic_increasing(x, strict=strict))
        direction = "increasing"
        relation = ">" if strict else ">="

    if mono:
        return

    dx = jnp.diff(x)
    if decreasing:
        bad_mask = dx > 0 if strict else dx >= 0
    else:
        bad_mask = dx <= 0 if strict else dx < 0

    bad_indices = jnp.argwhere(bad_mask)
    # argwhere returns shape (n, 1) for 1D input, so flatten to get index
    sample_pair = int(bad_indices[0, 0]) if bad_indices.size > 0 else None

    raise ValueError(
        f"{name} is not monotonic {direction} (strict={strict}). "
        f"Example violation at index {sample_pair}: "
        f"{name}[i+1] {relation} {name}[i] not satisfied."
    )


# ---------------------------------------------------------------------------
# Range checks
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("inclusive",))
def in_range(
    x: Array,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
    *,
    inclusive: bool = True,
) -> Array:
    """
    Elementwise range check.

    Parameters
    ----------
    x : ndarray
        Input values.
    lo : float or None, optional
        Lower bound. If None, no lower bound is applied.
    hi : float or None, optional
        Upper bound. If None, no upper bound is applied.
    inclusive : bool, optional
        If True, use <= / >=. If False, use < / >.

    Returns
    -------
    ndarray of bool
        Mask where x satisfies the specified bounds.
    """
    mask = jnp.ones_like(x, dtype=bool)

    if lo is not None:
        lo_arr = jnp.asarray(lo)
        if inclusive:
            mask = jnp.logical_and(mask, x >= lo_arr)
        else:
            mask = jnp.logical_and(mask, x > lo_arr)

    if hi is not None:
        hi_arr = jnp.asarray(hi)
        if inclusive:
            mask = jnp.logical_and(mask, x <= hi_arr)
        else:
            mask = jnp.logical_and(mask, x < hi_arr)

    return mask


@jax.jit
def all_in_range(
    x: Array,
    lo: float,
    hi: float,
) -> Array:
    """
    Check that all elements of x are within [lo, hi] inclusive.

    Parameters
    ----------
    x : ndarray
        Input array.
    lo : float
        Lower bound.
    hi : float
        Upper bound.

    Returns
    -------
    scalar boolean ndarray
        True if all elements satisfy lo <= x <= hi.
    """
    return jnp.all((x >= lo) & (x <= hi))


@jax.jit
def all_positive(x: Array) -> Array:
    """
    Check that all elements of x are strictly positive.

    Parameters
    ----------
    x : ndarray
        Input array.

    Returns
    -------
    scalar boolean ndarray
        True if all elements are > 0.
    """
    return jnp.all(x > 0)


@jax.jit
def all_non_negative(x: Array) -> Array:
    """
    Check that all elements of x are non-negative.

    Parameters
    ----------
    x : ndarray
        Input array.

    Returns
    -------
    scalar boolean ndarray
        True if all elements are >= 0.
    """
    return jnp.all(x >= 0)


def assert_in_range(
    x: Array,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
    *,
    inclusive: bool = True,
    name: str = "x",
) -> None:
    """
    Raise if any element of x is outside the given bounds.

    Parameters
    ----------
    x : ndarray
        Input values.
    lo : float or None, optional
        Lower bound. If None, no lower bound.
    hi : float or None, optional
        Upper bound. If None, no upper bound.
    inclusive : bool, optional
        If True, use <= / >=. If False, use < / >.
    name : str, optional
        Name to display in error messages.
    """
    mask = in_range(x, lo=lo, hi=hi, inclusive=inclusive)
    if bool(jnp.all(mask)):
        return

    bad_mask = ~mask
    n_total = x.size
    n_bad = int(n_total - int(jnp.sum(mask)))
    bad_indices = jnp.argwhere(bad_mask)
    sample_idx = tuple(map(int, bad_indices[0])) if bad_indices.size > 0 else None

    bounds_desc = []
    if lo is not None:
        op = ">=" if inclusive else ">"
        bounds_desc.append(f"{op} {lo}")
    if hi is not None:
        op = "<=" if inclusive else "<"
        bounds_desc.append(f"{op} {hi}")
    bounds_str = " and ".join(bounds_desc) if bounds_desc else "unbounded"

    raise ValueError(
        f"{name} has {n_bad} values out of range ({bounds_str}) "
        f"out of {n_total} total (example index {sample_idx})."
    )


def assert_positive(x: Array, name: str = "x") -> None:
    """
    Raise if any element of x is not strictly positive.

    Parameters
    ----------
    x : ndarray
        Input values.
    name : str, optional
        Name to display in error messages.
    """
    assert_in_range(x, lo=0.0, inclusive=False, name=name)


def assert_non_negative(x: Array, name: str = "x") -> None:
    """
    Raise if any element of x is negative.

    Parameters
    ----------
    x : ndarray
        Input values.
    name : str, optional
        Name to display in error messages.
    """
    assert_in_range(x, lo=0.0, inclusive=True, name=name)


__all__ = [
    # Finiteness
    "is_finite",
    "all_finite",
    "assert_all_finite",
    # Monotonicity
    "is_monotonic",
    "is_monotonic_increasing",
    "is_monotonic_decreasing",
    "assert_monotonic",
    # Range
    "in_range",
    "all_in_range",
    "all_positive",
    "all_non_negative",
    "assert_in_range",
    "assert_positive",
    "assert_non_negative",
]
