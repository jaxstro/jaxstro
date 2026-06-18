# src/jaxstro/numerics/compensated.py

"""
Compensated summation helpers (Neumaier-style) for JAX.

These routines reduce rounding errors when summing many terms, or when
terms span a wide range of magnitudes or have significant cancellation.

They are domain-agnostic and can be used in:
    - N-body force / energy accumulation (gravax),
    - stellar structure integrals (startrax, stellax),
    - flux integrals and likelihood sums (fluxax),
    - hydro / MHD updates (nebulax).

All functions are fully JAX-native: compatible with `jit`, `vmap`, and `grad`,
and work with dynamic array sizes without recompilation.

References
----------
Neumaier (1974), Z. Angew. Math. Mech., 54, 39–51
Higham (2002), Accuracy and Stability of Numerical Algorithms
Rein & Spiegel (2015), MNRAS, 446, 1424 (IAS15 integrator)
Kahan (1965), Commun. ACM, 8, 40–41
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float


@jax.jit
def neumaier_add(
    s: Float[Array, "..."],
    c: Float[Array, "..."],
    y: Float[Array, "..."],
) -> Tuple[Float[Array, "..."], Float[Array, "..."]]:
    """
    Single step of Neumaier's compensated summation.

    Parameters
    ----------
    s : ndarray
        Running sum.
    c : ndarray
        Running compensation term.
    y : ndarray
        New term to add.

    Returns
    -------
    s_new, c_new : tuple of ndarray
        Updated sum and compensation.
    """
    t = s + y
    bigger = jnp.abs(s) >= jnp.abs(y)
    c_new = c + jnp.where(bigger, (s - t) + y, (y - t) + s)
    return t, c_new


@jax.jit
def compensated_sum_array(terms: Float[Array, "k ..."]) -> Float[Array, "..."]:
    """
    Sum array elements along axis 0 with Neumaier compensated summation.

    This is the JAX-native version that works with dynamic array sizes.
    Use this when you have terms stacked into an array.

    Parameters
    ----------
    terms : ndarray, shape (K, ...) or (K,)
        Array of K terms to sum. Can be 1D (K scalars) or higher-dimensional
        (K arrays of identical shape).

    Returns
    -------
    ndarray, shape (...) or scalar
        Sum with reduced accumulation error, shape matches terms[0].

    Examples
    --------
    >>> terms = jnp.array([1e16, 1.0, -1e16, 1.0])
    >>> compensated_sum_array(terms)  # Returns 2.0, not 0.0
    """
    if terms.ndim == 0:
        return terms

    # Scan function for compensated sum (works for any shape)
    def _scan_fn(carry, y):
        s, c = carry
        s_new, c_new = neumaier_add(s, c, y)
        return (s_new, c_new), None

    # Handle 1D case: sum scalars
    if terms.ndim == 1:
        init = (jnp.zeros((), dtype=terms.dtype), jnp.zeros((), dtype=terms.dtype))
        (s_final, c_final), _ = lax.scan(_scan_fn, init, terms)
        return s_final + c_final

    # Handle ND case: sum along axis 0
    init = (jnp.zeros_like(terms[0]), jnp.zeros_like(terms[0]))
    (s_final, c_final), _ = lax.scan(_scan_fn, init, terms)
    return s_final + c_final


def compensated_sum(*terms: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Sum multiple arrays with Neumaier compensated summation.

    This is a convenience wrapper that accepts variadic arguments.
    For JIT-compiled code with dynamic numbers of terms, prefer
    `compensated_sum_array` with terms stacked into an array.

    Parameters
    ----------
    *terms : sequence of ndarray
        Arrays to sum, all with identical shape and dtype.

    Returns
    -------
    ndarray
        Sum with reduced accumulation error.

    Notes
    -----
    When JIT-compiled, this function will retrace if called with different
    numbers of arguments. For dynamic-length sums, stack terms into an array
    and use `compensated_sum_array` instead.

    Examples
    --------
    >>> a, b, c = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]), jnp.array([5.0, 6.0])
    >>> compensated_sum(a, b, c)
    Array([9., 12.], dtype=float32)
    """
    if not terms:
        raise ValueError("compensated_sum requires at least one term")
    if len(terms) == 1:
        return terms[0]

    # Stack terms and use the array-based implementation
    stacked = jnp.stack(terms, axis=0)
    return compensated_sum_array(stacked)


@jax.jit
def compensated_vector_sum(vectors: Float[Array, "n d"]) -> Float[Array, " d"]:
    """
    Sum vectors along axis 0 using compensated summation.

    Parameters
    ----------
    vectors : ndarray, shape (N, D)
        Stack of N vectors in D dimensions.

    Returns
    -------
    ndarray, shape (D,)
        Component-wise sum of all vectors.

    Examples
    --------
    >>> vecs = jnp.array([[1e16, 1.0], [1.0, -1e16], [-1e16, 1e16], [1.0, 1.0]])
    >>> compensated_vector_sum(vecs)  # Returns [2.0, 2.0]
    """
    if vectors.ndim != 2:
        raise ValueError(
            f"compensated_vector_sum expects (N, D) array, got shape {vectors.shape}"
        )

    # Use lax.scan over the N dimension
    def scan_fn(carry, vec):
        s, c = carry
        s_new, c_new = neumaier_add(s, c, vec)
        return (s_new, c_new), None

    D = vectors.shape[1]
    init = (jnp.zeros(D, dtype=vectors.dtype), jnp.zeros(D, dtype=vectors.dtype))
    (s_final, c_final), _ = lax.scan(scan_fn, init, vectors)
    return s_final + c_final


@jax.jit
def compensated_dot(a: Float[Array, " n"], b: Float[Array, " n"]) -> Float[Array, ""]:
    """
    Compute dot product with compensated summation.

    Parameters
    ----------
    a, b : ndarray, shape (N,)
        Input vectors.

    Returns
    -------
    scalar ndarray
        Dot product a·b with reduced accumulation error.

    Examples
    --------
    >>> a = jnp.array([1e16, 1.0, -1e16])
    >>> b = jnp.array([1.0, 1.0, 1.0])
    >>> compensated_dot(a, b)  # Returns 1.0, not 0.0
    """
    if a.shape != b.shape:
        raise ValueError(
            f"compensated_dot expects same-shaped vectors, got {a.shape} and {b.shape}"
        )
    if a.ndim != 1:
        raise ValueError(f"compensated_dot expects 1D vectors, got ndim={a.ndim}")

    # Compute products and sum with compensation
    products = a * b
    return compensated_sum_array(products)


__all__ = [
    "neumaier_add",
    "compensated_sum",
    "compensated_sum_array",
    "compensated_vector_sum",
    "compensated_dot",
]
