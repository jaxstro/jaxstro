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


@jax.jit
def neumaier_add(
    s: jnp.ndarray,
    c: jnp.ndarray,
    y: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
def compensated_sum(*terms: jnp.ndarray) -> jnp.ndarray:
    """
    Sum multiple arrays with Neumaier compensated summation.

    This is intended for summing **a small number of arrays** of identical
    shape (e.g., a handful of force contributions). For large vector sums,
    prefer using `compensated_vector_sum`.

    Parameters
    ----------
    *terms : sequence of ndarray
        Arrays to sum, all with identical shape and dtype.

    Returns
    -------
    ndarray
        Sum with reduced accumulation error.
    """
    if not terms:
        raise ValueError("compensated_sum requires at least one term")
    if len(terms) == 1:
        return terms[0]

    s = jnp.zeros_like(terms[0])
    c = jnp.zeros_like(terms[0])

    # Number of terms is typically small and static, so a Python loop is fine.
    for y in terms:
        s, c = neumaier_add(s, c, y)

    return s + c


@jax.jit
def compensated_vector_sum(vectors: jnp.ndarray) -> jnp.ndarray:
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
    """
    if vectors.ndim != 2:
        raise ValueError(f"compensated_vector_sum expects (N, D) array, got shape {vectors.shape}")

    N, D = vectors.shape
    result = jnp.zeros(D, dtype=vectors.dtype)

    def body_fun(d, res):
        # Component-wise compensated sum over the N entries
        component_terms = [vectors[i, d] for i in range(N)]
        comp_sum = compensated_sum(*component_terms)
        return res.at[d].set(comp_sum)

    # D is typically small and known at trace time; a Python loop is sufficient.
    for d in range(D):
        result = body_fun(d, result)

    return result


@jax.jit
def compensated_dot(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
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
    """
    if a.shape != b.shape:
        raise ValueError(f"compensated_dot expects same-shaped vectors, got {a.shape} and {b.shape}")
    if a.ndim != 1:
        raise ValueError(f"compensated_dot expects 1D vectors, got ndim={a.ndim}")

    products = [a[i] * b[i] for i in range(a.shape[0])]
    return compensated_sum(*products)


__all__ = [
    "neumaier_add",
    "compensated_sum",
    "compensated_vector_sum",
    "compensated_dot",
]
