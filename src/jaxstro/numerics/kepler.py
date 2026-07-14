"""Universal-variable Cartesian Kepler propagation.

This module owns package-independent two-body numerical mechanics. Callers own
units, physical domain policy, object identity, and state-commit decisions.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array

KEPLER_STATUS_CONVERGED = 0
KEPLER_STATUS_INVALID_INPUT = 1
KEPLER_STATUS_NONFINITE_ITERATION = 2
KEPLER_STATUS_SINGULAR_RADIUS = 3
KEPLER_STATUS_MAX_STEPS = 4

_STUMPFF_SERIES_TERMS = 8
_STUMPFF_SERIES_THRESHOLD = 1.0


class UniversalKeplerResult(NamedTuple):
    """Cartesian conic result with one exhaustive numerical status."""

    position: Array
    velocity: Array
    universal_anomaly: Array
    residual: Array
    iterations: Array
    status: Array
    valid: Array


def _stumpff_series(z: Array) -> tuple[Array, Array]:
    """Evaluate the fixed eight-term power series for C(z) and S(z)."""

    c = jnp.asarray(0.0, dtype=z.dtype)
    s = jnp.asarray(0.0, dtype=z.dtype)
    c_term = jnp.asarray(0.5, dtype=z.dtype)
    s_term = jnp.asarray(1.0 / 6.0, dtype=z.dtype)
    for k in range(_STUMPFF_SERIES_TERMS):
        c = c + c_term
        s = s + s_term
        c_term = c_term * (-z) / ((2 * k + 3) * (2 * k + 4))
        s_term = s_term * (-z) / ((2 * k + 4) * (2 * k + 5))
    return c, s


def _stumpff_pair(z: Array) -> tuple[Array, Array]:
    """Evaluate Stumpff C(z), S(z) with sanitized branch expressions."""

    z = jnp.asarray(z, dtype=jnp.result_type(z, 0.0))
    abs_z = jnp.abs(z)
    threshold = jnp.asarray(_STUMPFF_SERIES_THRESHOLD, dtype=z.dtype)
    use_series = abs_z <= threshold

    series_z = jnp.where(use_series, z, jnp.zeros_like(z))
    c_series, s_series = _stumpff_series(series_z)

    positive_z = jnp.where(z > threshold, z, jnp.ones_like(z))
    positive_root = jnp.sqrt(positive_z)
    c_positive = (1.0 - jnp.cos(positive_root)) / positive_z
    s_positive = (
        positive_root - jnp.sin(positive_root)
    ) / positive_root**3

    negative_abs_z = jnp.where(z < -threshold, -z, jnp.ones_like(z))
    negative_root = jnp.sqrt(negative_abs_z)
    c_negative = (jnp.cosh(negative_root) - 1.0) / negative_abs_z
    s_negative = (
        jnp.sinh(negative_root) - negative_root
    ) / negative_root**3

    c_outer = jnp.where(z > 0.0, c_positive, c_negative)
    s_outer = jnp.where(z > 0.0, s_positive, s_negative)
    return (
        jnp.where(use_series, c_series, c_outer),
        jnp.where(use_series, s_series, s_outer),
    )


__all__ = [
    "KEPLER_STATUS_CONVERGED",
    "KEPLER_STATUS_INVALID_INPUT",
    "KEPLER_STATUS_NONFINITE_ITERATION",
    "KEPLER_STATUS_SINGULAR_RADIUS",
    "KEPLER_STATUS_MAX_STEPS",
    "UniversalKeplerResult",
]
