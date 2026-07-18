"""Stopped fixed-look confidence intervals for randomized QMC replicates."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array


class FixedLookInterval(NamedTuple):
    mean: Array
    sample_variance: Array
    standard_error: Array
    critical_value: Array
    half_width: Array
    valid: Array


def _student_t_survival(value: Array, degrees_of_freedom: Array) -> Array:
    ratio = degrees_of_freedom / (degrees_of_freedom + value**2)
    return 0.5 * jsp.special.betainc(
        0.5 * degrees_of_freedom,
        0.5,
        ratio,
    )


def _student_t_positive_mass(value: Array, degrees_of_freedom: Array) -> Array:
    ratio = value**2 / (degrees_of_freedom + value**2)
    return 0.5 * jsp.special.betainc(
        0.5,
        0.5 * degrees_of_freedom,
        ratio,
    )


def student_t_quantile(probability: Array, degrees_of_freedom) -> Array:
    """Return a stopped positive Student-t quantile or NaN when unsupported."""
    probability = jnp.asarray(probability)
    if probability.ndim != 0:
        raise ValueError("Student-t probability must be scalar")
    if jnp.issubdtype(probability.dtype, jnp.complexfloating):
        raise TypeError("Student-t probability must have a real dtype")
    dtype = jnp.result_type(probability, 0.0)
    probability = jnp.asarray(probability, dtype=dtype)
    degrees = jnp.asarray(degrees_of_freedom, dtype=dtype)
    use_center_mass = probability <= jnp.asarray(0.75, dtype=dtype)
    target_center_mass = probability - jnp.asarray(0.5, dtype=dtype)
    target_survival = jnp.asarray(1.0, dtype=dtype) - probability
    candidates = jnp.asarray(2.0, dtype=dtype) ** jnp.arange(32, dtype=dtype)
    center_mass = _student_t_positive_mass(candidates, degrees)
    survival = _student_t_survival(candidates, degrees)
    covered = jnp.where(
        use_center_mass,
        center_mass >= target_center_mass,
        survival <= target_survival,
    )
    bracketed = jnp.any(covered)
    first = jnp.argmax(covered)
    initial_upper = candidates[first]

    def bisect_step(_, bounds):
        lower, upper = bounds
        midpoint = 0.5 * (lower + upper)
        midpoint_center_mass = _student_t_positive_mass(midpoint, degrees)
        midpoint_survival = _student_t_survival(midpoint, degrees)
        below = jnp.where(
            use_center_mass,
            midpoint_center_mass < target_center_mass,
            midpoint_survival > target_survival,
        )
        return jax.lax.cond(
            below,
            lambda _: (midpoint, upper),
            lambda _: (lower, midpoint),
            operand=None,
        )

    lower, upper = jax.lax.fori_loop(
        0,
        80,
        bisect_step,
        (jnp.asarray(0.0, dtype=dtype), initial_upper),
    )
    supported = (
        jnp.isfinite(probability)
        & jnp.isfinite(degrees)
        & (probability > 0.5)
        & (probability < 1.0)
        & (degrees > 0.0)
        & bracketed
    )
    quantile = jnp.where(supported, 0.5 * (lower + upper), jnp.nan)
    return jax.lax.stop_gradient(quantile)


def fixed_look_interval(
    estimates: Array,
    *,
    confidence_level,
) -> FixedLookInterval:
    """Compute one two-sided Student-t interval from independent replicates."""
    estimates = jnp.asarray(estimates)
    if estimates.ndim != 1:
        raise ValueError("fixed-look replicate estimates must be one-dimensional")
    if estimates.shape[0] < 2:
        raise ValueError("fixed-look intervals require at least two replicates")
    if jnp.issubdtype(estimates.dtype, jnp.complexfloating):
        raise TypeError("fixed-look replicate estimates must be real")
    confidence = jnp.asarray(confidence_level, dtype=estimates.dtype)
    if confidence.ndim != 0:
        raise ValueError("confidence_level must be scalar")
    replicate_count = estimates.shape[0]
    mean = jnp.mean(estimates)
    centered = estimates - mean
    sample_variance = jnp.sum(centered * centered) / (replicate_count - 1)
    standard_error = jnp.sqrt(sample_variance / replicate_count)
    critical = student_t_quantile(
        0.5 * (1.0 + confidence),
        replicate_count - 1,
    )
    half_width = critical * standard_error
    valid = (
        jnp.all(jnp.isfinite(estimates))
        & jnp.isfinite(confidence)
        & (confidence > 0.0)
        & (confidence < 1.0)
        & jnp.isfinite(critical)
        & jnp.isfinite(half_width)
    )
    return FixedLookInterval(
        mean=mean,
        sample_variance=sample_variance,
        standard_error=standard_error,
        critical_value=critical,
        half_width=half_width,
        valid=valid,
    )


def spent_alpha(alpha, inspection) -> Array:
    """Allocate one summable inspection-wise error probability."""
    alpha = jnp.asarray(alpha)
    inspection = jnp.asarray(inspection, dtype=alpha.dtype)
    return alpha * 6.0 / (jnp.pi**2 * (inspection + 1.0) ** 2)


def empirical_bernstein_half_width(
    estimates: Array,
    *,
    lower,
    upper,
    alpha,
) -> Array:
    """Return the bounded empirical-Bernstein half-width for one inspection."""
    estimates = jnp.asarray(estimates)
    if estimates.ndim != 1:
        raise ValueError("empirical-Bernstein estimates must be one-dimensional")
    if estimates.shape[0] < 2:
        raise ValueError(
            "empirical-Bernstein evidence requires at least two replicates"
        )
    if jnp.issubdtype(estimates.dtype, jnp.complexfloating):
        raise TypeError("empirical-Bernstein estimates must be real")
    dtype = jnp.result_type(estimates, lower, upper, alpha, 0.0)
    estimates = jnp.asarray(estimates, dtype=dtype)
    lower = jnp.asarray(lower, dtype=dtype)
    upper = jnp.asarray(upper, dtype=dtype)
    alpha = jnp.asarray(alpha, dtype=dtype)
    if lower.ndim != 0 or upper.ndim != 0 or alpha.ndim != 0:
        raise ValueError("empirical-Bernstein bounds and alpha must be scalar")
    replicate_count = estimates.shape[0]
    mean = jnp.mean(estimates)
    variance = jnp.sum((estimates - mean) ** 2) / (replicate_count - 1)
    log_term = jnp.log(2.0 / alpha)
    variance_term = jnp.sqrt(2.0 * variance * log_term / replicate_count)
    range_term = 7.0 * (upper - lower) * log_term / (3.0 * (replicate_count - 1))
    valid = (
        jnp.all(jnp.isfinite(estimates))
        & jnp.isfinite(lower)
        & jnp.isfinite(upper)
        & (lower <= upper)
        & jnp.isfinite(alpha)
        & (alpha > 0.0)
        & (alpha < 1.0)
    )
    return jnp.where(valid, variance_term + range_term, jnp.nan)


__all__ = [
    "FixedLookInterval",
    "empirical_bernstein_half_width",
    "fixed_look_interval",
    "spent_alpha",
    "student_t_quantile",
]
