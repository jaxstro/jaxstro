"""Typed evidence returned by adaptive integration methods."""

from enum import IntEnum
from typing import Any, NamedTuple

import jax.numpy as jnp
from jaxtyping import Array

from .tolerance import ErrorNorm, tolerance_threshold
from .tolerance import error_norm as reduce_error_norm


class QuadStatus(IntEnum):
    CONVERGED = 0
    MAX_EVALUATIONS = 1
    MAX_REGIONS = 2
    NONFINITE_INTEGRAND = 3
    ROUNDOFF_LIMITED = 4
    DIVERGENCE_SUSPECTED = 5
    INVALID_INPUT = 6
    ERROR_ESTIMATE_UNAVAILABLE = 7
    MAX_INDICES = 8


class ErrorKind(IntEnum):
    EMBEDDED_RULE = 0
    REFINEMENT_DIFFERENCE = 1
    SPARSE_GRID_SURPLUS = 2
    REPLICATE_STANDARD_ERROR = 3
    CONFIDENCE_INTERVAL_HALF_WIDTH = 4
    UNAVAILABLE = 5


class QuadError(NamedTuple):
    estimate: Any
    norm: Any
    kind: Array
    confidence_level: Array


class QuadWork(NamedTuple):
    evaluations: Array
    refinements: Array
    active_regions: Array
    levels: Array
    replicates: Array


class QuadResult(NamedTuple):
    value: Any
    error: QuadError
    tolerance: Any
    status: Array
    work: QuadWork


def unavailable_result(
    value,
    *,
    epsabs,
    epsrel,
    error_norm: ErrorNorm,
    evaluations: int,
    status,
) -> QuadResult:
    """Assemble a result for a formula without runtime error evidence."""
    value = jnp.asarray(value)
    if not jnp.issubdtype(value.dtype, jnp.inexact):
        value = jnp.asarray(value, dtype=jnp.result_type(value, 0.0))
    estimate_dtype = jnp.result_type(jnp.real(value), 0.0)
    estimate = jnp.full(value.shape, jnp.nan, dtype=estimate_dtype)
    estimate_norm = reduce_error_norm(estimate, error_norm)
    tolerance = tolerance_threshold(
        value,
        epsabs=epsabs,
        epsrel=epsrel,
        norm=error_norm,
    )
    status = jnp.asarray(status, dtype=jnp.int32)
    failed = (status == QuadStatus.INVALID_INPUT) | (
        status == QuadStatus.NONFINITE_INTEGRAND
    )
    sentinel = jnp.full_like(value, jnp.nan)
    value = jnp.where(failed, sentinel, value)
    zero = jnp.asarray(0, dtype=jnp.int32)
    return QuadResult(
        value=value,
        error=QuadError(
            estimate=estimate,
            norm=estimate_norm,
            kind=jnp.asarray(ErrorKind.UNAVAILABLE, dtype=jnp.int32),
            confidence_level=jnp.asarray(jnp.nan, dtype=estimate_norm.dtype),
        ),
        tolerance=tolerance,
        status=status,
        work=QuadWork(
            evaluations=jnp.asarray(evaluations, dtype=jnp.int32),
            refinements=zero,
            active_regions=zero,
            levels=zero,
            replicates=zero,
        ),
    )


def zero_volume_result(
    value,
    *,
    epsabs,
    epsrel,
    error_norm: ErrorNorm,
) -> QuadResult:
    """Return an exact zero-volume value without claiming an error estimator."""
    return unavailable_result(
        value,
        epsabs=epsabs,
        epsrel=epsrel,
        error_norm=error_norm,
        evaluations=0,
        status=QuadStatus.ERROR_ESTIMATE_UNAVAILABLE,
    )


__all__ = [
    "ErrorKind",
    "QuadError",
    "QuadResult",
    "QuadStatus",
    "QuadWork",
    "unavailable_result",
    "zero_volume_result",
]
