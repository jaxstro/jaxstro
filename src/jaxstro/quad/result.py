"""Typed evidence returned by adaptive integration methods."""

from enum import IntEnum
from typing import Any, NamedTuple

from jaxtyping import Array


class QuadStatus(IntEnum):
    CONVERGED = 0
    MAX_EVALUATIONS = 1
    MAX_REGIONS = 2
    NONFINITE_INTEGRAND = 3
    ROUNDOFF_LIMITED = 4
    DIVERGENCE_SUSPECTED = 5
    INVALID_INPUT = 6
    ERROR_ESTIMATE_UNAVAILABLE = 7


class ErrorKind(IntEnum):
    EMBEDDED_RULE = 0
    REFINEMENT_DIFFERENCE = 1
    SPARSE_GRID_SURPLUS = 2
    REPLICATE_STANDARD_ERROR = 3
    CONFIDENCE_INTERVAL_HALF_WIDTH = 4
    UNAVAILABLE = 5


class QuadError(NamedTuple):
    estimate: Any
    norm: Array
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
    tolerance: Array
    status: Array
    work: QuadWork
