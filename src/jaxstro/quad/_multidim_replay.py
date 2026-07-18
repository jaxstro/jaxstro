"""Accepted-formula evidence for multidimensional replay differentiation.

The adaptive controllers own method selection.  This module owns the smaller
stopped object that survives that selection: a normalized quadrature formula
on the unit hyperrectangle.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax.numpy as jnp

from ._multidim import evaluate_multidim
from .measures import LebesgueMeasure
from .result import QuadResult


@dataclass(frozen=True)
class MultidimConfig:
    """Static solver configuration retained by the private replay boundary."""

    fun: Callable
    method: Any
    measure: Any
    max_evaluations: int
    max_regions: int | None
    max_indices: int | None
    max_frontier: int | None
    max_nodes: int | None
    error_norm: Any


class ReplayFormula(NamedTuple):
    """One fixed-capacity normalized formula selected by a primal solve."""

    reference_points: Any
    reference_weights: Any
    active_mask: Any


class MultidimPrimalSolve(NamedTuple):
    """Private primal result and the exact accepted formula that produced it."""

    result: QuadResult
    formula: ReplayFormula
    config: MultidimConfig
    domain: Any
    args: Any


def replay_formula_value(
    config: MultidimConfig,
    domain,
    args,
    formula: ReplayFormula,
):
    """Evaluate a stopped normalized formula on the physical domain."""
    evaluated = evaluate_multidim(
        config.fun,
        domain,
        formula.reference_points,
        args=args,
        measure=LebesgueMeasure() if config.measure is None else config.measure,
    )
    coefficients = jnp.where(
        formula.active_mask,
        formula.reference_weights,
        jnp.zeros_like(formula.reference_weights),
    )
    factors = coefficients * evaluated.weights
    factor_shape = (factors.shape[0],) + (1,) * (evaluated.values.ndim - 1)
    return jnp.sum(evaluated.values * factors.reshape(factor_shape), axis=0)


__all__ = [
    "MultidimConfig",
    "MultidimPrimalSolve",
    "ReplayFormula",
    "replay_formula_value",
]
