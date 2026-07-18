"""Accepted-formula evidence for multidimensional replay differentiation.

The adaptive controllers own method selection.  This module owns the smaller
stopped object that survives that selection: a normalized quadrature formula
on the unit hyperrectangle.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from ._multidim import evaluate_multidim
from ._replay import result_tangent
from .measures import LebesgueMeasure
from .result import QuadResult, QuadStatus


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
    active = jax.lax.stop_gradient(jnp.asarray(formula.active_mask))
    points = jax.lax.stop_gradient(jnp.asarray(formula.reference_points))
    weights = jax.lax.stop_gradient(jnp.asarray(formula.reference_weights))
    safe_index = jnp.argmax(active)
    safe_point = points[safe_index]
    evaluation_points = jnp.where(
        active[:, None],
        points,
        safe_point[None, :],
    )
    evaluated = evaluate_multidim(
        config.fun,
        domain,
        evaluation_points,
        args=args,
        measure=LebesgueMeasure() if config.measure is None else config.measure,
    )
    factors = jnp.where(
        active,
        weights * evaluated.weights,
        jnp.zeros_like(weights),
    )
    active_values = active.reshape(
        active.shape + (1,) * (evaluated.values.ndim - active.ndim)
    )
    values = jnp.where(
        active_values,
        evaluated.values,
        jnp.zeros_like(evaluated.values),
    )
    factor_shape = factors.shape + (1,) * (values.ndim - factors.ndim)
    return jnp.sum(values * factors.reshape(factor_shape), axis=0)


def _replay_primal_result(result: QuadResult, domain) -> QuadResult:
    zero_width = jnp.any(jnp.asarray(domain.lower) == domain.upper)
    status = jnp.where(
        zero_width,
        jnp.asarray(QuadStatus.INVALID_INPUT, dtype=jnp.int32),
        result.status,
    )
    return result._replace(status=status)


@jax.custom_jvp
def _first_order_tangent(value):
    return value


@_first_order_tangent.defjvp
def _reject_higher_replay_derivative(_primals, _tangents):
    raise ValueError("multidimensional replay supports first derivatives only")


def _contains_higher_order_trace(tree) -> bool:
    return any(
        type(leaf).__name__ in {"JVPTracer", "LinearizeTracer"}
        for leaf in jax.tree.leaves(tree)
    )


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def multidim_replay_core(config, domain, args, key, epsabs, epsrel):
    """Return the primal solve while reserving its accepted formula for JVP."""
    from .integrate import _solve_multidim

    solve = _solve_multidim(config, domain, args, key, epsabs, epsrel)
    return _replay_primal_result(solve.result, domain)


@multidim_replay_core.defjvp
def _multidim_replay_jvp(config, primals, tangents):
    from .integrate import _solve_multidim

    domain, args, key, epsabs, epsrel = primals
    domain_tangent, args_tangent, _key_tangent, _, _ = tangents
    if _contains_higher_order_trace((domain, args)):
        raise ValueError("multidimensional replay supports first derivatives only")
    solve = _solve_multidim(config, domain, args, key, epsabs, epsrel)
    formula = jax.tree.map(jax.lax.stop_gradient, solve.formula)
    _, value_tangent = jax.jvp(
        lambda live_domain, live_args: replay_formula_value(
            config,
            live_domain,
            live_args,
            formula,
        ),
        (domain, args),
        (domain_tangent, args_tangent),
    )
    zero_width = jnp.any(jnp.asarray(domain.lower) == domain.upper)
    value_tangent = jnp.where(
        zero_width,
        jnp.full_like(value_tangent, jnp.nan),
        value_tangent,
    )
    value_tangent = _first_order_tangent(value_tangent)
    result = _replay_primal_result(solve.result, domain)
    return result, result_tangent(result, value_tangent)


__all__ = [
    "MultidimConfig",
    "MultidimPrimalSolve",
    "ReplayFormula",
    "multidim_replay_core",
    "replay_formula_value",
]
