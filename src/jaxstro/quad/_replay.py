"""Private fixed-formula replay differentiation for adaptive quadrature."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ._adaptive import select_segment, transformed_integrand
from ._gk import gauss_kronrod_data, gauss_kronrod_estimate_values
from .methods import GaussKronrod
from .result import QuadError, QuadResult, QuadWork


class RegionalReplayEvidence(NamedTuple):
    segment_local_lower: Any
    segment_local_upper: Any
    segment_id: Any
    active_mask: Any


class GlobalReplayEvidence(NamedTuple):
    accepted_level: Any


class PrimalSolve(NamedTuple):
    result: QuadResult
    evidence: RegionalReplayEvidence | GlobalReplayEvidence


@dataclass(frozen=True)
class IntegrateConfig:
    fun: Callable
    method: Any
    measure: Any
    max_evaluations: int
    max_regions: int
    error_norm: Any


def _zero_tangent(leaf):
    leaf = jnp.asarray(leaf)
    if jnp.issubdtype(leaf.dtype, jnp.inexact):
        return jnp.zeros_like(leaf)
    return np.zeros(leaf.shape, dtype=jax.dtypes.float0)


def result_tangent(result: QuadResult, value_tangent) -> QuadResult:
    """Attach the value JVP and exact zero tangents to diagnostics."""
    return QuadResult(
        value=value_tangent,
        error=QuadError(
            estimate=jax.tree.map(_zero_tangent, result.error.estimate),
            norm=_zero_tangent(result.error.norm),
            kind=_zero_tangent(result.error.kind),
            confidence_level=_zero_tangent(result.error.confidence_level),
        ),
        tolerance=_zero_tangent(result.tolerance),
        status=_zero_tangent(result.status),
        work=QuadWork(*(_zero_tangent(leaf) for leaf in result.work)),
    )


def replay_value(
    config: IntegrateConfig,
    domain,
    args,
    evidence: RegionalReplayEvidence | GlobalReplayEvidence,
    primal_value,
):
    """Reconstruct the stopped accepted quadrature formula."""
    if not isinstance(config.method, GaussKronrod) or not isinstance(
        evidence, RegionalReplayEvidence
    ):
        raise TypeError(f"{type(config.method).__name__} replay is not implemented")

    data = gauss_kronrod_data(
        config.method,
        dtype=evidence.segment_local_lower.dtype,
    )
    zero = jnp.zeros_like(primal_value)

    def evaluate_region(inputs):
        lower, upper, segment_id, active = inputs

        def evaluate(_operand):
            segment_domain = select_segment(domain, segment_id)
            transformed = transformed_integrand(
                config.fun,
                segment_domain,
                data.nodes,
                region_lower=lower,
                region_upper=upper,
                args=args,
                measure=config.measure,
                replay=True,
            )
            return gauss_kronrod_estimate_values(transformed.values, data).value

        return jax.lax.cond(active, evaluate, lambda _operand: zero, operand=None)

    values = jax.lax.map(
        evaluate_region,
        (
            evidence.segment_local_lower,
            evidence.segment_local_upper,
            evidence.segment_id,
            evidence.active_mask,
        ),
    )
    return jnp.sum(values, axis=0)


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def integrate_replay_core(config, domain, args, epsabs, epsrel):
    from .adaptive import _solve_raw

    return _solve_raw(config, domain, args, epsabs, epsrel).result


@integrate_replay_core.defjvp
def _integrate_replay_core_jvp(config, primals, tangents):
    from .adaptive import _solve_raw

    domain, args, epsabs, epsrel = primals
    domain_tangent, args_tangent, _epsabs_tangent, _epsrel_tangent = tangents
    solve = _solve_raw(config, domain, args, epsabs, epsrel)
    evidence = jax.tree.map(jax.lax.stop_gradient, solve.evidence)

    def replay(live_domain, live_args):
        return replay_value(
            config,
            live_domain,
            live_args,
            evidence,
            solve.result.value,
        )

    _, value_tangent = jax.jvp(
        replay,
        (domain, args),
        (domain_tangent, args_tangent),
    )
    return solve.result, result_tangent(solve.result, value_tangent)


__all__: list[str] = []
