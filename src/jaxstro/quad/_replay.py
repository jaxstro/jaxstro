"""Private fixed-formula replay differentiation for adaptive quadrature."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ._adaptive import (
    clenshaw_curtis_pair_data,
    nested_rule_estimate_values,
    select_segment,
    tanh_sinh_estimate_values,
    tanh_sinh_pair_data,
    transformed_integrand,
)
from ._gk import gauss_kronrod_data, gauss_kronrod_estimate_values
from ._romberg import romberg_replay_value, romberg_tanh_sinh_replay_value
from .methods import (
    AdaptiveClenshawCurtis,
    AdaptiveTanhSinh,
    GaussKronrod,
    Romberg,
    RombergTanhSinh,
)
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


def _regional_rule(config: IntegrateConfig, dtype):
    if isinstance(config.method, GaussKronrod):
        data = gauss_kronrod_data(config.method, dtype=dtype)
        return (
            data.nodes,
            lambda values: gauss_kronrod_estimate_values(values, data).value,
            False,
        )
    if isinstance(config.method, AdaptiveClenshawCurtis):
        clenshaw_curtis_pair = clenshaw_curtis_pair_data(config.method, dtype=dtype)
        return (
            clenshaw_curtis_pair.nodes,
            lambda values: (
                nested_rule_estimate_values(
                    values,
                    clenshaw_curtis_pair,
                ).value
            ),
            False,
        )
    if isinstance(config.method, AdaptiveTanhSinh):
        tanh_sinh_pair = tanh_sinh_pair_data(config.method, dtype=dtype)
        return (
            tanh_sinh_pair.nodes,
            lambda values: (
                tanh_sinh_estimate_values(
                    values,
                    tanh_sinh_pair,
                ).value
            ),
            True,
        )
    raise TypeError(f"{type(config.method).__name__} is not a regional replay method")


def replay_value(
    config: IntegrateConfig,
    domain,
    args,
    evidence: RegionalReplayEvidence | GlobalReplayEvidence,
    primal_value,
):
    """Reconstruct the stopped accepted quadrature formula."""
    if isinstance(evidence, GlobalReplayEvidence):
        dtype = jnp.real(jnp.asarray(primal_value)).dtype
        zero = jnp.zeros_like(primal_value)

        def evaluate_one(reference):
            transformed = transformed_integrand(
                config.fun,
                domain,
                jnp.reshape(reference, (1,)),
                args=args,
                measure=config.measure,
                replay=True,
            )
            return (
                transformed.values[0].astype(zero.dtype),
                transformed.nonfinite,
                transformed.roundoff,
            )

        replay_engine = (
            romberg_replay_value
            if isinstance(config.method, Romberg)
            else romberg_tanh_sinh_replay_value
            if isinstance(config.method, RombergTanhSinh)
            else None
        )
        if replay_engine is None:
            raise TypeError(
                f"{type(config.method).__name__} is not a global replay method"
            )
        return replay_engine(
            evaluate_one,
            zero,
            initial_level=config.method.initial_level,
            accepted_level=evidence.accepted_level,
            max_evaluations=config.max_evaluations,
            dtype=dtype,
        )

    if not isinstance(evidence, RegionalReplayEvidence):
        raise TypeError(f"{type(config.method).__name__} replay is not implemented")

    nodes, reduce_values, open_region = _regional_rule(
        config,
        evidence.segment_local_lower.dtype,
    )
    zero = jnp.zeros_like(primal_value)

    def evaluate_region(inputs):
        lower, upper, segment_id, active = inputs

        def evaluate(_operand):
            segment_domain = select_segment(domain, segment_id)
            transformed = transformed_integrand(
                config.fun,
                segment_domain,
                nodes,
                region_lower=lower,
                region_upper=upper,
                args=args,
                measure=config.measure,
                open_region=open_region,
                replay=True,
            )
            return reduce_values(transformed.values)

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
