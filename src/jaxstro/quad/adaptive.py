"""Public one-dimensional adaptive quadrature evaluator."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from ._adaptive import (
    LocalEstimate,
    adaptive_controller,
    clenshaw_curtis_pair_data,
    infer_payload_zero,
    nested_rule_estimate_values,
    reference_partition,
    tanh_sinh_estimate_values,
    tanh_sinh_pair_data,
    transformed_integrand,
    validate_adaptive_capacities,
)
from ._gk import gauss_kronrod_data, gauss_kronrod_estimate_values
from .domains import Infinite, Interval, LeftInfinite, RightInfinite
from .measures import LebesgueMeasure, WeightedMeasure
from .methods import AdaptiveClenshawCurtis, AdaptiveTanhSinh, GaussKronrod
from .result import ErrorKind, QuadError, QuadResult, QuadStatus, QuadWork
from .tolerance import ErrorNorm, MaxNorm
from .tolerance import error_norm as reduce_error_norm

Domain = Interval | RightInfinite | LeftInfinite | Infinite
AdaptiveMeasure = LebesgueMeasure | WeightedMeasure


def _assembled_result(controller, norm: ErrorNorm, kind: ErrorKind) -> QuadResult:
    error_norm = reduce_error_norm(controller.error, norm)
    return QuadResult(
        value=controller.value,
        error=QuadError(
            estimate=controller.error,
            norm=error_norm,
            kind=jnp.asarray(kind, dtype=jnp.int32),
            confidence_level=jnp.asarray(jnp.nan, dtype=error_norm.dtype),
        ),
        tolerance=controller.tolerance,
        status=controller.status,
        work=QuadWork(
            evaluations=controller.evaluations,
            refinements=controller.refinements,
            active_regions=controller.active_regions,
            levels=jnp.asarray(0, dtype=jnp.int32),
            replicates=jnp.asarray(0, dtype=jnp.int32),
        ),
    )


def _zero_result(value, epsabs, epsrel, norm: ErrorNorm, kind: ErrorKind) -> QuadResult:
    error = jnp.zeros_like(jnp.real(value))
    value_norm = reduce_error_norm(value, norm)
    dtype = jnp.result_type(value_norm, epsabs, epsrel, 0.0)
    tolerance = jnp.maximum(
        jnp.asarray(epsabs, dtype=dtype),
        jnp.asarray(epsrel, dtype=dtype) * value_norm,
    )
    error_norm = reduce_error_norm(error, norm)
    return QuadResult(
        value=value,
        error=QuadError(
            estimate=error,
            norm=error_norm,
            kind=jnp.asarray(kind, dtype=jnp.int32),
            confidence_level=jnp.asarray(jnp.nan, dtype=error_norm.dtype),
        ),
        tolerance=tolerance,
        status=jnp.asarray(QuadStatus.CONVERGED, dtype=jnp.int32),
        work=QuadWork(
            evaluations=jnp.asarray(0, dtype=jnp.int32),
            refinements=jnp.asarray(0, dtype=jnp.int32),
            active_regions=jnp.asarray(0, dtype=jnp.int32),
            levels=jnp.asarray(0, dtype=jnp.int32),
            replicates=jnp.asarray(0, dtype=jnp.int32),
        ),
    )


def integrate(
    fun: Callable,
    domain: Domain,
    *,
    args: Any = (),
    method,
    measure: AdaptiveMeasure | None = None,
    epsabs,
    epsrel,
    max_evaluations: int,
    max_regions: int,
    error_norm: ErrorNorm = MaxNorm(),
    gradient: str = "stop",
) -> QuadResult:
    """Adaptively integrate a raw-array one-dimensional integrand."""
    if gradient != "stop":
        raise ValueError(
            'Phase A2 accepts only gradient="stop"; Phase A3 replay is not yet '
            "implemented"
        )
    if not isinstance(method, (GaussKronrod, AdaptiveClenshawCurtis, AdaptiveTanhSinh)):
        raise TypeError(f"{type(method).__name__} is not implemented in Phase A2")
    if not isinstance(domain, Interval) and not isinstance(method, AdaptiveTanhSinh):
        raise TypeError(f"{type(method).__name__} requires a finite Interval")
    selected_measure: AdaptiveMeasure = (
        LebesgueMeasure() if measure is None else measure
    )
    if not isinstance(selected_measure, (LebesgueMeasure, WeightedMeasure)):
        raise TypeError(
            "adaptive quadrature requires LebesgueMeasure or WeightedMeasure"
        )

    initial_regions = len(domain.breakpoints) + 1 if isinstance(domain, Interval) else 1
    if isinstance(method, GaussKronrod):
        node_cost = method.pair
    elif isinstance(method, AdaptiveClenshawCurtis):
        node_cost = method.initial_order
    else:
        node_cost = None
    if node_cost is not None:
        validate_adaptive_capacities(
            node_cost=node_cost,
            max_evaluations=max_evaluations,
            max_regions=max_regions,
            initial_regions=initial_regions,
        )

    absolute_tolerance = jnp.asarray(epsabs)
    relative_tolerance = jnp.asarray(epsrel)
    if absolute_tolerance.ndim != 0 or relative_tolerance.ndim != 0:
        raise ValueError("adaptive tolerances must be scalar")
    if jnp.issubdtype(absolute_tolerance.dtype, jnp.complexfloating) or jnp.issubdtype(
        relative_tolerance.dtype, jnp.complexfloating
    ):
        raise TypeError("adaptive tolerances must have a real dtype")

    if isinstance(domain, Interval):
        domain_scalars = (domain.lower, domain.upper, *domain.breakpoints)
    elif isinstance(domain, RightInfinite):
        domain_scalars = (domain.lower,)
    elif isinstance(domain, LeftInfinite):
        domain_scalars = (domain.upper,)
    else:
        domain_scalars = ()
    dtype = jnp.result_type(
        *domain_scalars, absolute_tolerance, relative_tolerance, 0.0
    )
    if isinstance(method, GaussKronrod):
        data = gauss_kronrod_data(method, dtype=dtype)
        rule_nodes = data.nodes
        error_kind = ErrorKind.EMBEDDED_RULE

        def reduce_values(values):
            return gauss_kronrod_estimate_values(values, data)

    elif isinstance(method, AdaptiveClenshawCurtis):
        pair = clenshaw_curtis_pair_data(method, dtype=dtype)
        rule_nodes = pair.nodes
        error_kind = ErrorKind.REFINEMENT_DIFFERENCE

        def reduce_values(values):
            return nested_rule_estimate_values(values, pair)

    else:
        tanh_sinh_pair = tanh_sinh_pair_data(method, dtype=dtype)
        rule_nodes = tanh_sinh_pair.nodes
        node_cost = rule_nodes.shape[0]
        error_kind = ErrorKind.REFINEMENT_DIFFERENCE
        validate_adaptive_capacities(
            node_cost=node_cost,
            max_evaluations=max_evaluations,
            max_regions=max_regions,
            initial_regions=initial_regions,
        )

        def reduce_values(values):
            return tanh_sinh_estimate_values(values, tanh_sinh_pair)

    partition = reference_partition(domain)
    if node_cost is None:
        raise AssertionError("adaptive method dispatch did not select a node cost")
    inferred_zero = infer_payload_zero(
        fun,
        args=args,
        node_count=node_cost,
        node_dtype=rule_nodes.dtype,
    )
    if jnp.issubdtype(inferred_zero.dtype, jnp.complexfloating):
        zero_dtype = (
            jnp.complex64 if rule_nodes.dtype == jnp.float32 else jnp.complex128
        )
    else:
        zero_dtype = rule_nodes.dtype
    zero_value = inferred_zero.astype(zero_dtype)

    def local_estimator(lower, upper):
        transformed = transformed_integrand(
            fun,
            domain,
            rule_nodes,
            region_lower=lower,
            region_upper=upper,
            args=args,
            measure=selected_measure,
            open_region=isinstance(method, AdaptiveTanhSinh),
        )
        estimate = reduce_values(transformed.values)
        return LocalEstimate(
            value=estimate.value,
            error=estimate.error,
            nonfinite=transformed.nonfinite | estimate.nonfinite,
            roundoff=transformed.roundoff,
        )

    tolerance_valid = (
        jnp.isfinite(absolute_tolerance)
        & jnp.isfinite(relative_tolerance)
        & (absolute_tolerance >= 0.0)
        & (relative_tolerance >= 0.0)
    )
    zero_width = (
        jnp.asarray(domain.lower) == jnp.asarray(domain.upper)
        if isinstance(domain, Interval)
        else jnp.asarray(False)
    )
    use_zero = partition.valid & tolerance_valid & zero_width

    def run_controller(_operand):
        controller = adaptive_controller(
            partition,
            local_estimator,
            node_cost=node_cost,
            max_evaluations=max_evaluations,
            max_regions=max_regions,
            epsabs=epsabs,
            epsrel=epsrel,
            error_norm=error_norm,
        )
        return _assembled_result(controller, error_norm, error_kind)

    result = jax.lax.cond(
        use_zero,
        lambda _operand: _zero_result(
            zero_value, epsabs, epsrel, error_norm, error_kind
        ),
        run_controller,
        operand=None,
    )
    return jax.tree.map(jax.lax.stop_gradient, result)


__all__ = ["integrate"]
