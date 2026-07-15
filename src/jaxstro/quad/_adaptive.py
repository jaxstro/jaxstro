"""Shared reference-partition and transformed-integrand substrate."""

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ._chebyshev import chebyshev_rule_data
from ._integrand import (
    call_integrand,
    density_values,
    expand_node_factor,
    has_explicit_args,
    infer_payload_zero,
    validate_node_values,
)
from .domains import Infinite, Interval, LeftInfinite, RightInfinite, interval_is_valid
from .measures import LebesgueMeasure, WeightedMeasure
from .methods import AdaptiveClenshawCurtis
from .result import QuadStatus
from .rules import ClenshawCurtisRule
from .tolerance import ErrorNorm
from .tolerance import error_norm as reduce_error_norm
from .transforms import map_domain

Domain = Interval | RightInfinite | LeftInfinite | Infinite
AdaptiveMeasure = LebesgueMeasure | WeightedMeasure


class ReferencePartition(NamedTuple):
    """Fixed-shape normalized regions and dynamic domain validity."""

    lower: Array
    upper: Array
    valid: Array


class TransformedIntegrand(NamedTuple):
    """Mapped nodes and contribution values on one normalized region."""

    reference: Array
    x: Array
    jacobian: Array
    values: Array
    valid: Array
    nonfinite: Array


class LocalEstimate(NamedTuple):
    """Method-specific value and payload error for one reference region."""

    value: Array
    error: Array
    nonfinite: Array


class NestedRulePair(NamedTuple):
    """One high rule and its independently weighted nested low subset."""

    nodes: Array
    high_weights: Array
    low_indices: Array
    low_weights: Array


class NestedRuleEstimate(NamedTuple):
    """High-rule estimate with refinement-difference error evidence."""

    value: Array
    error: Array
    raw_error: Array
    roundoff_floor: Array
    nonfinite: Array


class AdaptiveControllerResult(NamedTuple):
    """Global estimate, work, status, and fixed-capacity partition evidence."""

    value: Array
    error: Array
    tolerance: Array
    status: Array
    evaluations: Array
    refinements: Array
    active_regions: Array
    region_lower: Array
    region_upper: Array
    region_active: Array
    no_improvement_count: Array
    growth_count: Array


class _ControllerState(NamedTuple):
    values: Array
    errors: Array
    priorities: Array
    lower: Array
    upper: Array
    active: Array
    global_value: Array
    global_error: Array
    tolerance: Array
    status: Array
    evaluations: Array
    refinements: Array
    active_regions: Array
    no_improvement_count: Array
    growth_count: Array


def clenshaw_curtis_pair_data(
    method: AdaptiveClenshawCurtis, *, dtype=None
) -> NestedRulePair:
    """Construct a nested Clenshaw-Curtis pair through the A1 cosine owner."""
    high = chebyshev_rule_data(ClenshawCurtisRule(method.initial_order), dtype=dtype)
    low_order = (method.initial_order + 1) // 2
    low = chebyshev_rule_data(ClenshawCurtisRule(low_order), dtype=dtype)
    return NestedRulePair(
        nodes=high.nodes,
        high_weights=high.weights,
        low_indices=jnp.arange(0, method.initial_order, 2, dtype=jnp.int32),
        low_weights=low.weights,
    )


def _node_weighted_sum(values: Array, weights: Array) -> Array:
    shape = (weights.shape[0],) + (1,) * (values.ndim - 1)
    return jnp.sum(values * jnp.reshape(weights, shape), axis=0)


def nested_rule_estimate_values(
    values: Array, pair: NestedRulePair
) -> NestedRuleEstimate:
    """Reduce one nested pair after a single high-node evaluation."""
    values = validate_node_values(
        values, pair.nodes.shape[0], context="nested quadrature"
    )
    if jnp.issubdtype(values.dtype, jnp.complexfloating):
        target_dtype = (
            jnp.complex64 if pair.nodes.dtype == jnp.float32 else jnp.complex128
        )
    else:
        target_dtype = pair.nodes.dtype
    values = values.astype(target_dtype)
    high_value = _node_weighted_sum(values, pair.high_weights)
    low_value = _node_weighted_sum(values[pair.low_indices], pair.low_weights)
    raw_error = jnp.abs(high_value - low_value)
    resabs = _node_weighted_sum(jnp.abs(values), pair.high_weights)
    machine = jnp.finfo(pair.nodes.dtype)
    floor = jnp.where(
        resabs > machine.tiny / (50.0 * machine.eps),
        50.0 * machine.eps * resabs,
        0.0,
    )
    error = jnp.maximum(raw_error, floor)
    nonfinite = ~(
        jnp.all(jnp.isfinite(values))
        & jnp.all(jnp.isfinite(high_value))
        & jnp.all(jnp.isfinite(low_value))
        & jnp.all(jnp.isfinite(raw_error))
        & jnp.all(jnp.isfinite(resabs))
        & jnp.all(jnp.isfinite(floor))
        & jnp.all(jnp.isfinite(error))
    )
    return NestedRuleEstimate(
        value=high_value,
        error=error,
        raw_error=raw_error,
        roundoff_floor=floor,
        nonfinite=nonfinite,
    )


def validate_adaptive_capacities(
    *, node_cost: int, max_evaluations: int, max_regions: int, initial_regions: int
) -> None:
    """Reject structurally impossible adaptive workspaces before tracing user code."""
    for name, value in (
        ("node_cost", node_cost),
        ("max_evaluations", max_evaluations),
        ("max_regions", max_regions),
        ("initial_regions", initial_regions),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"adaptive {name} must be a positive integer")
    initial_cost = initial_regions * node_cost
    if max_regions < initial_regions:
        raise ValueError("max_regions is smaller than the initial partition")
    if max_evaluations < initial_cost:
        raise ValueError("max_evaluations is smaller than the initial node cost")


def _masked_region_sum(values: Array, active: Array) -> Array:
    mask = jnp.reshape(active, active.shape + (1,) * (values.ndim - 1))
    return jnp.sum(jnp.where(mask, values, 0.0), axis=0)


def _controller_tolerance(value_norm: Array, epsabs, epsrel) -> Array:
    dtype = jnp.result_type(value_norm, epsabs, epsrel, 0.0)
    absolute = jnp.asarray(epsabs, dtype=dtype)
    relative = jnp.asarray(epsrel, dtype=dtype) * value_norm
    return jnp.maximum(absolute, relative)


def _value_stagnation_scale(parent_value: Array, child_value: Array, norm) -> Array:
    parent_norm = reduce_error_norm(parent_value, norm)
    child_norm = reduce_error_norm(child_value, norm)
    real_dtype = jnp.real(jnp.asarray(parent_value)).dtype
    machine = jnp.finfo(real_dtype)
    return (
        32.0
        * machine.eps
        * jnp.maximum(jnp.maximum(parent_norm, child_norm), machine.tiny)
    )


def _stagnation_hits(
    *,
    value_delta: Array,
    value_scale: Array,
    child_priority: Array,
    parent_priority: Array,
    refinements: Array,
) -> tuple[Array, Array]:
    no_improvement = (value_delta <= value_scale) & (
        child_priority >= 0.99 * parent_priority
    )
    growth = (refinements >= 10) & (child_priority > 1.01 * parent_priority)
    return no_improvement, growth


def reference_partition(domain: Domain) -> ReferencePartition:
    """Build initial normalized regions without encoding physical orientation."""
    if isinstance(domain, Interval):
        dtype = jnp.result_type(domain.lower, domain.upper, *domain.breakpoints, 0.0)
        lower = jnp.asarray(domain.lower, dtype=dtype)
        upper = jnp.asarray(domain.upper, dtype=dtype)
        lo = jnp.minimum(lower, upper)
        hi = jnp.maximum(lower, upper)
        half_width = 0.5 * (hi - lo)
        safe_half_width = jnp.where(half_width > 0.0, half_width, 1.0)
        midpoint = 0.5 * (hi + lo)
        if domain.breakpoints:
            points = jax.lax.stop_gradient(
                jnp.sort(jnp.asarray(domain.breakpoints, dtype=dtype))
            )
            normalized = (points - midpoint) / safe_half_width
        else:
            normalized = jnp.empty((0,), dtype=dtype)
        endpoints = jnp.concatenate(
            (
                jnp.asarray([-1.0], dtype=dtype),
                normalized,
                jnp.asarray([1.0], dtype=dtype),
            )
        )
        return ReferencePartition(
            lower=endpoints[:-1],
            upper=endpoints[1:],
            valid=interval_is_valid(domain),
        )

    dtype = (
        jnp.result_type(domain.lower, 0.0)
        if isinstance(domain, RightInfinite)
        else jnp.result_type(domain.upper, 0.0)
        if isinstance(domain, LeftInfinite)
        else jnp.asarray(0.0).dtype
    )
    if isinstance(domain, RightInfinite):
        valid = jnp.isfinite(jnp.asarray(domain.lower))
    elif isinstance(domain, LeftInfinite):
        valid = jnp.isfinite(jnp.asarray(domain.upper))
    elif isinstance(domain, Infinite):
        valid = jnp.asarray(True)
    else:
        raise TypeError("unsupported Phase A integration domain")
    return ReferencePartition(
        lower=jnp.asarray([-1.0], dtype=dtype),
        upper=jnp.asarray([1.0], dtype=dtype),
        valid=valid,
    )


def transformed_integrand(
    fun: Callable,
    domain: Domain,
    nodes: Array,
    *,
    region_lower=-1.0,
    region_upper=1.0,
    args: Any = (),
    measure: AdaptiveMeasure | None = None,
) -> TransformedIntegrand:
    """Evaluate one local reference region with every map and density applied."""
    selected_measure: AdaptiveMeasure = (
        LebesgueMeasure() if measure is None else measure
    )
    if not isinstance(selected_measure, (LebesgueMeasure, WeightedMeasure)):
        raise TypeError(
            "adaptive quadrature requires LebesgueMeasure or WeightedMeasure"
        )

    nodes = jnp.asarray(nodes)
    lower = jnp.asarray(region_lower, dtype=nodes.dtype)
    upper = jnp.asarray(region_upper, dtype=nodes.dtype)
    half_width = 0.5 * (upper - lower)
    midpoint = 0.5 * (upper + lower)
    reference = midpoint + half_width * nodes
    mapped = map_domain(domain, reference)
    has_args = has_explicit_args(args)
    raw_values = validate_node_values(
        call_integrand(fun, mapped.x, args, has_args),
        nodes.shape[0],
        context="adaptive quadrature",
    )
    density = density_values(selected_measure, mapped.x, args)
    jacobian = half_width * mapped.jacobian
    node_factor = expand_node_factor(
        mapped.orientation * density * jacobian, raw_values.ndim
    )
    values = raw_values * node_factor
    local_valid = (
        jnp.isfinite(lower)
        & jnp.isfinite(upper)
        & (lower >= -1.0)
        & (upper <= 1.0)
        & (lower <= upper)
    )
    valid = mapped.valid & local_valid
    nonfinite = ~(
        jnp.all(jnp.isfinite(reference))
        & jnp.all(jnp.isfinite(mapped.x))
        & jnp.all(jnp.isfinite(jacobian))
        & jnp.all(jnp.isfinite(density))
        & jnp.all(jnp.isfinite(raw_values))
        & jnp.all(jnp.isfinite(values))
    )
    return TransformedIntegrand(
        reference=reference,
        x=mapped.x,
        jacobian=jacobian,
        values=values,
        valid=valid,
        nonfinite=nonfinite,
    )


def adaptive_controller(
    partition: ReferencePartition,
    local_estimator: Callable[[Array, Array], LocalEstimate],
    *,
    node_cost: int,
    max_evaluations: int,
    max_regions: int,
    epsabs,
    epsrel,
    error_norm: ErrorNorm,
) -> AdaptiveControllerResult:
    """Run one deterministic fixed-capacity h-adaptive refinement loop."""
    initial_regions = partition.lower.shape[0]
    validate_adaptive_capacities(
        node_cost=node_cost,
        max_evaluations=max_evaluations,
        max_regions=max_regions,
        initial_regions=initial_regions,
    )
    initial_cost = initial_regions * node_cost
    absolute_tolerance = jnp.asarray(epsabs)
    relative_tolerance = jnp.asarray(epsrel)
    if absolute_tolerance.ndim != 0 or relative_tolerance.ndim != 0:
        raise ValueError("adaptive tolerances must be scalar")
    if jnp.issubdtype(absolute_tolerance.dtype, jnp.complexfloating) or jnp.issubdtype(
        relative_tolerance.dtype, jnp.complexfloating
    ):
        raise TypeError("adaptive tolerances must have a real dtype")

    initial = jax.vmap(local_estimator)(partition.lower, partition.upper)
    if initial.error.shape[1:] != initial.value.shape[1:]:
        raise ValueError("adaptive local error must match the value payload shape")
    if not jnp.issubdtype(initial.error.dtype, jnp.floating):
        raise TypeError("adaptive local error must have a real floating dtype")
    priorities = jax.vmap(lambda error: reduce_error_norm(error, error_norm))(
        initial.error
    )
    payload_shape = initial.value.shape[1:]
    error_shape = initial.error.shape[1:]
    values = (
        jnp.zeros((max_regions,) + payload_shape, dtype=initial.value.dtype)
        .at[:initial_regions]
        .set(initial.value)
    )
    errors = (
        jnp.zeros((max_regions,) + error_shape, dtype=initial.error.dtype)
        .at[:initial_regions]
        .set(initial.error)
    )
    region_priorities = (
        jnp.full((max_regions,), -jnp.inf, dtype=priorities.dtype)
        .at[:initial_regions]
        .set(priorities)
    )
    lower = (
        jnp.zeros((max_regions,), dtype=partition.lower.dtype)
        .at[:initial_regions]
        .set(partition.lower)
    )
    upper = (
        jnp.zeros((max_regions,), dtype=partition.upper.dtype)
        .at[:initial_regions]
        .set(partition.upper)
    )
    active = jnp.arange(max_regions) < initial_regions
    global_value = _masked_region_sum(values, active)
    global_error = _masked_region_sum(errors, active)
    value_norm = reduce_error_norm(global_value, error_norm)
    global_error_norm = reduce_error_norm(global_error, error_norm)
    tolerance = _controller_tolerance(value_norm, epsabs, epsrel)
    tolerance_valid = (
        jnp.isfinite(absolute_tolerance)
        & jnp.isfinite(relative_tolerance)
        & (absolute_tolerance >= 0.0)
        & (relative_tolerance >= 0.0)
    )
    initial_nonfinite = (
        jnp.any(initial.nonfinite)
        | ~jnp.all(jnp.isfinite(global_value))
        | ~jnp.all(jnp.isfinite(global_error))
        | ~jnp.all(jnp.isfinite(priorities))
        | ~jnp.isfinite(value_norm)
        | ~jnp.isfinite(global_error_norm)
        | ~jnp.isfinite(tolerance)
    )
    converged = global_error_norm <= tolerance
    running = jnp.asarray(-1, dtype=jnp.int32)
    status = jnp.where(
        ~(partition.valid & tolerance_valid),
        jnp.asarray(QuadStatus.INVALID_INPUT, dtype=jnp.int32),
        jnp.where(
            initial_nonfinite,
            jnp.asarray(QuadStatus.NONFINITE_INTEGRAND, dtype=jnp.int32),
            jnp.where(
                converged,
                jnp.asarray(QuadStatus.CONVERGED, dtype=jnp.int32),
                running,
            ),
        ),
    )
    state = _ControllerState(
        values=values,
        errors=errors,
        priorities=region_priorities,
        lower=lower,
        upper=upper,
        active=active,
        global_value=global_value,
        global_error=global_error,
        tolerance=tolerance,
        status=status,
        evaluations=jnp.asarray(initial_cost, dtype=jnp.int32),
        refinements=jnp.asarray(0, dtype=jnp.int32),
        active_regions=jnp.asarray(initial_regions, dtype=jnp.int32),
        no_improvement_count=jnp.asarray(0, dtype=jnp.int32),
        growth_count=jnp.asarray(0, dtype=jnp.int32),
    )

    def condition(current: _ControllerState) -> Array:
        return current.status == running

    def body(current: _ControllerState) -> _ControllerState:
        selected = jnp.argmax(jnp.where(current.active, current.priorities, -jnp.inf))
        region_lower = current.lower[selected]
        region_upper = current.upper[selected]
        midpoint = 0.5 * (region_lower + region_upper)
        midpoint_collapsed = (midpoint == region_lower) | (midpoint == region_upper)
        evaluation_exhausted = current.evaluations + 2 * node_cost > max_evaluations
        region_exhausted = current.active_regions + 1 > max_regions
        can_split = ~(midpoint_collapsed | evaluation_exhausted | region_exhausted)

        def stop_without_split(operand: _ControllerState) -> _ControllerState:
            stop_status = jnp.where(
                midpoint_collapsed,
                jnp.asarray(QuadStatus.ROUNDOFF_LIMITED, dtype=jnp.int32),
                jnp.where(
                    evaluation_exhausted,
                    jnp.asarray(QuadStatus.MAX_EVALUATIONS, dtype=jnp.int32),
                    jnp.asarray(QuadStatus.MAX_REGIONS, dtype=jnp.int32),
                ),
            )
            return operand._replace(status=stop_status)

        def split(operand: _ControllerState) -> _ControllerState:
            child_lower = jnp.stack((region_lower, midpoint))
            child_upper = jnp.stack((midpoint, region_upper))
            children = jax.vmap(local_estimator)(child_lower, child_upper)
            child_priorities = jax.vmap(
                lambda error: reduce_error_norm(error, error_norm)
            )(children.error)
            append_index = operand.active_regions
            new_values = operand.values.at[selected].set(children.value[0])
            new_values = new_values.at[append_index].set(children.value[1])
            new_errors = operand.errors.at[selected].set(children.error[0])
            new_errors = new_errors.at[append_index].set(children.error[1])
            new_priorities = operand.priorities.at[selected].set(child_priorities[0])
            new_priorities = new_priorities.at[append_index].set(child_priorities[1])
            new_lower = operand.lower.at[selected].set(region_lower)
            new_lower = new_lower.at[append_index].set(midpoint)
            new_upper = operand.upper.at[selected].set(midpoint)
            new_upper = new_upper.at[append_index].set(region_upper)
            new_active = operand.active.at[append_index].set(True)

            parent_value = operand.values[selected]
            child_value = children.value[0] + children.value[1]
            new_global_value = _masked_region_sum(new_values, new_active)
            new_global_error = _masked_region_sum(new_errors, new_active)
            new_value_norm = reduce_error_norm(new_global_value, error_norm)
            new_error_norm = reduce_error_norm(new_global_error, error_norm)
            new_tolerance = _controller_tolerance(new_value_norm, epsabs, epsrel)
            new_refinements = operand.refinements + 1

            value_delta = reduce_error_norm(child_value - parent_value, error_norm)
            value_scale = _value_stagnation_scale(parent_value, child_value, error_norm)
            parent_priority = operand.priorities[selected]
            child_priority = child_priorities[0] + child_priorities[1]
            no_improvement_hit, growth_hit = _stagnation_hits(
                value_delta=value_delta,
                value_scale=value_scale,
                child_priority=child_priority,
                parent_priority=parent_priority,
                refinements=new_refinements,
            )
            no_improvement_count = jnp.where(
                no_improvement_hit, operand.no_improvement_count + 1, 0
            )
            growth_count = jnp.where(growth_hit, operand.growth_count + 1, 0)
            child_nonfinite = (
                jnp.any(children.nonfinite)
                | ~jnp.all(jnp.isfinite(new_global_value))
                | ~jnp.all(jnp.isfinite(new_global_error))
                | ~jnp.all(jnp.isfinite(child_priorities))
                | ~jnp.isfinite(new_value_norm)
                | ~jnp.isfinite(new_error_norm)
                | ~jnp.isfinite(new_tolerance)
            )
            now_converged = new_error_norm <= new_tolerance
            roundoff = (no_improvement_count >= 6) | (growth_count >= 5)
            new_status = jnp.where(
                child_nonfinite,
                jnp.asarray(QuadStatus.NONFINITE_INTEGRAND, dtype=jnp.int32),
                jnp.where(
                    now_converged,
                    jnp.asarray(QuadStatus.CONVERGED, dtype=jnp.int32),
                    jnp.where(
                        roundoff,
                        jnp.asarray(QuadStatus.ROUNDOFF_LIMITED, dtype=jnp.int32),
                        running,
                    ),
                ),
            )
            return _ControllerState(
                values=new_values,
                errors=new_errors,
                priorities=new_priorities,
                lower=new_lower,
                upper=new_upper,
                active=new_active,
                global_value=new_global_value,
                global_error=new_global_error,
                tolerance=new_tolerance,
                status=new_status,
                evaluations=operand.evaluations + 2 * node_cost,
                refinements=new_refinements,
                active_regions=operand.active_regions + 1,
                no_improvement_count=no_improvement_count,
                growth_count=growth_count,
            )

        return jax.lax.cond(can_split, split, stop_without_split, current)

    final = jax.lax.while_loop(condition, body, state)
    return AdaptiveControllerResult(
        value=final.global_value,
        error=final.global_error,
        tolerance=final.tolerance,
        status=final.status,
        evaluations=final.evaluations,
        refinements=final.refinements,
        active_regions=final.active_regions,
        region_lower=final.lower,
        region_upper=final.upper,
        region_active=final.active,
        no_improvement_count=final.no_improvement_count,
        growth_count=final.growth_count,
    )


__all__ = [
    "LocalEstimate",
    "adaptive_controller",
    "infer_payload_zero",
    "reference_partition",
    "transformed_integrand",
]
