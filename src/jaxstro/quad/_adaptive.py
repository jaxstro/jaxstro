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
    node_weighted_sum,
    payload_dtype,
    validate_node_values,
)
from ._tanh_sinh import _tanh_sinh_lattice_data
from .domains import (
    Infinite,
    Interval,
    LeftInfinite,
    RightInfinite,
    improper_scale_is_valid,
    interval_is_valid,
    sorted_breakpoints,
)
from .measures import LebesgueMeasure, WeightedMeasure
from .methods import AdaptiveClenshawCurtis, AdaptiveTanhSinh
from .result import QuadStatus
from .rules import ClenshawCurtisRule
from .tolerance import ErrorNorm
from .tolerance import error_norm as reduce_error_norm
from .transforms import map_domain_complements

Domain = Interval | RightInfinite | LeftInfinite | Infinite
AdaptiveMeasure = LebesgueMeasure | WeightedMeasure


class ReferencePartition(NamedTuple):
    """Fixed-shape normalized regions and dynamic domain validity.

    ``lower`` and ``upper`` have shape ``(regions, 2)``: each reference bound
    ``t`` is stored as ``(1 + t, 1 - t)``.
    """

    lower: Array
    upper: Array
    segment_id: Array
    valid: Array


class TransformedIntegrand(NamedTuple):
    """Mapped nodes and contribution values on one normalized region."""

    reference: Array
    x: Array
    jacobian: Array
    values: Array
    valid: Array
    nonfinite: Array
    roundoff: Array
    moved: Any = False


class LocalEstimate(NamedTuple):
    """Method-specific value and payload error for one reference region."""

    value: Array
    error: Array
    nonfinite: Array
    roundoff: Any = False


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


class TanhSinhPair(NamedTuple):
    """Adjacent nested active tanh-sinh levels and explicit tail metadata."""

    nodes: Array
    high_indices: Array
    high_weights: Array
    high_density_weights: Array
    low_indices: Array
    low_nodes: Array
    low_weights: Array
    outer_shell: Array
    terminal_index: Array
    dtype_exhausted: Array


class TanhSinhEstimate(NamedTuple):
    """Adjacent-level tanh-sinh estimate with separated error evidence."""

    value: Array
    error: Array
    discretization_error: Array
    summation_error: Array
    tail_error: Array
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
    region_segment_id: Array
    no_improvement_count: Array
    growth_count: Array


class _ControllerState(NamedTuple):
    values: Array
    errors: Array
    priorities: Array
    lower: Array
    upper: Array
    active: Array
    segment_id: Array
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


def nested_rule_estimate_values(
    values: Array, pair: NestedRulePair
) -> NestedRuleEstimate:
    """Reduce one nested pair after a single high-node evaluation."""
    values = validate_node_values(
        values, pair.nodes.shape[0], context="nested quadrature"
    )
    target_dtype = payload_dtype(values.dtype, pair.nodes.dtype)
    values = values.astype(target_dtype)
    high_value = node_weighted_sum(values, pair.high_weights)
    low_value = node_weighted_sum(values[pair.low_indices], pair.low_weights)
    raw_error = jnp.abs(high_value - low_value)
    resabs = node_weighted_sum(jnp.abs(values), pair.high_weights)
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


def tanh_sinh_pair_data(method: AdaptiveTanhSinh, *, dtype=None) -> TanhSinhPair:
    """Construct adjacent compact levels through the A1 active-lattice owner."""
    low = _tanh_sinh_lattice_data(method.initial_level, dtype=dtype)
    high = _tanh_sinh_lattice_data(method.initial_level + 1, dtype=dtype)
    low_indices = high.coarse_to_fine
    outer_shell = (jnp.abs(high.compact_indices) > 2 * low.terminal_index) & (
        jnp.abs(high.compact_indices) < high.terminal_index
    )
    return TanhSinhPair(
        nodes=high.compact_nodes,
        high_indices=high.compact_indices,
        high_weights=high.compact_weights,
        high_density_weights=high.compact_density_weights,
        low_indices=low_indices,
        low_nodes=high.compact_nodes[low_indices],
        low_weights=low.compact_weights,
        outer_shell=outer_shell,
        terminal_index=high.terminal_index,
        dtype_exhausted=high.dtype_exhausted,
    )


def _summation_gamma(node_count: int, dtype) -> Array:
    scaled_epsilon = node_count * jnp.finfo(dtype).eps
    return jnp.asarray(scaled_epsilon / (1.0 - scaled_epsilon), dtype=dtype)


def tanh_sinh_estimate_values(values: Array, pair: TanhSinhPair) -> TanhSinhEstimate:
    """Reduce adjacent active levels and retain discretization, sum, and tail evidence."""
    values = validate_node_values(
        values, pair.nodes.shape[0], context="adaptive tanh-sinh"
    )
    target_dtype = payload_dtype(values.dtype, pair.nodes.dtype)
    values = values.astype(target_dtype)
    high_value = node_weighted_sum(values, pair.high_weights)
    low_values = values[pair.low_indices]
    low_value = node_weighted_sum(low_values, pair.low_weights)
    discretization = jnp.abs(high_value - low_value)
    high_resabs = node_weighted_sum(jnp.abs(values), pair.high_weights)
    low_resabs = node_weighted_sum(jnp.abs(low_values), pair.low_weights)
    summation = (
        _summation_gamma(pair.nodes.shape[0], pair.nodes.dtype) * high_resabs
        + _summation_gamma(pair.low_indices.shape[0], pair.nodes.dtype) * low_resabs
    )
    contribution = values * jnp.reshape(
        pair.high_weights,
        (pair.nodes.shape[0],) + (1,) * (values.ndim - 1),
    )
    shell_mask = jnp.reshape(
        pair.outer_shell,
        (pair.nodes.shape[0],) + (1,) * (values.ndim - 1),
    )
    shell = jnp.sum(jnp.where(shell_mask, jnp.abs(contribution), 0.0), axis=0)
    density_shape = (pair.nodes.shape[0],) + (1,) * (values.ndim - 1)
    density_values = values * jnp.reshape(pair.high_density_weights, density_shape)
    terminal = jnp.abs(density_values[0]) + jnp.abs(density_values[-1])
    tail = shell + terminal
    error = discretization + summation + tail
    nonfinite = ~(
        jnp.all(jnp.isfinite(values))
        & jnp.all(jnp.isfinite(high_value))
        & jnp.all(jnp.isfinite(low_value))
        & jnp.all(jnp.isfinite(discretization))
        & jnp.all(jnp.isfinite(summation))
        & jnp.all(jnp.isfinite(tail))
        & jnp.all(jnp.isfinite(error))
    )
    return TanhSinhEstimate(
        value=high_value,
        error=error,
        discretization_error=discretization,
        summation_error=summation,
        tail_error=tail,
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
        count = len(domain.breakpoints) + 1
        return ReferencePartition(
            lower=jnp.tile(jnp.asarray([0.0, 2.0], dtype=dtype), (count, 1)),
            upper=jnp.tile(jnp.asarray([2.0, 0.0], dtype=dtype), (count, 1)),
            segment_id=jnp.arange(count, dtype=jnp.int32),
            valid=interval_is_valid(domain),
        )

    scale = () if domain.scale is None else (domain.scale,)
    dtype = (
        jnp.result_type(domain.lower, *scale, 0.0)
        if isinstance(domain, RightInfinite)
        else jnp.result_type(domain.upper, *scale, 0.0)
        if isinstance(domain, LeftInfinite)
        else jnp.result_type(*scale, 0.0)
    )
    if isinstance(domain, RightInfinite):
        valid = jnp.isfinite(jnp.asarray(domain.lower)) & improper_scale_is_valid(
            domain
        )
    elif isinstance(domain, LeftInfinite):
        valid = jnp.isfinite(jnp.asarray(domain.upper)) & improper_scale_is_valid(
            domain
        )
    elif isinstance(domain, Infinite):
        valid = improper_scale_is_valid(domain)
    else:
        raise TypeError(f"unsupported quadrature domain: {type(domain).__name__}")
    return ReferencePartition(
        lower=jnp.asarray([[0.0, 2.0]], dtype=dtype),
        upper=jnp.asarray([[2.0, 0.0]], dtype=dtype),
        segment_id=jnp.asarray([0], dtype=jnp.int32),
        valid=valid,
    )


def interval_segment_bounds(domain: Interval) -> tuple[Array, Array]:
    """Return stopped physical bounds for each original interval segment."""
    points = sorted_breakpoints(domain)
    dtype = jnp.result_type(domain.lower, domain.upper, *domain.breakpoints, 0.0)
    lower = jnp.reshape(jnp.asarray(domain.lower, dtype=dtype), (1,))
    upper = jnp.reshape(jnp.asarray(domain.upper, dtype=dtype), (1,))
    return jnp.concatenate((lower, points)), jnp.concatenate((points, upper))


def select_segment(domain: Domain, segment_id: Array) -> Domain:
    """Select one stopped physical segment from a normalized partition."""
    if isinstance(domain, Interval):
        lower, upper = interval_segment_bounds(domain)
        return Interval(lower[segment_id], upper[segment_id])
    return domain


def reference_pair(point, dtype) -> Array:
    """Return ``(1 + t, 1 - t)`` for a reference point given as ``t`` or as a pair.

    Regions are stored by these complements rather than by ``t``: near
    ``t = -1`` the value ``1 + t`` keeps relative precision down to the
    smallest normal number, where ``t`` itself resolves only about ``1e-16``.
    """
    point = jnp.asarray(point, dtype=dtype)
    if point.ndim >= 1 and point.shape[-1] == 2:
        return point
    return jnp.stack((1.0 + point, 1.0 - point), axis=-1)


def transformed_integrand(
    fun: Callable,
    domain: Domain,
    nodes: Array,
    *,
    region_lower=-1.0,
    region_upper=1.0,
    args: Any = (),
    measure: AdaptiveMeasure | None = None,
    open_region: bool = False,
    replay: bool = False,
) -> TransformedIntegrand:
    """Evaluate one local reference region with every map and density applied.

    ``region_lower`` and ``region_upper`` are reference points, each either
    ``t`` or the pair ``(1 + t, 1 - t)``. Node positions are formed in the
    complement nearer to the region, so a region next to a domain end keeps
    its nodes distinct from that end.
    """
    selected_measure: AdaptiveMeasure = (
        LebesgueMeasure() if measure is None else measure
    )
    if not isinstance(selected_measure, (LebesgueMeasure, WeightedMeasure)):
        raise TypeError(
            "adaptive quadrature requires LebesgueMeasure or WeightedMeasure"
        )

    nodes = jnp.asarray(nodes)
    lower_minus, lower_plus = jnp.moveaxis(
        reference_pair(region_lower, nodes.dtype), -1, 0
    )
    upper_minus, upper_plus = jnp.moveaxis(
        reference_pair(region_upper, nodes.dtype), -1, 0
    )
    near_lower = lower_minus + upper_minus <= lower_plus + upper_plus
    half_width = jnp.where(
        near_lower,
        0.5 * (upper_minus - lower_minus),
        0.5 * (lower_plus - upper_plus),
    )
    minus = lower_minus + half_width * (1.0 + nodes)
    plus = upper_plus + half_width * (1.0 - nodes)
    roundoff = jnp.asarray(False)
    if open_region:
        # An open rule must not reach the domain's reference boundary, where the
        # infinite-domain maps are singular. In complements that happens only
        # when a distance underflows to zero, far below the spacing of t.
        tiny = jnp.finfo(nodes.dtype).tiny
        reached = (minus < tiny) | (plus < tiny)
        roundoff = jnp.any(reached)
        minus = jnp.maximum(minus, tiny)
        plus = jnp.maximum(plus, tiny)
    reference = jnp.where(minus <= plus, minus - 1.0, 1.0 - plus)
    mapped = map_domain_complements(domain, minus, plus, replay=replay)
    moved = jnp.zeros(nodes.shape, dtype=bool)
    if isinstance(domain, Interval):
        # x cannot lie closer to a finite endpoint a than its spacing, so a node
        # meant to be interior (|t| < 1) can round onto a, where an integrable
        # singularity is infinite. Such a node moves to the nearest interior
        # float (a real point of the domain) and is reported in ``moved``; the
        # estimator decides whether that limits the region. Rules whose nodes
        # include t = +-1 (Clenshaw-Curtis) keep their endpoint nodes. XLA on
        # CPU flushes subnormals, so next to zero the bound is the smallest
        # normal number rather than nextafter.
        tiny = jnp.finfo(mapped.x.dtype).tiny
        a = jax.lax.stop_gradient(jnp.asarray(domain.lower, dtype=mapped.x.dtype))
        b = jax.lax.stop_gradient(jnp.asarray(domain.upper, dtype=mapped.x.dtype))
        low = jnp.minimum(a, b)
        high = jnp.maximum(a, b)
        first_inside = jnp.maximum(jnp.nextafter(low, high), low + tiny)
        last_inside = jnp.minimum(jnp.nextafter(high, low), high - tiny)
        x_value = jax.lax.stop_gradient(mapped.x)
        moved = (
            (jnp.abs(nodes) < 1.0)
            & ((x_value < first_inside) | (x_value > last_inside))
            & (first_inside <= last_inside)
        )
        inside = jnp.clip(x_value, first_inside, last_inside)
        mapped = mapped._replace(x=jnp.where(moved, inside, mapped.x))
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
        jnp.isfinite(lower_minus)
        & jnp.isfinite(lower_plus)
        & jnp.isfinite(upper_minus)
        & jnp.isfinite(upper_plus)
        & (lower_minus >= 0.0)
        & (upper_plus >= 0.0)
        & (lower_minus <= upper_minus)
        & (upper_plus <= lower_plus)
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
        roundoff=roundoff,
        moved=moved,
    )


def moved_node_limits_region(
    transformed: TransformedIntegrand,
    weights: Array,
    error: Array,
    *,
    open_region: bool,
) -> Array:
    """Whether nodes moved off a finite endpoint limit this region.

    For Gauss-Kronrod and Clenshaw-Curtis a node reaches an endpoint only in a
    region about one float spacing wide, which cannot be refined: any move
    reports roundoff. Tanh-sinh tail nodes lie within a spacing of the endpoint
    by design; a move limits the region only when the moved nodes' weighted
    contribution is at least the region's rule error, as at an endpoint
    singularity. For a smooth integrand it is about 1e-16 |f|.
    """
    moved = jnp.asarray(transformed.moved)
    if not open_region:
        return jnp.any(moved)
    shape = (moved.shape[0],) + (1,) * (transformed.values.ndim - 1)
    contribution = jnp.abs(transformed.values) * jnp.reshape(jnp.abs(weights), shape)
    moved_mass = jnp.max(
        jnp.sum(jnp.where(jnp.reshape(moved, shape), contribution, 0.0), axis=0)
    )
    return jnp.any(moved) & (moved_mass >= jnp.max(jnp.abs(error)))


def adaptive_controller(
    partition: ReferencePartition,
    local_estimator: Callable[[Array, Array, Array], LocalEstimate],
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

    initial = jax.vmap(local_estimator)(
        partition.lower,
        partition.upper,
        partition.segment_id,
    )
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
        jnp.zeros((max_regions, 2), dtype=partition.lower.dtype)
        .at[:initial_regions]
        .set(partition.lower)
    )
    upper = (
        jnp.zeros((max_regions, 2), dtype=partition.upper.dtype)
        .at[:initial_regions]
        .set(partition.upper)
    )
    active = jnp.arange(max_regions) < initial_regions
    segment_id = (
        jnp.zeros((max_regions,), dtype=jnp.int32)
        .at[:initial_regions]
        .set(partition.segment_id)
    )
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
    initial_roundoff = jnp.any(initial.roundoff)
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
                jnp.where(
                    initial_roundoff,
                    jnp.asarray(QuadStatus.ROUNDOFF_LIMITED, dtype=jnp.int32),
                    running,
                ),
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
        segment_id=segment_id,
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
        region_segment_id = current.segment_id[selected]
        # Bisect in both complements; the smaller one carries the precision.
        midpoint = 0.5 * (region_lower + region_upper)
        midpoint_collapsed = jnp.all(midpoint == region_lower) | jnp.all(
            midpoint == region_upper
        )
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
            child_segment_id = jnp.stack((region_segment_id, region_segment_id))
            children = jax.vmap(local_estimator)(
                child_lower,
                child_upper,
                child_segment_id,
            )
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
            new_segment_id = operand.segment_id.at[selected].set(region_segment_id)
            new_segment_id = new_segment_id.at[append_index].set(region_segment_id)

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
            child_roundoff = jnp.any(children.roundoff)
            now_converged = new_error_norm <= new_tolerance
            roundoff = (no_improvement_count >= 6) | (growth_count >= 5)
            new_status = jnp.where(
                child_nonfinite,
                jnp.asarray(QuadStatus.NONFINITE_INTEGRAND, dtype=jnp.int32),
                jnp.where(
                    now_converged,
                    jnp.asarray(QuadStatus.CONVERGED, dtype=jnp.int32),
                    jnp.where(
                        child_roundoff,
                        jnp.asarray(QuadStatus.ROUNDOFF_LIMITED, dtype=jnp.int32),
                        jnp.where(
                            roundoff,
                            jnp.asarray(QuadStatus.ROUNDOFF_LIMITED, dtype=jnp.int32),
                            running,
                        ),
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
                segment_id=new_segment_id,
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
        region_segment_id=final.segment_id,
        no_improvement_count=final.no_improvement_count,
        growth_count=final.growth_count,
    )


__all__ = [
    "LocalEstimate",
    "moved_node_limits_region",
    "adaptive_controller",
    "infer_payload_zero",
    "reference_partition",
    "select_segment",
    "transformed_integrand",
]
