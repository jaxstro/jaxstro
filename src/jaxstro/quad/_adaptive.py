"""Shared reference-partition and transformed-integrand substrate."""

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array

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


__all__ = [
    "infer_payload_zero",
    "reference_partition",
    "transformed_integrand",
]
