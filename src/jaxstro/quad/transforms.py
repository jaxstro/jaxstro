"""Reference-domain maps used by quadrature rules."""

from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array

from ._quantity import validate_raw_domain
from .domains import (
    Infinite,
    Interval,
    LeftInfinite,
    RightInfinite,
    improper_scale_is_valid,
    improper_scale_value,
    interval_is_valid,
    interval_orientation,
)


class AffineMapResult(NamedTuple):
    x: Array
    jacobian: Array
    orientation: Array
    valid: Array


class DomainMapResult(NamedTuple):
    x: Array
    jacobian: Array
    orientation: Array
    valid: Array


def map_interval(domain: Interval, reference: Array) -> AffineMapResult:
    validate_raw_domain(domain)
    lower = jnp.asarray(domain.lower)
    upper = jnp.asarray(domain.upper)
    lo = jnp.minimum(lower, upper)
    hi = jnp.maximum(lower, upper)
    half_width = 0.5 * (hi - lo)
    midpoint = 0.5 * (hi + lo)
    reference = jnp.asarray(reference)
    return AffineMapResult(
        x=midpoint + half_width * reference,
        jacobian=half_width,
        orientation=interval_orientation(domain),
        valid=interval_is_valid(domain),
    )


def map_interval_replay(domain: Interval, reference: Array) -> AffineMapResult:
    """Map a finite interval with its signed affine Jacobian."""
    validate_raw_domain(domain)
    lower = jnp.asarray(domain.lower)
    upper = jnp.asarray(domain.upper)
    half_width = 0.5 * (upper - lower)
    midpoint = 0.5 * (upper + lower)
    reference = jnp.asarray(reference)
    return AffineMapResult(
        x=midpoint + half_width * reference,
        jacobian=half_width,
        orientation=jnp.asarray(1.0, dtype=half_width.dtype),
        valid=interval_is_valid(domain),
    )


def map_domain(
    domain: Interval | RightInfinite | LeftInfinite | Infinite,
    reference: Array,
) -> DomainMapResult:
    """Map reference coordinates in ``(-1, 1)`` to a Phase A domain."""
    validate_raw_domain(domain)
    reference = jnp.asarray(reference)
    if isinstance(domain, Interval):
        mapped = map_interval(domain, reference)
        return DomainMapResult(*mapped)
    if isinstance(domain, RightInfinite):
        lower = jnp.asarray(domain.lower)
        ratio = (1.0 + reference) / (1.0 - reference)
        if domain.scale is None:
            x = lower + ratio
            jacobian = 2.0 / (1.0 - reference) ** 2
        else:
            scale = improper_scale_value(domain)
            x = lower + scale * ratio
            jacobian = 2.0 * scale / (1.0 - reference) ** 2
        return DomainMapResult(
            x=x,
            jacobian=jacobian,
            orientation=jnp.asarray(1.0),
            valid=jnp.isfinite(lower) & improper_scale_is_valid(domain),
        )
    if isinstance(domain, LeftInfinite):
        upper = jnp.asarray(domain.upper)
        ratio = (1.0 - reference) / (1.0 + reference)
        if domain.scale is None:
            x = upper - ratio
            jacobian = 2.0 / (1.0 + reference) ** 2
        else:
            scale = improper_scale_value(domain)
            x = upper - scale * ratio
            jacobian = 2.0 * scale / (1.0 + reference) ** 2
        return DomainMapResult(
            x=x,
            jacobian=jacobian,
            orientation=jnp.asarray(1.0),
            valid=jnp.isfinite(upper) & improper_scale_is_valid(domain),
        )
    if isinstance(domain, Infinite):
        denominator = 1.0 - reference**2
        if domain.scale is None:
            x = reference / denominator
            jacobian = (1.0 + reference**2) / denominator**2
        else:
            scale = improper_scale_value(domain)
            x = scale * reference / denominator
            jacobian = scale * (1.0 + reference**2) / denominator**2
        return DomainMapResult(
            x=x,
            jacobian=jacobian,
            orientation=jnp.asarray(1.0),
            valid=improper_scale_is_valid(domain),
        )
    raise TypeError(f"unsupported quadrature domain: {type(domain).__name__}")


def map_domain_replay(
    domain: Interval | RightInfinite | LeftInfinite | Infinite,
    reference: Array,
) -> DomainMapResult:
    """Use signed finite maps while retaining established improper maps."""
    validate_raw_domain(domain)
    if isinstance(domain, Interval):
        return DomainMapResult(*map_interval_replay(domain, reference))
    return map_domain(domain, reference)


__all__ = [
    "AffineMapResult",
    "DomainMapResult",
    "map_domain",
    "map_domain_replay",
    "map_interval",
    "map_interval_replay",
]
