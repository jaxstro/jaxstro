"""Reference-domain maps used by quadrature rules."""

from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array

from .domains import Interval, interval_is_valid, interval_orientation


class AffineMapResult(NamedTuple):
    x: Array
    jacobian: Array
    orientation: Array
    valid: Array


def map_interval(domain: Interval, reference: Array) -> AffineMapResult:
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
