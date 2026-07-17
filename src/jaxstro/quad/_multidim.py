"""Coordinate-last mapping and raw evaluation for finite hyperrectangles."""

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ._integrand import call_integrand, has_explicit_args
from .domains import (
    Hyperrectangle,
    hyperrectangle_is_valid,
    hyperrectangle_orientation,
)
from .measures import LebesgueMeasure, WeightedMeasure


class MultidimMapResult(NamedTuple):
    x: Array
    jacobian: Array
    orientation: Array
    valid: Array


class PointEvaluation(NamedTuple):
    values: Array
    weights: Array
    nonfinite: Array
    valid: Array


def infer_multidim_payload_zero(fun, *, args, dimension: int, dtype):
    point = jax.ShapeDtypeStruct((1, dimension), dtype)
    abstract = jax.eval_shape(
        lambda x: call_integrand(fun, x, args, has_explicit_args(args)),
        point,
    )
    if not hasattr(abstract, "shape") or abstract.shape[:1] != (1,):
        raise ValueError(
            "multidimensional integrand output must have a leading point axis"
        )
    return jnp.zeros(abstract.shape[1:], dtype=abstract.dtype)


def map_hyperrectangle(
    domain: Hyperrectangle,
    reference: Array,
) -> MultidimMapResult:
    reference = jnp.asarray(reference)
    if reference.ndim != 2 or reference.shape[-1] != domain.dimension:
        raise ValueError("reference points must have shape (point_count, dimension)")
    lower = jnp.asarray(domain.lower)
    width = jnp.asarray(domain.upper) - lower
    return MultidimMapResult(
        x=lower + reference * width,
        jacobian=jnp.prod(jnp.abs(width)),
        orientation=hyperrectangle_orientation(domain),
        valid=hyperrectangle_is_valid(domain),
    )


def _density_values(measure, x: Array, args: Any) -> Array:
    if isinstance(measure, LebesgueMeasure):
        return jnp.ones(x.shape[0], dtype=x.dtype)
    if isinstance(measure, WeightedMeasure):
        density = jnp.asarray(measure.density(x, args))
        if density.shape != x.shape[:-1]:
            raise ValueError("multidimensional density must have shape (point_count,)")
        return density
    raise TypeError("multidimensional integration requires a finite measure")


def evaluate_multidim(
    fun: Callable,
    domain: Hyperrectangle,
    reference: Array,
    *,
    args: Any,
    measure,
) -> PointEvaluation:
    reference = jnp.asarray(reference)
    mapped = map_hyperrectangle(domain, reference)
    values = jnp.asarray(call_integrand(fun, mapped.x, args, has_explicit_args(args)))
    if values.ndim == 0 or values.shape[0] != reference.shape[0]:
        raise ValueError(
            "multidimensional integrand output must have a leading point axis"
        )
    density = _density_values(measure, mapped.x, args)
    weights = mapped.orientation * mapped.jacobian * density
    nonfinite = ~(jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(weights)))
    return PointEvaluation(values, weights, nonfinite, mapped.valid)
