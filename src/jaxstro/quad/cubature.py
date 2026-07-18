"""Public h-adaptive cubature declarations and stopped evaluator."""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from ._cubature import (
    cubature_controller,
    genz_malik_data,
    validate_cubature_capacity,
)
from ._multidim import infer_multidim_payload_zero
from .domains import Hyperrectangle, hyperrectangle_is_valid
from .measures import LebesgueMeasure, WeightedMeasure
from .result import (
    ErrorKind,
    QuadError,
    QuadResult,
    QuadStatus,
    QuadWork,
    zero_volume_result,
)
from .tolerance import ErrorNorm, tolerance_threshold
from .tolerance import error_norm as reduce_error_norm


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GenzMalik:
    """Degree-7 Genz-Malik rule with an embedded degree-5 formula."""

    def tree_flatten(self):
        return (), None

    @classmethod
    def tree_unflatten(cls, _metadata, _children):
        return cls()


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AdaptiveCubature:
    """Fixed-capacity h-adaptive cubature using the Genz-Malik pair.

    Scalar eager and JIT solves physically skip child evaluation after a
    terminal status. ``jax.vmap`` preserves result semantics and per-lane
    logical work, but select-style batching does not guarantee physical
    per-lane skipping. Cost-sensitive heterogeneous batches should apply
    ``jax.lax.map`` around scalar :func:`jaxstro.quad.integrate` calls.
    """

    rule: GenzMalik = field(default_factory=GenzMalik)

    def __post_init__(self) -> None:
        if not isinstance(self.rule, GenzMalik):
            raise TypeError("AdaptiveCubature requires GenzMalik in Phase B1")

    def tree_flatten(self):
        return (), self.rule

    @classmethod
    def tree_unflatten(cls, rule, _children):
        return cls(rule=rule)


def _cubature_result(
    value,
    error,
    *,
    error_value_norm,
    tolerance,
    status,
    evaluations,
    refinements,
    active_regions,
    deepest_depth,
) -> QuadResult:
    value = jnp.asarray(value)
    error = jnp.asarray(error)
    error_value_norm = jnp.asarray(error_value_norm)
    status = jnp.asarray(status, dtype=jnp.int32)
    failed = (status == QuadStatus.INVALID_INPUT) | (
        status == QuadStatus.NONFINITE_INTEGRAND
    )
    value = jnp.where(failed, jnp.full_like(value, jnp.nan), value)
    error = jnp.where(failed, jnp.full_like(error, jnp.nan), error)
    error_value_norm = jnp.where(
        failed,
        jnp.asarray(jnp.nan, dtype=error_value_norm.dtype),
        error_value_norm,
    )
    zero = jnp.asarray(0, dtype=jnp.int32)
    return QuadResult(
        value=value,
        error=QuadError(
            estimate=error,
            norm=error_value_norm,
            kind=jnp.asarray(ErrorKind.EMBEDDED_RULE, dtype=jnp.int32),
            confidence_level=jnp.asarray(jnp.nan, dtype=error_value_norm.dtype),
        ),
        tolerance=tolerance,
        status=status,
        work=QuadWork(
            evaluations=jnp.asarray(evaluations, dtype=jnp.int32),
            refinements=jnp.asarray(refinements, dtype=jnp.int32),
            active_regions=jnp.asarray(active_regions, dtype=jnp.int32),
            levels=jnp.asarray(deepest_depth, dtype=jnp.int32),
            replicates=zero,
        ),
    )


def integrate_cubature(
    fun,
    domain: Hyperrectangle,
    *,
    args,
    method: AdaptiveCubature,
    measure,
    epsabs,
    epsrel,
    max_evaluations: int,
    max_regions: int | None,
    error_norm: ErrorNorm,
    _return_leaves: bool = False,
) -> QuadResult | tuple[QuadResult, tuple[jax.Array, jax.Array, jax.Array]]:
    """Evaluate one bounded h-adaptive Genz-Malik region controller."""
    capacity = validate_cubature_capacity(
        dimension=domain.dimension,
        max_evaluations=max_evaluations,
        max_regions=max_regions,
    )
    absolute_tolerance = jnp.asarray(epsabs)
    relative_tolerance = jnp.asarray(epsrel)
    if absolute_tolerance.ndim != 0 or relative_tolerance.ndim != 0:
        raise ValueError("adaptive cubature tolerances must be scalar")
    if jnp.issubdtype(absolute_tolerance.dtype, jnp.complexfloating) or jnp.issubdtype(
        relative_tolerance.dtype,
        jnp.complexfloating,
    ):
        raise TypeError("adaptive cubature tolerances must have a real dtype")
    selected_measure = LebesgueMeasure() if measure is None else measure
    if not isinstance(selected_measure, (LebesgueMeasure, WeightedMeasure)):
        raise TypeError("adaptive cubature requires LebesgueMeasure or WeightedMeasure")

    dtype = jnp.result_type(domain.lower, domain.upper, 0.0)
    data = genz_malik_data(domain.dimension, dtype)
    zero = infer_multidim_payload_zero(
        fun,
        args=args,
        dimension=domain.dimension,
        dtype=data.points.dtype,
    )
    value_dtype = jnp.result_type(zero, data.points)
    zero = jnp.asarray(zero, dtype=value_dtype)
    tolerance_valid = (
        jnp.isfinite(absolute_tolerance)
        & jnp.isfinite(relative_tolerance)
        & (absolute_tolerance >= 0.0)
        & (relative_tolerance >= 0.0)
    )
    invalid = ~hyperrectangle_is_valid(domain) | ~tolerance_valid
    zero_width = jnp.any(jnp.asarray(domain.lower) == domain.upper)
    empty_lower = jnp.zeros(
        (capacity.store_capacity, domain.dimension),
        dtype=data.points.dtype,
    )
    empty_upper = jnp.zeros_like(empty_lower)
    empty_active = jnp.zeros((capacity.store_capacity,), dtype=jnp.bool_)

    def invalid_branch(_):
        error = jnp.full_like(jnp.real(zero), jnp.nan)
        error_value_norm = reduce_error_norm(error, error_norm)
        tolerance = tolerance_threshold(
            zero,
            epsabs=epsabs,
            epsrel=epsrel,
            norm=error_norm,
        )
        return (
            _cubature_result(
                zero,
                error,
                error_value_norm=error_value_norm,
                tolerance=tolerance,
                status=QuadStatus.INVALID_INPUT,
                evaluations=0,
                refinements=0,
                active_regions=0,
                deepest_depth=0,
            ),
            (empty_lower, empty_upper, empty_active),
        )

    def zero_branch(_):
        return (
            zero_volume_result(
                zero,
                epsabs=epsabs,
                epsrel=epsrel,
                error_norm=error_norm,
            ),
            (empty_lower, empty_upper, empty_active),
        )

    def evaluate_branch(_):
        controller = cubature_controller(
            fun,
            domain,
            args=args,
            measure=selected_measure,
            epsabs=epsabs,
            epsrel=epsrel,
            max_evaluations=max_evaluations,
            max_regions=capacity.store_capacity,
            error_norm=error_norm,
            zero=zero,
            data=data,
            capacity=capacity,
        )
        return (
            _cubature_result(
                controller.value,
                controller.error,
                error_value_norm=controller.error_norm,
                tolerance=controller.tolerance,
                status=controller.status,
                evaluations=controller.evaluations,
                refinements=controller.refinements,
                active_regions=controller.active_regions,
                deepest_depth=controller.deepest_depth,
            ),
            (
                controller.evidence.lower,
                controller.evidence.upper,
                controller.evidence.active,
            ),
        )

    result, leaves = jax.lax.cond(
        invalid,
        invalid_branch,
        lambda _: jax.lax.cond(
            zero_width,
            zero_branch,
            evaluate_branch,
            operand=None,
        ),
        operand=None,
    )
    if _return_leaves:
        return result, leaves
    return result


__all__ = ["AdaptiveCubature", "GenzMalik", "integrate_cubature"]
