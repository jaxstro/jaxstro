"""Fixed tensor-product quadrature on finite hyperrectangles."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ._multidim import evaluate_multidim, infer_multidim_payload_zero
from ._tensor import (
    adaptive_tensor_controller,
    tensor_point_count,
    tensor_rule_data,
    validate_adaptive_tensor_capacity,
)
from .domains import Hyperrectangle, hyperrectangle_is_valid
from .measures import LebesgueMeasure
from .result import (
    ErrorKind,
    QuadError,
    QuadResult,
    QuadStatus,
    QuadWork,
    unavailable_result,
    zero_volume_result,
)
from .rules import (
    ClenshawCurtisRule,
    FejerIIRule,
    FejerIRule,
    GaussianRule,
    TanhSinhRule,
)
from .tolerance import ErrorNorm
from .tolerance import error_norm as reduce_error_norm

Rule = GaussianRule | ClenshawCurtisRule | FejerIRule | FejerIIRule | TanhSinhRule


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class TensorProduct:
    rules: Rule | tuple[Rule, ...]

    def tree_flatten(self):
        return (), self.rules

    @classmethod
    def tree_unflatten(cls, rules, _children):
        return cls(rules)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AdaptiveTensorClenshawCurtis:
    """Anisotropic p-adaptive tensor Clenshaw-Curtis declaration."""

    initial_level: int = 2

    def __post_init__(self) -> None:
        if (
            isinstance(self.initial_level, bool)
            or not isinstance(self.initial_level, int)
            or self.initial_level < 2
        ):
            raise ValueError("initial_level must be an integer at least 2")

    def tree_flatten(self):
        return (), self.initial_level

    @classmethod
    def tree_unflatten(cls, initial_level, _children):
        return cls(initial_level=initial_level)


def integrate_tensor(
    fun,
    domain: Hyperrectangle,
    *,
    args,
    method: TensorProduct,
    measure,
    epsabs,
    epsrel,
    max_evaluations: int,
    error_norm: ErrorNorm,
) -> QuadResult:
    """Evaluate one fixed tensor formula with unavailable error evidence."""
    dtype = jnp.result_type(domain.lower, domain.upper, 0.0)
    point_count = tensor_point_count(method, domain.dimension, dtype)
    if point_count > max_evaluations:
        raise ValueError(
            f"TensorProduct requires {point_count} evaluations, "
            f"exceeding max_evaluations={max_evaluations}"
        )
    data = tensor_rule_data(method, domain.dimension, dtype)
    zero = infer_multidim_payload_zero(
        fun,
        args=args,
        dimension=domain.dimension,
        dtype=data.points.dtype,
    )
    value_dtype = jnp.result_type(zero, data.points)
    zero = jnp.asarray(zero, dtype=value_dtype)

    def invalid_branch(_):
        return unavailable_result(
            zero,
            epsabs=epsabs,
            epsrel=epsrel,
            error_norm=error_norm,
            evaluations=0,
            status=QuadStatus.INVALID_INPUT,
        )

    def zero_branch(_):
        return zero_volume_result(
            zero,
            epsabs=epsabs,
            epsrel=epsrel,
            error_norm=error_norm,
        )

    def evaluate_branch(_):
        evaluated = evaluate_multidim(
            fun,
            domain,
            data.points,
            args=args,
            measure=LebesgueMeasure() if measure is None else measure,
        )
        factors = data.weights * evaluated.weights
        value = jnp.sum(
            evaluated.values
            * factors.reshape((data.point_count,) + (1,) * (evaluated.values.ndim - 1)),
            axis=0,
        )
        status = jnp.where(
            ~evaluated.valid,
            jnp.asarray(QuadStatus.INVALID_INPUT, dtype=jnp.int32),
            jnp.where(
                evaluated.nonfinite,
                jnp.asarray(QuadStatus.NONFINITE_INTEGRAND, dtype=jnp.int32),
                jnp.asarray(
                    QuadStatus.ERROR_ESTIMATE_UNAVAILABLE,
                    dtype=jnp.int32,
                ),
            ),
        )
        return unavailable_result(
            value,
            epsabs=epsabs,
            epsrel=epsrel,
            error_norm=error_norm,
            evaluations=data.point_count,
            status=status,
        )

    invalid = ~hyperrectangle_is_valid(domain)
    zero_width = jnp.any(jnp.asarray(domain.lower) == domain.upper)
    return jax.lax.cond(
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


def _adaptive_result(
    value,
    error,
    *,
    frontier_error,
    tolerance,
    status,
    evaluations,
    refinements,
    levels,
) -> QuadResult:
    value = jnp.asarray(value)
    error = jnp.asarray(error)
    frontier_error = jnp.asarray(frontier_error)
    status = jnp.asarray(status, dtype=jnp.int32)
    failed = (status == QuadStatus.INVALID_INPUT) | (
        status == QuadStatus.NONFINITE_INTEGRAND
    )
    value = jnp.where(failed, jnp.full_like(value, jnp.nan), value)
    error = jnp.where(failed, jnp.full_like(error, jnp.nan), error)
    error_norm = jnp.where(
        failed,
        jnp.asarray(jnp.nan, dtype=frontier_error.dtype),
        frontier_error,
    )
    zero_work = jnp.asarray(0, dtype=jnp.int32)
    return QuadResult(
        value=value,
        error=QuadError(
            estimate=error,
            norm=error_norm,
            kind=jnp.asarray(ErrorKind.REFINEMENT_DIFFERENCE, dtype=jnp.int32),
            confidence_level=jnp.asarray(jnp.nan, dtype=error_norm.dtype),
        ),
        tolerance=tolerance,
        status=status,
        work=QuadWork(
            evaluations=jnp.asarray(evaluations, dtype=jnp.int32),
            refinements=jnp.asarray(refinements, dtype=jnp.int32),
            active_regions=zero_work,
            levels=jnp.max(jnp.asarray(levels, dtype=jnp.int32)),
            replicates=zero_work,
        ),
    )


def integrate_adaptive_tensor(
    fun,
    domain: Hyperrectangle,
    *,
    args,
    method: AdaptiveTensorClenshawCurtis,
    measure,
    epsabs,
    epsrel,
    max_evaluations: int,
    error_norm: ErrorNorm,
    _return_levels: bool = False,
) -> QuadResult | tuple[QuadResult, jax.Array]:
    """Evaluate one bounded anisotropic tensor frontier controller."""
    dtype = jnp.result_type(domain.lower, domain.upper, 0.0)
    capacity = validate_adaptive_tensor_capacity(
        initial_level=method.initial_level,
        dimension=domain.dimension,
        max_evaluations=max_evaluations,
        dtype=dtype,
    )
    absolute_tolerance = jnp.asarray(epsabs)
    relative_tolerance = jnp.asarray(epsrel)
    if absolute_tolerance.ndim != 0 or relative_tolerance.ndim != 0:
        raise ValueError("adaptive tensor tolerances must be scalar")
    zero = infer_multidim_payload_zero(
        fun,
        args=args,
        dimension=domain.dimension,
        dtype=dtype,
    )
    value_dtype = jnp.result_type(zero, jnp.zeros((), dtype=dtype))
    zero = jnp.asarray(zero, dtype=value_dtype)
    tolerance_valid = (
        jnp.isfinite(absolute_tolerance)
        & jnp.isfinite(relative_tolerance)
        & (absolute_tolerance >= 0.0)
        & (relative_tolerance >= 0.0)
    )
    invalid = ~hyperrectangle_is_valid(domain) | ~tolerance_valid
    zero_width = jnp.any(jnp.asarray(domain.lower) == domain.upper)
    selected_measure = LebesgueMeasure() if measure is None else measure

    def invalid_branch(_):
        error = jnp.full_like(jnp.real(zero), jnp.nan)
        tolerance = jnp.maximum(
            jnp.asarray(epsabs, dtype=dtype),
            jnp.asarray(epsrel, dtype=dtype) * reduce_error_norm(zero, error_norm),
        )
        result = _adaptive_result(
            zero,
            error,
            frontier_error=jnp.asarray(jnp.nan, dtype=dtype),
            tolerance=tolerance,
            status=QuadStatus.INVALID_INPUT,
            evaluations=0,
            refinements=0,
            levels=jnp.full(
                (domain.dimension,),
                method.initial_level,
                dtype=jnp.int32,
            ),
        )
        return result, jnp.full(
            (domain.dimension,),
            method.initial_level,
            dtype=jnp.int32,
        )

    def zero_branch(_):
        return (
            zero_volume_result(
                zero,
                epsabs=epsabs,
                epsrel=epsrel,
                error_norm=error_norm,
            ),
            jnp.full(
                (domain.dimension,),
                method.initial_level,
                dtype=jnp.int32,
            ),
        )

    def evaluate_branch(_):
        controller = adaptive_tensor_controller(
            fun,
            domain,
            args=args,
            measure=selected_measure,
            initial_level=method.initial_level,
            epsabs=epsabs,
            epsrel=epsrel,
            max_evaluations=max_evaluations,
            error_norm=error_norm,
            zero=zero,
            capacity=capacity,
        )
        return (
            _adaptive_result(
                controller.value,
                controller.error,
                frontier_error=controller.frontier_error,
                tolerance=controller.tolerance,
                status=controller.status,
                evaluations=controller.evaluations,
                refinements=controller.refinements,
                levels=controller.levels,
            ),
            controller.evidence.levels,
        )

    result, levels = jax.lax.cond(
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
    if _return_levels:
        return result, levels
    return result


__all__ = [
    "AdaptiveTensorClenshawCurtis",
    "TensorProduct",
    "integrate_adaptive_tensor",
    "integrate_tensor",
]
