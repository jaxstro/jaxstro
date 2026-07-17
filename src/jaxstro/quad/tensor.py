"""Fixed tensor-product quadrature on finite hyperrectangles."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ._multidim import evaluate_multidim, infer_multidim_payload_zero
from ._tensor import tensor_point_count, tensor_rule_data
from .domains import Hyperrectangle, hyperrectangle_is_valid
from .measures import LebesgueMeasure
from .result import (
    QuadResult,
    QuadStatus,
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


__all__ = ["TensorProduct", "integrate_tensor"]
