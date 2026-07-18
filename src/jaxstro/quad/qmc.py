"""Deterministic and randomized quasi-Monte-Carlo integration."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxstro.numerics.checks import try_concrete_bool

from ._multidim import evaluate_multidim, infer_multidim_payload_zero
from ._scramble import DigitalShift, LinearMatrixScramble, OwenScramble
from ._sobol import resolve_sobol_bits, sobol_points
from .domains import Hyperrectangle, hyperrectangle_is_valid
from .measures import LebesgueMeasure
from .result import QuadResult, QuadStatus, zero_volume_result
from .result import unavailable_result as _unavailable_result
from .tolerance import ErrorNorm


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Sobol:
    """One deterministic power-of-two Sobol prefix."""

    level: int
    bits: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.level, bool)
            or not isinstance(self.level, int)
            or self.level < 0
        ):
            raise ValueError("Sobol level must be a nonnegative integer")
        if self.bits is not None and (
            isinstance(self.bits, bool)
            or not isinstance(self.bits, int)
            or self.bits < 1
        ):
            raise ValueError("Sobol bits must be a positive integer")
        if self.bits is not None and self.level > self.bits:
            raise ValueError("Sobol requires 0 <= level <= bits")

    def tree_flatten(self):
        return (), (self.level, self.bits)

    @classmethod
    def tree_unflatten(cls, metadata, _children):
        level, bits = metadata
        return cls(level=level, bits=bits)


def _validate_evaluation_budget(level: int, max_evaluations: int) -> int:
    if (
        isinstance(max_evaluations, bool)
        or not isinstance(max_evaluations, int)
        or max_evaluations <= 0
    ):
        raise ValueError("max_evaluations must be a positive integer")
    required = 1 << level
    if required > jnp.iinfo(jnp.int32).max:
        raise ValueError(
            f"Sobol level {level} requires {required} evaluations, "
            "exceeding the int32 work-accounting limit"
        )
    if required > max_evaluations:
        raise ValueError(
            f"Sobol level {level} requires {required} evaluations, "
            f"exceeding max_evaluations={max_evaluations}"
        )
    return required


def _with_qmc_work(result: QuadResult, *, evaluations: int, level: int) -> QuadResult:
    return result._replace(
        work=result.work._replace(
            evaluations=jnp.asarray(evaluations, dtype=jnp.int32),
            levels=jnp.asarray(level, dtype=jnp.int32),
        )
    )


def integrate_qmc(
    fun,
    domain: Hyperrectangle,
    *,
    args,
    method: Sobol,
    measure,
    epsabs,
    epsrel,
    max_evaluations: int,
    key,
    error_norm: ErrorNorm,
) -> QuadResult:
    """Evaluate one deterministic Sobol prefix with unavailable error evidence."""
    if key is not None:
        raise TypeError("deterministic Sobol integration does not accept a key")
    point_count = _validate_evaluation_budget(method.level, max_evaluations)
    dtype = jnp.result_type(domain.lower, domain.upper, 0.0)
    resolved_bits = resolve_sobol_bits(
        method.level,
        dtype,
        bits=method.bits,
    )
    absolute_tolerance = jnp.asarray(epsabs)
    relative_tolerance = jnp.asarray(epsrel)
    if absolute_tolerance.ndim != 0 or relative_tolerance.ndim != 0:
        raise ValueError("Sobol tolerances must be scalar")
    if jnp.issubdtype(absolute_tolerance.dtype, jnp.complexfloating) or jnp.issubdtype(
        relative_tolerance.dtype,
        jnp.complexfloating,
    ):
        raise TypeError("Sobol tolerances must have a real dtype")
    zero = infer_multidim_payload_zero(
        fun,
        args=args,
        dimension=domain.dimension,
        dtype=dtype,
    )
    value_dtype = jnp.result_type(zero, dtype)
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
        return _unavailable_result(
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
        points = sobol_points(
            method.level,
            domain.dimension,
            dtype,
            bits=resolved_bits,
        )
        evaluated = evaluate_multidim(
            fun,
            domain,
            points,
            args=args,
            measure=selected_measure,
        )
        factor_shape = (point_count,) + (1,) * (evaluated.values.ndim - 1)
        value = jnp.mean(
            evaluated.values * evaluated.weights.reshape(factor_shape),
            axis=0,
        )
        nonfinite = (
            evaluated.nonfinite | ~evaluated.valid | ~jnp.all(jnp.isfinite(value))
        )
        status = jnp.where(
            nonfinite,
            jnp.asarray(QuadStatus.NONFINITE_INTEGRAND, dtype=jnp.int32),
            jnp.asarray(
                QuadStatus.ERROR_ESTIMATE_UNAVAILABLE,
                dtype=jnp.int32,
            ),
        )
        result = _unavailable_result(
            value,
            epsabs=epsabs,
            epsrel=epsrel,
            error_norm=error_norm,
            evaluations=point_count,
            status=status,
        )
        return _with_qmc_work(
            result,
            evaluations=point_count,
            level=method.level,
        )

    invalid_concrete = try_concrete_bool(invalid)
    if invalid_concrete is True:
        return invalid_branch(None)
    zero_concrete = try_concrete_bool(zero_width)
    if zero_concrete is True:
        return zero_branch(None)

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


__all__ = [
    "DigitalShift",
    "LinearMatrixScramble",
    "OwenScramble",
    "Sobol",
    "integrate_qmc",
]
