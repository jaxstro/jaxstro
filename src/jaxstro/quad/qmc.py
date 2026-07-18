"""Deterministic and randomized quasi-Monte-Carlo integration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp

from jaxstro.numerics.checks import try_concrete_bool
from jaxstro.quantity import Quantity
from jaxstro.quantity.errors import DimensionError

from ._multidim import evaluate_multidim, infer_multidim_payload_zero
from ._qmc_interval import (
    empirical_bernstein_half_width,
    fixed_look_interval,
    spent_alpha,
)
from ._scramble import (
    DigitalShift,
    LinearMatrixScramble,
    OwenScramble,
    scramble_integers,
)
from ._sobol import (
    resolve_sobol_bits,
    sobol_integer_points,
    sobol_points,
)
from .domains import (
    Hyperrectangle,
    hyperrectangle_is_valid,
    hyperrectangle_orientation,
)
from .measures import LebesgueMeasure
from .result import (
    ErrorKind,
    QuadError,
    QuadResult,
    QuadStatus,
    QuadWork,
    zero_volume_result,
)
from .result import unavailable_result as _unavailable_result
from .tolerance import ErrorNorm, tolerance_threshold
from .tolerance import error_norm as reduce_error_norm


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


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ScrambledSobol:
    """One fixed-look interval from independent scrambled Sobol replicates."""

    level: int
    replicates: int = 8
    scramble: DigitalShift | LinearMatrixScramble | OwenScramble = field(
        default_factory=LinearMatrixScramble
    )
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if (
            isinstance(self.level, bool)
            or not isinstance(self.level, int)
            or self.level < 0
        ):
            raise ValueError("ScrambledSobol level must be nonnegative")
        if (
            isinstance(self.replicates, bool)
            or not isinstance(self.replicates, int)
            or self.replicates < 8
        ):
            raise ValueError("ScrambledSobol requires at least 8 replicates")
        if not isinstance(
            self.scramble,
            (DigitalShift, LinearMatrixScramble, OwenScramble),
        ):
            raise TypeError("ScrambledSobol requires a supported Sobol randomization")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie strictly between 0 and 1")

    def tree_flatten(self):
        return (), (
            self.level,
            self.replicates,
            self.scramble,
            self.confidence_level,
        )

    @classmethod
    def tree_unflatten(cls, metadata, _children):
        level, replicates, scramble, confidence_level = metadata
        return cls(
            level=level,
            replicates=replicates,
            scramble=scramble,
            confidence_level=confidence_level,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AdaptiveScrambledSobol:
    """Bounded sequential RQMC under a predeclared inspection schedule."""

    schedule: tuple[tuple[int, int], ...]
    estimate_bounds: tuple[float | Quantity, float | Quantity] | None = None
    integrand_bounds: tuple[float | Quantity, float | Quantity] | None = None
    scramble: DigitalShift | LinearMatrixScramble | OwenScramble = field(
        default_factory=LinearMatrixScramble
    )
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, tuple) or not self.schedule:
            raise ValueError("AdaptiveScrambledSobol schedule cannot be empty")
        if any(not isinstance(row, tuple) or len(row) != 2 for row in self.schedule):
            raise ValueError("schedule rows must be (level, replicate_count) tuples")
        levels, replicates = zip(*self.schedule, strict=True)
        if any(
            isinstance(level, bool) or not isinstance(level, int) or level < 0
            for level in levels
        ):
            raise ValueError("schedule levels must be nonnegative integers")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 8
            for count in replicates
        ):
            raise ValueError("schedule requires at least 8 integer replicates")
        pairs = zip(self.schedule, self.schedule[1:])
        if any(
            next_level < level
            or next_count < count
            or (next_level == level and next_count == count)
            for (level, count), (next_level, next_count) in pairs
        ):
            raise ValueError("schedule must be monotone with strict progress")
        if replicates[-1] <= replicates[0]:
            raise ValueError("schedule must include replicate growth")
        if levels[-1] <= levels[0]:
            raise ValueError("schedule must include point-level growth")
        if (self.estimate_bounds is None) == (self.integrand_bounds is None):
            raise ValueError(
                "provide exactly one of estimate_bounds or integrand_bounds"
            )
        selected_bounds = (
            self.estimate_bounds
            if self.estimate_bounds is not None
            else self.integrand_bounds
        )
        if not isinstance(selected_bounds, tuple) or len(selected_bounds) != 2:
            raise ValueError("bounds must be a (lower, upper) tuple")
        lower, upper = selected_bounds
        if isinstance(lower, Quantity) or isinstance(upper, Quantity):
            if not isinstance(lower, Quantity) or not isinstance(upper, Quantity):
                raise TypeError("quantity bounds must both be Quantity values")
            if not lower.unit.is_compatible_with(upper.unit):
                raise DimensionError(
                    "Quantity bounds must have compatible units.",
                    operation="quad-rqmc-bounds",
                    expected=lower.unit.dimensions,
                    actual=upper.unit.dimensions,
                )
            lower_value = lower.value
            upper_value = upper.to_value(lower.unit)
        else:
            lower_value = lower
            upper_value = upper
        if (
            not isinstance(lower_value, int | float)
            or isinstance(lower_value, bool)
            or not isinstance(upper_value, int | float)
            or isinstance(upper_value, bool)
            or not math.isfinite(lower_value)
            or not math.isfinite(upper_value)
            or lower_value > upper_value
        ):
            raise ValueError("bounds must be finite and ordered")
        if not isinstance(
            self.scramble,
            (DigitalShift, LinearMatrixScramble, OwenScramble),
        ):
            raise TypeError(
                "AdaptiveScrambledSobol requires a supported Sobol randomization"
            )
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie strictly between 0 and 1")

    def tree_flatten(self):
        return (), (
            self.schedule,
            self.estimate_bounds,
            self.integrand_bounds,
            self.scramble,
            self.confidence_level,
        )

    @classmethod
    def tree_unflatten(cls, metadata, _children):
        schedule, estimate_bounds, integrand_bounds, scramble, confidence_level = (
            metadata
        )
        return cls(
            schedule=schedule,
            estimate_bounds=estimate_bounds,
            integrand_bounds=integrand_bounds,
            scramble=scramble,
            confidence_level=confidence_level,
        )


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


def _validate_replicated_budget(
    level: int,
    replicates: int,
    max_evaluations: int,
) -> tuple[int, int]:
    point_count = _validate_evaluation_budget(level, max_evaluations)
    total = replicates * point_count
    if total > jnp.iinfo(jnp.int32).max:
        raise ValueError(
            f"ScrambledSobol requires {total} evaluations, "
            "exceeding the int32 work-accounting limit"
        )
    if total > max_evaluations:
        raise ValueError(
            f"ScrambledSobol requires {total} evaluations, "
            f"exceeding max_evaluations={max_evaluations}"
        )
    return point_count, total


def _fixed_qmc_result(
    value,
    half_width,
    *,
    tolerance,
    confidence_level,
    status,
    evaluations: int,
    level: int,
    replicates: int,
    error_norm: ErrorNorm,
    refinements=0,
) -> QuadResult:
    value = jnp.asarray(value)
    half_width = jnp.asarray(half_width)
    half_width_norm = reduce_error_norm(half_width, error_norm)
    status = jnp.asarray(status, dtype=jnp.int32)
    failed = (status == QuadStatus.INVALID_INPUT) | (
        status == QuadStatus.NONFINITE_INTEGRAND
    )
    value = jnp.where(failed, jnp.full_like(value, jnp.nan), value)
    half_width = jnp.where(
        failed,
        jnp.full_like(half_width, jnp.nan),
        half_width,
    )
    half_width_norm = jnp.where(
        failed,
        jnp.asarray(jnp.nan, dtype=half_width_norm.dtype),
        half_width_norm,
    )
    zero = jnp.asarray(0, dtype=jnp.int32)
    return QuadResult(
        value=value,
        error=QuadError(
            estimate=half_width,
            norm=half_width_norm,
            kind=jnp.asarray(
                ErrorKind.CONFIDENCE_INTERVAL_HALF_WIDTH,
                dtype=jnp.int32,
            ),
            confidence_level=jnp.asarray(
                confidence_level,
                dtype=half_width_norm.dtype,
            ),
        ),
        tolerance=tolerance,
        status=status,
        work=QuadWork(
            evaluations=jnp.asarray(evaluations, dtype=jnp.int32),
            refinements=jnp.asarray(refinements, dtype=jnp.int32),
            active_regions=zero,
            levels=jnp.asarray(level, dtype=jnp.int32),
            replicates=jnp.asarray(replicates, dtype=jnp.int32),
        ),
    )


class _SequentialState(NamedTuple):
    sums: jax.Array
    nonfinite: jax.Array
    value: jax.Array
    half_width: jax.Array
    tolerance: jax.Array
    status: jax.Array
    evaluations: jax.Array
    refinements: jax.Array
    level: jax.Array
    replicates: jax.Array
    done: jax.Array


def _validate_adaptive_budget(
    method: AdaptiveScrambledSobol,
    max_evaluations: int,
) -> tuple[int, int]:
    final_level, final_replicates = method.schedule[-1]
    point_count = _validate_evaluation_budget(final_level, max_evaluations)
    total = final_replicates * point_count
    if total > jnp.iinfo(jnp.int32).max:
        raise ValueError(
            f"AdaptiveScrambledSobol requires {total} evaluations, "
            "exceeding the int32 work-accounting limit"
        )
    if total > max_evaluations:
        raise ValueError(
            f"AdaptiveScrambledSobol requires {total} evaluations, "
            f"exceeding max_evaluations={max_evaluations}"
        )
    return point_count, total


def _resolve_sequential_estimate_bounds(
    method: AdaptiveScrambledSobol,
    domain: Hyperrectangle,
    measure,
    dtype,
) -> tuple[jax.Array, jax.Array]:
    if method.estimate_bounds is not None:
        bounds = jnp.asarray(method.estimate_bounds, dtype=dtype)
        return bounds[0], bounds[1]
    if not isinstance(measure, LebesgueMeasure):
        raise ValueError(
            "integrand_bounds require LebesgueMeasure in Phase B3; "
            "supply direct estimate_bounds for other finite measures"
        )
    absolute_volume = jnp.prod(
        jnp.abs(jnp.asarray(domain.upper) - jnp.asarray(domain.lower))
    )
    scaled = (
        hyperrectangle_orientation(domain)
        * absolute_volume
        * jnp.asarray(method.integrand_bounds, dtype=dtype)
    )
    return jnp.min(scaled), jnp.max(scaled)


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


def integrate_scrambled_qmc(
    fun,
    domain: Hyperrectangle,
    *,
    args,
    method: ScrambledSobol,
    measure,
    epsabs,
    epsrel,
    max_evaluations: int,
    key,
    error_norm: ErrorNorm,
) -> QuadResult:
    """Evaluate independent scrambles and return one fixed-look interval."""
    if key is None:
        raise TypeError("ScrambledSobol requires an explicit JAX key")
    point_count, total_evaluations = _validate_replicated_budget(
        method.level,
        method.replicates,
        max_evaluations,
    )
    dtype = jnp.result_type(domain.lower, domain.upper, 0.0)
    resolved_bits = resolve_sobol_bits(method.level, dtype)
    absolute_tolerance = jnp.asarray(epsabs)
    relative_tolerance = jnp.asarray(epsrel)
    if absolute_tolerance.ndim != 0 or relative_tolerance.ndim != 0:
        raise ValueError("ScrambledSobol tolerances must be scalar")
    if jnp.issubdtype(absolute_tolerance.dtype, jnp.complexfloating) or jnp.issubdtype(
        relative_tolerance.dtype,
        jnp.complexfloating,
    ):
        raise TypeError("ScrambledSobol tolerances must have a real dtype")
    zero = infer_multidim_payload_zero(
        fun,
        args=args,
        dimension=domain.dimension,
        dtype=dtype,
    )
    if jnp.ndim(zero) != 0 or jnp.issubdtype(
        jnp.asarray(zero).dtype,
        jnp.complexfloating,
    ):
        raise ValueError("ScrambledSobol requires a scalar real integrand payload")
    zero = jnp.asarray(zero, dtype=jnp.result_type(zero, dtype))
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
        tolerance = tolerance_threshold(
            zero,
            epsabs=epsabs,
            epsrel=epsrel,
            norm=error_norm,
        )
        return _fixed_qmc_result(
            zero,
            jnp.asarray(jnp.nan, dtype=jnp.real(zero).dtype),
            tolerance=tolerance,
            confidence_level=method.confidence_level,
            status=QuadStatus.INVALID_INPUT,
            evaluations=0,
            level=0,
            replicates=0,
            error_norm=error_norm,
        )

    def zero_branch(_):
        return zero_volume_result(
            zero,
            epsabs=epsabs,
            epsrel=epsrel,
            error_norm=error_norm,
        )

    def evaluate_branch(_):
        integer_points = sobol_integer_points(
            method.level,
            domain.dimension,
            bits=resolved_bits,
        )
        scale = jnp.asarray(2.0**resolved_bits, dtype=dtype)

        def evaluate_replicate(replicate):
            replicate_key = jax.random.fold_in(key, replicate)
            scrambled = scramble_integers(
                integer_points,
                method=method.scramble,
                key=replicate_key,
                bits=resolved_bits,
            )
            points = scrambled.astype(dtype) / scale
            evaluated = evaluate_multidim(
                fun,
                domain,
                points,
                args=args,
                measure=selected_measure,
            )
            estimate = jnp.mean(evaluated.values * evaluated.weights)
            nonfinite = evaluated.nonfinite | ~evaluated.valid | ~jnp.isfinite(estimate)
            return estimate, nonfinite

        estimates, nonfinite = jax.lax.map(
            evaluate_replicate,
            jnp.arange(method.replicates, dtype=jnp.uint32),
        )
        interval = fixed_look_interval(
            estimates,
            confidence_level=method.confidence_level,
        )
        tolerance = tolerance_threshold(
            interval.mean,
            epsabs=epsabs,
            epsrel=epsrel,
            norm=error_norm,
        )
        any_nonfinite = jnp.any(nonfinite)
        invalid_interval = ~jnp.isfinite(interval.critical_value) | (
            ~interval.valid & ~any_nonfinite
        )
        status = jnp.where(
            invalid_interval,
            jnp.asarray(QuadStatus.INVALID_INPUT, dtype=jnp.int32),
            jnp.where(
                any_nonfinite,
                jnp.asarray(QuadStatus.NONFINITE_INTEGRAND, dtype=jnp.int32),
                jnp.where(
                    interval.half_width <= tolerance,
                    jnp.asarray(QuadStatus.CONVERGED, dtype=jnp.int32),
                    jnp.asarray(QuadStatus.MAX_EVALUATIONS, dtype=jnp.int32),
                ),
            ),
        )
        return _fixed_qmc_result(
            interval.mean,
            interval.half_width,
            tolerance=tolerance,
            confidence_level=method.confidence_level,
            status=status,
            evaluations=total_evaluations,
            level=method.level,
            replicates=method.replicates,
            error_norm=error_norm,
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


def integrate_adaptive_scrambled_qmc(
    fun,
    domain: Hyperrectangle,
    *,
    args,
    method: AdaptiveScrambledSobol,
    measure,
    epsabs,
    epsrel,
    max_evaluations: int,
    key,
    error_norm: ErrorNorm,
) -> QuadResult:
    """Run a predeclared bounded RQMC confidence-sequence schedule."""
    if key is None:
        raise TypeError("AdaptiveScrambledSobol requires an explicit JAX key")
    _validate_adaptive_budget(method, max_evaluations)
    final_level, final_replicates = method.schedule[-1]
    dtype = jnp.result_type(domain.lower, domain.upper, 0.0)
    resolved_bits = resolve_sobol_bits(final_level, dtype)
    absolute_tolerance = jnp.asarray(epsabs)
    relative_tolerance = jnp.asarray(epsrel)
    if absolute_tolerance.ndim != 0 or relative_tolerance.ndim != 0:
        raise ValueError("AdaptiveScrambledSobol tolerances must be scalar")
    if jnp.issubdtype(absolute_tolerance.dtype, jnp.complexfloating) or jnp.issubdtype(
        relative_tolerance.dtype,
        jnp.complexfloating,
    ):
        raise TypeError("AdaptiveScrambledSobol tolerances must have a real dtype")
    zero = infer_multidim_payload_zero(
        fun,
        args=args,
        dimension=domain.dimension,
        dtype=dtype,
    )
    if jnp.ndim(zero) != 0 or jnp.issubdtype(
        jnp.asarray(zero).dtype,
        jnp.complexfloating,
    ):
        raise ValueError(
            "AdaptiveScrambledSobol requires a scalar real integrand payload"
        )
    zero = jnp.asarray(zero, dtype=jnp.result_type(zero, dtype))
    tolerance_valid = (
        jnp.isfinite(absolute_tolerance)
        & jnp.isfinite(relative_tolerance)
        & (absolute_tolerance >= 0.0)
        & (relative_tolerance >= 0.0)
    )
    invalid = ~hyperrectangle_is_valid(domain) | ~tolerance_valid
    zero_width = jnp.any(jnp.asarray(domain.lower) == domain.upper)
    selected_measure = LebesgueMeasure() if measure is None else measure
    estimate_lower, estimate_upper = _resolve_sequential_estimate_bounds(
        method,
        domain,
        selected_measure,
        dtype,
    )

    def invalid_branch(_):
        tolerance = tolerance_threshold(
            zero,
            epsabs=epsabs,
            epsrel=epsrel,
            norm=error_norm,
        )
        return _fixed_qmc_result(
            zero,
            jnp.asarray(jnp.nan, dtype=jnp.real(zero).dtype),
            tolerance=tolerance,
            confidence_level=method.confidence_level,
            status=QuadStatus.INVALID_INPUT,
            evaluations=0,
            level=0,
            replicates=0,
            refinements=0,
            error_norm=error_norm,
        )

    def zero_branch(_):
        return zero_volume_result(
            zero,
            epsabs=epsabs,
            epsrel=epsrel,
            error_norm=error_norm,
        )

    def evaluate_branch(_):
        integer_points = sobol_integer_points(
            final_level,
            domain.dimension,
            bits=resolved_bits,
        )
        scale = jnp.asarray(2.0**resolved_bits, dtype=dtype)
        initial_tolerance = tolerance_threshold(
            zero,
            epsabs=epsabs,
            epsrel=epsrel,
            norm=error_norm,
        )
        state = _SequentialState(
            sums=jnp.zeros((final_replicates,), dtype=zero.dtype),
            nonfinite=jnp.zeros((final_replicates,), dtype=jnp.bool_),
            value=zero,
            half_width=jnp.asarray(jnp.nan, dtype=jnp.real(zero).dtype),
            tolerance=initial_tolerance,
            status=jnp.asarray(QuadStatus.MAX_EVALUATIONS, dtype=jnp.int32),
            evaluations=jnp.asarray(0, dtype=jnp.int32),
            refinements=jnp.asarray(0, dtype=jnp.int32),
            level=jnp.asarray(0, dtype=jnp.int32),
            replicates=jnp.asarray(0, dtype=jnp.int32),
            done=jnp.asarray(False),
        )
        previous_level = 0
        previous_replicates = 0
        overall_alpha = jnp.asarray(
            1.0 - method.confidence_level,
            dtype=jnp.real(zero).dtype,
        )

        for inspection, (level, replicate_count) in enumerate(method.schedule):
            endpoint = 1 << level
            previous_endpoint = 0 if inspection == 0 else 1 << previous_level
            existing_points = integer_points[previous_endpoint:endpoint]
            new_points = integer_points[:endpoint]
            work_increment = (
                previous_replicates * (endpoint - previous_endpoint)
                + (replicate_count - previous_replicates) * endpoint
            )
            is_last = inspection == len(method.schedule) - 1

            def inspect(current: _SequentialState) -> _SequentialState:
                def evaluate_points(replicate, selected_points):
                    replicate_key = jax.random.fold_in(key, replicate)
                    scrambled = scramble_integers(
                        selected_points,
                        method=method.scramble,
                        key=replicate_key,
                        bits=resolved_bits,
                    )
                    points = scrambled.astype(dtype) / scale
                    evaluated = evaluate_multidim(
                        fun,
                        domain,
                        points,
                        args=args,
                        measure=selected_measure,
                    )
                    values = evaluated.values * evaluated.weights
                    subtotal = jnp.sum(values)
                    nonfinite = (
                        evaluated.nonfinite | ~evaluated.valid | ~jnp.isfinite(subtotal)
                    )
                    return subtotal, nonfinite

                def evaluate_replicate(replicate):
                    def inactive(_):
                        return (
                            jnp.asarray(0, dtype=zero.dtype),
                            jnp.asarray(False),
                        )

                    def evaluate_existing(_):
                        if endpoint == previous_endpoint:
                            return inactive(None)
                        return evaluate_points(
                            replicate,
                            existing_points,
                        )

                    def active(_):
                        return jax.lax.cond(
                            replicate < previous_replicates,
                            evaluate_existing,
                            lambda _: evaluate_points(replicate, new_points),
                            operand=None,
                        )

                    return jax.lax.cond(
                        replicate < replicate_count,
                        active,
                        inactive,
                        operand=None,
                    )

                additions, new_nonfinite = jax.lax.map(
                    evaluate_replicate,
                    jnp.arange(final_replicates, dtype=jnp.uint32),
                )
                sums = current.sums + additions
                nonfinite = current.nonfinite | new_nonfinite
                estimates = sums[:replicate_count] / endpoint
                inspection_alpha = spent_alpha(overall_alpha, inspection)
                half_width = empirical_bernstein_half_width(
                    estimates,
                    lower=estimate_lower,
                    upper=estimate_upper,
                    alpha=inspection_alpha,
                )
                value = jnp.mean(estimates)
                tolerance = tolerance_threshold(
                    value,
                    epsabs=epsabs,
                    epsrel=epsrel,
                    norm=error_norm,
                )
                active_nonfinite = jnp.any(nonfinite[:replicate_count])
                outside_bounds = jnp.any(
                    (estimates < estimate_lower) | (estimates > estimate_upper)
                )
                inspection_invalid = (
                    active_nonfinite | outside_bounds | ~jnp.isfinite(half_width)
                )
                converged = ~inspection_invalid & (half_width <= tolerance)
                status = jnp.where(
                    inspection_invalid,
                    jnp.asarray(QuadStatus.INVALID_INPUT, dtype=jnp.int32),
                    jnp.where(
                        converged,
                        jnp.asarray(QuadStatus.CONVERGED, dtype=jnp.int32),
                        jnp.asarray(
                            QuadStatus.MAX_EVALUATIONS,
                            dtype=jnp.int32,
                        ),
                    ),
                )
                return _SequentialState(
                    sums=sums,
                    nonfinite=nonfinite,
                    value=value,
                    half_width=half_width,
                    tolerance=tolerance,
                    status=status,
                    evaluations=current.evaluations + work_increment,
                    refinements=current.refinements
                    + jnp.asarray(inspection > 0, dtype=jnp.int32),
                    level=jnp.asarray(level, dtype=jnp.int32),
                    replicates=jnp.asarray(replicate_count, dtype=jnp.int32),
                    done=inspection_invalid | converged | is_last,
                )

            state = jax.lax.cond(
                state.done,
                lambda current: current,
                inspect,
                state,
            )
            previous_level = level
            previous_replicates = replicate_count

        return _fixed_qmc_result(
            state.value,
            state.half_width,
            tolerance=state.tolerance,
            confidence_level=method.confidence_level,
            status=state.status,
            evaluations=state.evaluations,
            level=state.level,
            replicates=state.replicates,
            refinements=state.refinements,
            error_norm=error_norm,
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
    "AdaptiveScrambledSobol",
    "DigitalShift",
    "LinearMatrixScramble",
    "OwenScramble",
    "ScrambledSobol",
    "Sobol",
    "integrate_qmc",
    "integrate_adaptive_scrambled_qmc",
    "integrate_scrambled_qmc",
]
