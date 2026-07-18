"""Fixed and adaptive Smolyak sparse-grid declarations and evaluators."""

import math
from dataclasses import dataclass
from numbers import Real

import jax
import jax.numpy as jnp

from ._multidim import evaluate_multidim, infer_multidim_payload_zero
from ._sparse import (
    adaptive_sparse_controller,
    fixed_index_set,
    fixed_sparse_node_identities,
    materialize_smolyak_rule,
    required_frontier_capacity,
    smolyak_host_data,
)
from .domains import Hyperrectangle, hyperrectangle_is_valid
from .measures import LebesgueMeasure
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
class Smolyak:
    """Fixed isotropic or statically anisotropic Smolyak declaration."""

    level: int
    anisotropy: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.level, bool)
            or not isinstance(self.level, int)
            or self.level < 1
        ):
            raise ValueError("Smolyak level must be a positive integer")
        if self.anisotropy is not None:
            if not isinstance(self.anisotropy, tuple) or any(
                isinstance(weight, bool)
                or not isinstance(weight, Real)
                or not math.isfinite(weight)
                or weight <= 0.0
                for weight in self.anisotropy
            ):
                raise ValueError(
                    "Smolyak anisotropy weights must be finite and positive"
                )
            object.__setattr__(
                self,
                "anisotropy",
                tuple(float(weight) for weight in self.anisotropy),
            )

    def tree_flatten(self):
        return (), (self.level, self.anisotropy)

    @classmethod
    def tree_unflatten(cls, metadata, _children):
        level, anisotropy = metadata
        return cls(level=level, anisotropy=anisotropy)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AdaptiveSmolyak:
    """Dimension-adaptive Smolyak declaration."""

    initial_level: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.initial_level, bool)
            or not isinstance(self.initial_level, int)
            or self.initial_level < 1
        ):
            raise ValueError("initial_level must be a positive integer")

    def tree_flatten(self):
        return (), self.initial_level

    @classmethod
    def tree_unflatten(cls, initial_level, _children):
        return cls(initial_level=initial_level)


def _validate_positive_capacity(name: str, value: int | None) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_fixed_capacity(
    *,
    point_count: int,
    index_count: int,
    frontier_count: int,
    max_evaluations: int,
    max_indices: int | None,
    max_frontier: int | None,
    max_nodes: int | None,
) -> None:
    max_evaluations = _validate_positive_capacity(
        "max_evaluations",
        max_evaluations,
    )
    max_indices = _validate_positive_capacity("max_indices", max_indices)
    max_frontier = _validate_positive_capacity("max_frontier", max_frontier)
    max_nodes = _validate_positive_capacity("max_nodes", max_nodes)
    if index_count > max_indices:
        raise ValueError(
            f"fixed Smolyak requires {index_count} indices, "
            f"exceeding max_indices={max_indices}"
        )
    if frontier_count > max_frontier:
        raise ValueError(
            f"fixed Smolyak requires {frontier_count} frontier rows, "
            f"exceeding max_frontier={max_frontier}"
        )
    if point_count > max_nodes:
        raise ValueError(
            f"fixed Smolyak requires {point_count} unique nodes, "
            f"exceeding max_nodes={max_nodes}"
        )
    if point_count > max_evaluations:
        raise ValueError(
            f"fixed Smolyak requires {point_count} evaluations, "
            f"exceeding max_evaluations={max_evaluations}"
        )


def _sparse_result(
    value,
    error,
    *,
    frontier_error,
    tolerance,
    status,
    evaluations,
    refinements,
    level,
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
    frontier_error = jnp.where(
        failed,
        jnp.asarray(jnp.nan, dtype=frontier_error.dtype),
        frontier_error,
    )
    zero = jnp.asarray(0, dtype=jnp.int32)
    return QuadResult(
        value=value,
        error=QuadError(
            estimate=error,
            norm=frontier_error,
            kind=jnp.asarray(ErrorKind.SPARSE_GRID_SURPLUS, dtype=jnp.int32),
            confidence_level=jnp.asarray(jnp.nan, dtype=frontier_error.dtype),
        ),
        tolerance=tolerance,
        status=status,
        work=QuadWork(
            evaluations=jnp.asarray(evaluations, dtype=jnp.int32),
            refinements=jnp.asarray(refinements, dtype=jnp.int32),
            active_regions=zero,
            levels=jnp.asarray(level, dtype=jnp.int32),
            replicates=zero,
        ),
    )


def integrate_sparse(
    fun,
    domain: Hyperrectangle,
    *,
    args,
    method: Smolyak,
    measure,
    epsabs,
    epsrel,
    max_evaluations: int,
    max_indices: int | None,
    max_frontier: int | None,
    max_nodes: int | None,
    error_norm: ErrorNorm,
) -> QuadResult:
    """Evaluate one exact-node-coalesced fixed Smolyak formula."""
    max_evaluations = _validate_positive_capacity(
        "max_evaluations",
        max_evaluations,
    )
    max_indices = _validate_positive_capacity("max_indices", max_indices)
    max_frontier = _validate_positive_capacity("max_frontier", max_frontier)
    max_nodes = _validate_positive_capacity("max_nodes", max_nodes)
    indices = fixed_index_set(method, domain.dimension)
    accepted = set(indices)
    frontier_count = len(
        {
            index
            for index in indices
            if not any(
                index[:axis] + (index[axis] + 1,) + index[axis + 1 :] in accepted
                for axis in range(domain.dimension)
            )
        }
    )
    if len(indices) > max_indices:
        raise ValueError(
            f"fixed Smolyak requires {len(indices)} indices, "
            f"exceeding max_indices={max_indices}"
        )
    if frontier_count > max_frontier:
        raise ValueError(
            f"fixed Smolyak requires {frontier_count} frontier rows, "
            f"exceeding max_frontier={max_frontier}"
        )
    node_limit = min(max_nodes, max_evaluations)
    limit_name = "max_nodes" if max_nodes <= max_evaluations else "max_evaluations"
    fixed_sparse_node_identities(
        indices,
        node_limit=node_limit,
        limit_name=limit_name,
    )
    dtype = jnp.result_type(domain.lower, domain.upper, 0.0)
    host = smolyak_host_data(method, domain.dimension, dtype)
    _validate_fixed_capacity(
        point_count=host.points.shape[0],
        index_count=len(host.indices),
        frontier_count=sum(bool(active) for active in host.frontier_mask),
        max_evaluations=max_evaluations,
        max_indices=max_indices,
        max_frontier=max_frontier,
        max_nodes=max_nodes,
    )
    data = materialize_smolyak_rule(host)
    absolute_tolerance = jnp.asarray(epsabs)
    relative_tolerance = jnp.asarray(epsrel)
    if absolute_tolerance.ndim != 0 or relative_tolerance.ndim != 0:
        raise ValueError("fixed Smolyak tolerances must be scalar")
    if jnp.issubdtype(absolute_tolerance.dtype, jnp.complexfloating) or jnp.issubdtype(
        relative_tolerance.dtype,
        jnp.complexfloating,
    ):
        raise TypeError("fixed Smolyak tolerances must have a real dtype")
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
    selected_measure = LebesgueMeasure() if measure is None else measure

    def invalid_branch(_):
        error = jnp.full_like(jnp.real(zero), jnp.nan)
        frontier_error = reduce_error_norm(error, error_norm)
        tolerance = tolerance_threshold(
            zero,
            epsabs=epsabs,
            epsrel=epsrel,
            norm=error_norm,
        )
        return _sparse_result(
            zero,
            error,
            frontier_error=frontier_error,
            tolerance=tolerance,
            status=QuadStatus.INVALID_INPUT,
            evaluations=0,
            refinements=0,
            level=method.level,
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
            measure=selected_measure,
        )
        factor_shape = (data.point_count,) + (1,) * (evaluated.values.ndim - 1)
        contributions = evaluated.values * evaluated.weights.reshape(factor_shape)
        increments = jnp.tensordot(
            data.increment_weights,
            contributions,
            axes=((1,), (0,)),
        )
        value = jnp.sum(increments, axis=0)
        frontier_shape = (data.index_count,) + (1,) * (increments.ndim - 1)
        frontier_increments = jnp.where(
            data.frontier_mask.reshape(frontier_shape),
            increments,
            jnp.zeros_like(increments),
        )
        error = jnp.sum(jnp.abs(frontier_increments), axis=0)
        increment_norms = jax.vmap(
            lambda increment: reduce_error_norm(increment, error_norm)
        )(frontier_increments)
        frontier_error = jnp.sum(jnp.where(data.frontier_mask, increment_norms, 0.0))
        tolerance = tolerance_threshold(
            value,
            epsabs=epsabs,
            epsrel=epsrel,
            norm=error_norm,
        )
        nonfinite = (
            evaluated.nonfinite
            | ~evaluated.valid
            | ~jnp.all(jnp.isfinite(increments))
            | ~jnp.all(jnp.isfinite(error))
            | ~jnp.isfinite(frontier_error)
            | ~jnp.isfinite(tolerance)
        )
        converged = frontier_error <= tolerance
        status = jnp.where(
            nonfinite,
            jnp.asarray(QuadStatus.NONFINITE_INTEGRAND, dtype=jnp.int32),
            jnp.where(
                converged,
                jnp.asarray(QuadStatus.CONVERGED, dtype=jnp.int32),
                jnp.asarray(QuadStatus.MAX_INDICES, dtype=jnp.int32),
            ),
        )
        return _sparse_result(
            value,
            error,
            frontier_error=frontier_error,
            tolerance=tolerance,
            status=status,
            evaluations=data.point_count,
            refinements=data.index_count - 1,
            level=method.level,
        )

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


def integrate_adaptive_sparse(
    fun,
    domain: Hyperrectangle,
    *,
    args,
    method: AdaptiveSmolyak,
    measure,
    epsabs,
    epsrel,
    max_evaluations: int,
    max_indices: int | None,
    max_frontier: int | None,
    max_nodes: int | None,
    error_norm: ErrorNorm,
) -> QuadResult:
    """Evaluate one fixed-capacity dimension-adaptive Smolyak formula."""
    max_evaluations = _validate_positive_capacity(
        "max_evaluations",
        max_evaluations,
    )
    max_indices = _validate_positive_capacity("max_indices", max_indices)
    max_frontier = _validate_positive_capacity("max_frontier", max_frontier)
    max_nodes = _validate_positive_capacity("max_nodes", max_nodes)
    required_frontier = required_frontier_capacity(
        domain.dimension,
        max_indices,
    )
    if max_frontier < required_frontier:
        raise ValueError(
            f"max_frontier must be at least {required_frontier} "
            "for the declared dimension and max_indices"
        )
    initial_indices = fixed_index_set(
        Smolyak(level=method.initial_level),
        domain.dimension,
    )
    if len(initial_indices) > max_indices:
        raise ValueError(
            f"initial adaptive Smolyak set requires {len(initial_indices)} indices, "
            f"exceeding max_indices={max_indices}"
        )
    absolute_tolerance = jnp.asarray(epsabs)
    relative_tolerance = jnp.asarray(epsrel)
    if absolute_tolerance.ndim != 0 or relative_tolerance.ndim != 0:
        raise ValueError("adaptive Smolyak tolerances must be scalar")
    if jnp.issubdtype(absolute_tolerance.dtype, jnp.complexfloating) or jnp.issubdtype(
        relative_tolerance.dtype,
        jnp.complexfloating,
    ):
        raise TypeError("adaptive Smolyak tolerances must have a real dtype")
    dtype = jnp.result_type(domain.lower, domain.upper, 0.0)
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
        frontier_error = reduce_error_norm(error, error_norm)
        tolerance = tolerance_threshold(
            zero,
            epsabs=epsabs,
            epsrel=epsrel,
            norm=error_norm,
        )
        return _sparse_result(
            zero,
            error,
            frontier_error=frontier_error,
            tolerance=tolerance,
            status=QuadStatus.INVALID_INPUT,
            evaluations=0,
            refinements=0,
            level=method.initial_level,
        )

    def zero_branch(_):
        return zero_volume_result(
            zero,
            epsabs=epsabs,
            epsrel=epsrel,
            error_norm=error_norm,
        )

    def evaluate_branch(_):
        controller = adaptive_sparse_controller(
            fun,
            domain,
            args=args,
            measure=selected_measure,
            initial_indices=initial_indices,
            epsabs=epsabs,
            epsrel=epsrel,
            max_evaluations=max_evaluations,
            max_indices=max_indices,
            max_frontier=max_frontier,
            max_nodes=max_nodes,
            error_norm=error_norm,
            zero=zero,
        )
        return _sparse_result(
            controller.value,
            controller.error,
            frontier_error=controller.frontier_error,
            tolerance=controller.tolerance,
            status=controller.status,
            evaluations=controller.evaluations,
            refinements=controller.refinements,
            level=controller.level,
        )

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
    "AdaptiveSmolyak",
    "Smolyak",
    "integrate_adaptive_sparse",
    "integrate_sparse",
]
