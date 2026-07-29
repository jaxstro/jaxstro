from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp

from ._cubature import genz_malik_data, validate_cubature_capacity
from ._multidim_replay import (
    MultidimConfig,
    MultidimPrimalSolve,
    ReplayFormula,
    multidim_replay_core,
)
from ._quantity import (
    normalize_multidim_call,
    quantity_mode,
    restore_result,
)
from ._scramble import scramble_integers
from ._sobol import resolve_sobol_bits, sobol_integer_points, sobol_points
from ._sparse import (
    identities_to_points,
    materialize_smolyak_rule,
    smolyak_host_data,
)
from ._tensor import (
    adaptive_tensor_replay_formula,
    adaptive_tensor_tables,
    tensor_rule_data,
    validate_adaptive_tensor_capacity,
)
from .adaptive import integrate as integrate_1d
from .cubature import AdaptiveCubature, integrate_cubature
from .domains import Hyperrectangle
from .measures import LebesgueMeasure
from .qmc import (
    AdaptiveScrambledSobol,
    ScrambledSobol,
    Sobol,
    integrate_adaptive_scrambled_qmc,
    integrate_qmc,
    integrate_scrambled_qmc,
)
from .sparse import (
    AdaptiveSmolyak,
    Smolyak,
    integrate_adaptive_sparse,
    integrate_sparse,
)
from .tensor import (
    AdaptiveTensorClenshawCurtis,
    TensorProduct,
    integrate_adaptive_tensor,
    integrate_tensor,
)
from .tolerance import ErrorNorm, MaxNorm


def _fixed_tensor_formula(method, domain) -> ReplayFormula:
    dtype = jnp.result_type(domain.lower, domain.upper, 0.0)
    data = tensor_rule_data(method, domain.dimension, dtype)
    return ReplayFormula(
        data.points,
        data.weights,
        jnp.ones((data.point_count,), dtype=jnp.bool_),
    )


def _adaptive_tensor_formula(method, domain, levels, max_evaluations) -> ReplayFormula:
    dtype = jnp.result_type(domain.lower, domain.upper, 0.0)
    capacity = validate_adaptive_tensor_capacity(
        initial_level=method.initial_level,
        dimension=domain.dimension,
        max_evaluations=max_evaluations,
        dtype=dtype,
    )
    tables = adaptive_tensor_tables(
        initial_level=method.initial_level,
        max_level=capacity.max_level,
        dtype=dtype,
    )
    return ReplayFormula(
        *adaptive_tensor_replay_formula(levels, tables, max_evaluations)
    )


def _cubature_formula(domain, leaves, max_evaluations, max_regions):
    dtype = jnp.result_type(domain.lower, domain.upper, 0.0)
    capacity = validate_cubature_capacity(
        dimension=domain.dimension,
        max_evaluations=max_evaluations,
        max_regions=max_regions,
    )
    data = genz_malik_data(domain.dimension, dtype)
    lower, upper, active = leaves
    width = upper - lower
    points = lower[:, None, :] + width[:, None, :] * data.points[None, :, :]
    weights = jnp.prod(width, axis=-1)[:, None] * data.high_weights[None, :]
    formula_active = jnp.broadcast_to(
        active[:, None],
        (capacity.store_capacity, data.point_count),
    )
    return ReplayFormula(
        points.reshape((-1, domain.dimension)),
        weights.reshape(-1),
        formula_active.reshape(-1),
    )


def _fixed_sparse_formula(method, domain) -> ReplayFormula:
    dtype = jnp.result_type(domain.lower, domain.upper, 0.0)
    data = materialize_smolyak_rule(smolyak_host_data(method, domain.dimension, dtype))
    return ReplayFormula(
        data.points,
        data.weights,
        jnp.ones((data.point_count,), dtype=jnp.bool_),
    )


def _sparse_identity_points(node_ids, *, dimension: int, dtype):
    identities = node_ids.reshape((node_ids.shape[0], dimension, 2))
    return identities_to_points(identities, dtype)


def _adaptive_sparse_formula(domain, nodes) -> ReplayFormula:
    dtype = jnp.result_type(domain.lower, domain.upper, 0.0)
    node_ids, coefficients, active = nodes
    return ReplayFormula(
        _sparse_identity_points(
            node_ids,
            dimension=domain.dimension,
            dtype=dtype,
        ),
        coefficients,
        active,
    )


def _qmc_formula(method, domain, key, *, result=None) -> ReplayFormula:
    dtype = jnp.result_type(domain.lower, domain.upper, 0.0)
    if isinstance(method, Sobol):
        points = sobol_points(
            method.level,
            domain.dimension,
            dtype,
            bits=method.bits,
        )
        count = points.shape[0]
        return ReplayFormula(
            points,
            jnp.full((count,), 1.0 / count, dtype=dtype),
            jnp.ones((count,), dtype=jnp.bool_),
        )

    if isinstance(method, ScrambledSobol):
        level = method.level
        replicate_capacity = method.replicates
    else:
        level, replicate_capacity = method.schedule[-1]
    resolved_bits = resolve_sobol_bits(level, dtype)
    integer_points = sobol_integer_points(
        level,
        domain.dimension,
        bits=resolved_bits,
    )
    scale = jnp.asarray(2.0**resolved_bits, dtype=dtype)

    def one_replicate(replicate):
        replicate_key = jax.random.fold_in(key, replicate)
        return (
            scramble_integers(
                integer_points,
                method=method.scramble,
                key=replicate_key,
                bits=resolved_bits,
            ).astype(dtype)
            / scale
        )

    points = jax.lax.map(
        one_replicate,
        jnp.arange(replicate_capacity, dtype=jnp.uint32),
    )
    point_capacity = points.shape[1]
    if isinstance(method, AdaptiveScrambledSobol):
        active_level = result.work.levels
        active_replicates = result.work.replicates
        active_points = jnp.left_shift(
            jnp.asarray(1, dtype=jnp.int32),
            active_level,
        )
        active = (jnp.arange(replicate_capacity)[:, None] < active_replicates) & (
            jnp.arange(point_capacity)[None, :] < active_points
        )
        denominator = jnp.asarray(
            active_replicates * active_points,
            dtype=dtype,
        )
    else:
        active = jnp.ones(
            (replicate_capacity, point_capacity),
            dtype=jnp.bool_,
        )
        denominator = jnp.asarray(
            replicate_capacity * point_capacity,
            dtype=dtype,
        )
    return ReplayFormula(
        points.reshape((-1, domain.dimension)),
        jnp.full(
            (replicate_capacity * point_capacity,),
            jnp.asarray(1.0, dtype=dtype) / denominator,
            dtype=dtype,
        ),
        active.reshape(-1),
    )


def _require_stop_gradient(method, gradient: str, *, phase: str) -> None:
    if gradient != "stop":
        method_name = type(method).__name__
        raise ValueError(
            f'{method_name} supports only gradient="stop" in {phase}; '
            'gradient="replay" is introduced in Phase B4'
        )


def _method_phase(method) -> str:
    if isinstance(
        method,
        (TensorProduct, AdaptiveTensorClenshawCurtis, AdaptiveCubature),
    ):
        return "Phase B1"
    if isinstance(method, (Smolyak, AdaptiveSmolyak)):
        return "Phase B2"
    if isinstance(method, (Sobol, ScrambledSobol, AdaptiveScrambledSobol)):
        return "Phase B3"
    raise TypeError(f"{type(method).__name__} is not an implemented Phase B method")


def _solve_multidim(
    config: MultidimConfig,
    domain,
    args,
    key,
    epsabs,
    epsrel,
) -> MultidimPrimalSolve:
    method = config.method
    common = {
        "args": args,
        "method": method,
        "measure": config.measure,
        "epsabs": epsabs,
        "epsrel": epsrel,
        "max_evaluations": config.max_evaluations,
        "error_norm": config.error_norm,
    }
    if isinstance(method, AdaptiveScrambledSobol):
        result = integrate_adaptive_scrambled_qmc(
            config.fun,
            domain,
            **common,
            key=key,
        )
        formula = _qmc_formula(method, domain, key, result=result)
    elif isinstance(method, ScrambledSobol):
        result = integrate_scrambled_qmc(
            config.fun,
            domain,
            **common,
            key=key,
        )
        formula = _qmc_formula(method, domain, key)
    elif isinstance(method, Sobol):
        result = integrate_qmc(
            config.fun,
            domain,
            **common,
            key=key,
        )
        formula = _qmc_formula(method, domain, key)
    elif isinstance(method, AdaptiveSmolyak):
        result, nodes = cast(
            tuple[
                Any,
                tuple[jax.Array, jax.Array, jax.Array],
            ],
            integrate_adaptive_sparse(
                config.fun,
                domain,
                **common,
                max_indices=config.max_indices,
                max_frontier=config.max_frontier,
                max_nodes=config.max_nodes,
                _return_nodes=True,
            ),
        )
        formula = _adaptive_sparse_formula(domain, nodes)
    elif isinstance(method, Smolyak):
        result = integrate_sparse(
            config.fun,
            domain,
            **common,
            max_indices=config.max_indices,
            max_frontier=config.max_frontier,
            max_nodes=config.max_nodes,
        )
        formula = _fixed_sparse_formula(method, domain)
    elif isinstance(method, AdaptiveCubature):
        result, leaves = cast(
            tuple[
                Any,
                tuple[jax.Array, jax.Array, jax.Array],
            ],
            integrate_cubature(
                config.fun,
                domain,
                **common,
                max_regions=config.max_regions,
                _return_leaves=True,
            ),
        )
        formula = _cubature_formula(
            domain,
            leaves,
            config.max_evaluations,
            config.max_regions,
        )
    elif isinstance(method, AdaptiveTensorClenshawCurtis):
        result, levels = cast(
            tuple[Any, jax.Array],
            integrate_adaptive_tensor(
                config.fun,
                domain,
                **common,
                _return_levels=True,
            ),
        )
        formula = _adaptive_tensor_formula(
            method,
            domain,
            levels,
            config.max_evaluations,
        )
    elif isinstance(method, TensorProduct):
        result = integrate_tensor(config.fun, domain, **common)
        formula = _fixed_tensor_formula(method, domain)
    else:
        _method_phase(method)
        raise AssertionError("unreachable")
    return MultidimPrimalSolve(result, formula, config, domain, args)


def _prepare_multidim_solve(
    fun,
    domain,
    *,
    args,
    method,
    measure,
    key,
    epsabs,
    epsrel,
    max_evaluations,
    max_regions,
    max_indices,
    max_frontier,
    max_nodes,
    error_norm,
) -> MultidimPrimalSolve:
    config = MultidimConfig(
        fun=fun,
        method=method,
        measure=measure,
        max_evaluations=max_evaluations,
        max_regions=max_regions,
        max_indices=max_indices,
        max_frontier=max_frontier,
        max_nodes=max_nodes,
        error_norm=error_norm,
    )
    return _solve_multidim(config, domain, args, key, epsabs, epsrel)


def _integrate_hyperrectangle(fun, domain, **kwargs):
    method = kwargs["method"]
    _method_phase(method)
    if kwargs["gradient"] not in ("replay", "stop"):
        raise ValueError('gradient must be "replay" or "stop"')
    if kwargs["gradient"] == "replay":
        config = MultidimConfig(
            fun=fun,
            method=method,
            measure=kwargs["measure"],
            max_evaluations=kwargs["max_evaluations"],
            max_regions=kwargs["max_regions"],
            max_indices=kwargs["max_indices"],
            max_frontier=kwargs["max_frontier"],
            max_nodes=kwargs["max_nodes"],
            error_norm=kwargs["error_norm"],
        )
        return multidim_replay_core(
            config,
            domain,
            kwargs["args"],
            kwargs["key"],
            kwargs["epsabs"],
            kwargs["epsrel"],
        )
    _require_stop_gradient(
        method,
        kwargs["gradient"],
        phase=_method_phase(method),
    )
    common = {
        "args": kwargs["args"],
        "method": method,
        "measure": kwargs["measure"],
        "epsabs": kwargs["epsabs"],
        "epsrel": kwargs["epsrel"],
        "max_evaluations": kwargs["max_evaluations"],
        "error_norm": kwargs["error_norm"],
    }
    if isinstance(method, AdaptiveScrambledSobol):
        result = integrate_adaptive_scrambled_qmc(
            fun,
            domain,
            **common,
            key=kwargs["key"],
        )
    elif isinstance(method, ScrambledSobol):
        result = integrate_scrambled_qmc(
            fun,
            domain,
            **common,
            key=kwargs["key"],
        )
    elif isinstance(method, Sobol):
        result = integrate_qmc(
            fun,
            domain,
            **common,
            key=kwargs["key"],
        )
    elif isinstance(method, AdaptiveSmolyak):
        result = integrate_adaptive_sparse(
            fun,
            domain,
            **common,
            max_indices=kwargs["max_indices"],
            max_frontier=kwargs["max_frontier"],
            max_nodes=kwargs["max_nodes"],
        )
    elif isinstance(method, Smolyak):
        result = integrate_sparse(
            fun,
            domain,
            **common,
            max_indices=kwargs["max_indices"],
            max_frontier=kwargs["max_frontier"],
            max_nodes=kwargs["max_nodes"],
        )
    elif isinstance(method, AdaptiveCubature):
        result = integrate_cubature(
            fun,
            domain,
            **common,
            max_regions=kwargs["max_regions"],
        )
    elif isinstance(method, AdaptiveTensorClenshawCurtis):
        result = integrate_adaptive_tensor(fun, domain, **common)
    elif isinstance(method, TensorProduct):
        result = integrate_tensor(fun, domain, **common)
    else:
        raise AssertionError("unreachable")
    return jax.tree.map(jax.lax.stop_gradient, result)


def integrate(
    fun: Callable,
    domain,
    *,
    args: Any = (),
    method,
    measure=None,
    epsabs,
    epsrel,
    max_evaluations: int,
    max_regions: int | None = None,
    max_indices: int | None = None,
    max_frontier: int | None = None,
    max_nodes: int | None = None,
    key=None,
    error_norm: ErrorNorm = MaxNorm(),
    gradient: str = "replay",
):
    """Integrate one domain with the selected static method.

    For :class:`AdaptiveCubature`, scalar eager and JIT execution physically
    skips child evaluation after termination. ``jax.vmap`` preserves result
    semantics and logical work only; its select-style batching may still
    evaluate inactive child branches. Apply ``jax.lax.map`` around scalar
    ``integrate`` calls when physical per-lane masking matters for an expensive
    heterogeneous batch.
    """
    if isinstance(domain, Hyperrectangle):
        normalized = None
        if quantity_mode(domain, epsabs):
            normalized = normalize_multidim_call(
                fun,
                domain,
                args,
                method,
                LebesgueMeasure() if measure is None else measure,
                epsabs,
                epsrel,
            )
            fun = normalized.fun
            domain = normalized.domain
            args = normalized.args
            method = normalized.method
            measure = normalized.measure
            epsabs = normalized.epsabs
            epsrel = normalized.epsrel
        result = _integrate_hyperrectangle(
            fun,
            domain,
            args=args,
            method=method,
            measure=measure,
            epsabs=epsabs,
            epsrel=epsrel,
            max_evaluations=max_evaluations,
            max_regions=max_regions,
            max_indices=max_indices,
            max_frontier=max_frontier,
            max_nodes=max_nodes,
            key=key,
            error_norm=error_norm,
            gradient=gradient,
        )
        if normalized is not None:
            return restore_result(result, normalized.result_unit)
        return result
    if max_regions is None:
        raise ValueError("one-dimensional integration requires max_regions")
    if any(value is not None for value in (max_indices, max_frontier, max_nodes, key)):
        raise TypeError(
            "one-dimensional integration does not accept multidimensional "
            "capacity controls or key"
        )
    return integrate_1d(
        fun,
        domain,
        args=args,
        method=method,
        measure=measure,
        epsabs=epsabs,
        epsrel=epsrel,
        max_evaluations=max_evaluations,
        max_regions=max_regions,
        error_norm=error_norm,
        gradient=gradient,
    )
