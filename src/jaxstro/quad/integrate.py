from collections.abc import Callable
from typing import Any

import jax

from .adaptive import integrate as integrate_1d
from .cubature import AdaptiveCubature, integrate_cubature
from .domains import Hyperrectangle
from .qmc import ScrambledSobol, Sobol, integrate_qmc, integrate_scrambled_qmc
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


def _require_stop_gradient(method, gradient: str, *, phase: str) -> None:
    if gradient != "stop":
        method_name = type(method).__name__
        raise ValueError(
            f'{method_name} supports only gradient="stop" in {phase}; '
            'gradient="replay" is introduced in Phase B4'
        )


def _integrate_hyperrectangle(*args, **kwargs):
    method = kwargs["method"]
    if isinstance(method, ScrambledSobol):
        _require_stop_gradient(method, kwargs["gradient"], phase="Phase B3")
        result = integrate_scrambled_qmc(
            *args,
            args=kwargs["args"],
            method=method,
            measure=kwargs["measure"],
            epsabs=kwargs["epsabs"],
            epsrel=kwargs["epsrel"],
            max_evaluations=kwargs["max_evaluations"],
            key=kwargs["key"],
            error_norm=kwargs["error_norm"],
        )
        return jax.tree.map(jax.lax.stop_gradient, result)
    if isinstance(method, Sobol):
        _require_stop_gradient(method, kwargs["gradient"], phase="Phase B3")
        result = integrate_qmc(
            *args,
            args=kwargs["args"],
            method=method,
            measure=kwargs["measure"],
            epsabs=kwargs["epsabs"],
            epsrel=kwargs["epsrel"],
            max_evaluations=kwargs["max_evaluations"],
            key=kwargs["key"],
            error_norm=kwargs["error_norm"],
        )
        return jax.tree.map(jax.lax.stop_gradient, result)
    if isinstance(method, AdaptiveSmolyak):
        _require_stop_gradient(method, kwargs["gradient"], phase="Phase B2")
        result = integrate_adaptive_sparse(
            *args,
            args=kwargs["args"],
            method=method,
            measure=kwargs["measure"],
            epsabs=kwargs["epsabs"],
            epsrel=kwargs["epsrel"],
            max_evaluations=kwargs["max_evaluations"],
            max_indices=kwargs["max_indices"],
            max_frontier=kwargs["max_frontier"],
            max_nodes=kwargs["max_nodes"],
            error_norm=kwargs["error_norm"],
        )
        return jax.tree.map(jax.lax.stop_gradient, result)
    if isinstance(method, Smolyak):
        _require_stop_gradient(method, kwargs["gradient"], phase="Phase B2")
        result = integrate_sparse(
            *args,
            args=kwargs["args"],
            method=method,
            measure=kwargs["measure"],
            epsabs=kwargs["epsabs"],
            epsrel=kwargs["epsrel"],
            max_evaluations=kwargs["max_evaluations"],
            max_indices=kwargs["max_indices"],
            max_frontier=kwargs["max_frontier"],
            max_nodes=kwargs["max_nodes"],
            error_norm=kwargs["error_norm"],
        )
        return jax.tree.map(jax.lax.stop_gradient, result)
    if isinstance(method, AdaptiveCubature):
        _require_stop_gradient(method, kwargs["gradient"], phase="Phase B1")
        result = integrate_cubature(
            *args,
            args=kwargs["args"],
            method=method,
            measure=kwargs["measure"],
            epsabs=kwargs["epsabs"],
            epsrel=kwargs["epsrel"],
            max_evaluations=kwargs["max_evaluations"],
            max_regions=kwargs["max_regions"],
            error_norm=kwargs["error_norm"],
        )
        return jax.tree.map(jax.lax.stop_gradient, result)
    if isinstance(method, AdaptiveTensorClenshawCurtis):
        _require_stop_gradient(method, kwargs["gradient"], phase="Phase B1")
        result = integrate_adaptive_tensor(
            *args,
            args=kwargs["args"],
            method=method,
            measure=kwargs["measure"],
            epsabs=kwargs["epsabs"],
            epsrel=kwargs["epsrel"],
            max_evaluations=kwargs["max_evaluations"],
            error_norm=kwargs["error_norm"],
        )
        return jax.tree.map(jax.lax.stop_gradient, result)
    if isinstance(method, TensorProduct):
        _require_stop_gradient(method, kwargs["gradient"], phase="Phase B1")
        result = integrate_tensor(
            *args,
            args=kwargs["args"],
            method=method,
            measure=kwargs["measure"],
            epsabs=kwargs["epsabs"],
            epsrel=kwargs["epsrel"],
            max_evaluations=kwargs["max_evaluations"],
            error_norm=kwargs["error_norm"],
        )
        return jax.tree.map(jax.lax.stop_gradient, result)
    raise TypeError(f"{type(method).__name__} is not an implemented Phase B method")


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
        return _integrate_hyperrectangle(
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
