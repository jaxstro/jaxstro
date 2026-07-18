from collections.abc import Callable
from typing import Any

import jax

from .adaptive import integrate as integrate_1d
from .cubature import AdaptiveCubature, integrate_cubature
from .domains import Hyperrectangle
from .tensor import (
    AdaptiveTensorClenshawCurtis,
    TensorProduct,
    integrate_adaptive_tensor,
    integrate_tensor,
)
from .tolerance import ErrorNorm, MaxNorm


def _integrate_hyperrectangle(*args, **kwargs):
    method = kwargs["method"]
    if isinstance(method, AdaptiveCubature):
        if kwargs["gradient"] != "stop":
            raise ValueError('AdaptiveCubature requires gradient="stop"')
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
        if kwargs["gradient"] != "stop":
            raise ValueError('AdaptiveTensorClenshawCurtis requires gradient="stop"')
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
        if kwargs["gradient"] != "stop":
            raise ValueError('TensorProduct requires gradient="stop"')
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
