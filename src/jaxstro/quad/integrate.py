from collections.abc import Callable
from typing import Any

import jax

from .adaptive import integrate as integrate_1d
from .domains import Hyperrectangle
from .tensor import TensorProduct, integrate_tensor
from .tolerance import ErrorNorm, MaxNorm


def _integrate_hyperrectangle(*args, **kwargs):
    method = kwargs["method"]
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
