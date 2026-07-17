"""Synchronized compilation and runtime measurements for quadrature evidence."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class TimingRecord:
    lower_seconds: float
    compile_seconds: float
    warm_seconds: tuple[float, ...]
    median_warm_seconds: float
    mad_warm_seconds: float
    minimum_warm_seconds: float
    maximum_warm_seconds: float


@dataclass(frozen=True)
class _PreparedCallable:
    compiled: Any
    lower_seconds: float
    compile_seconds: float


def ready_tree(value: Any) -> Any:
    """Block on every device leaf without changing the PyTree structure."""
    return jax.tree.map(
        lambda leaf: (
            leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf
        ),
        value,
    )


def _prepare(fun: Callable[[Any], Any], argument: Any) -> _PreparedCallable:
    jitted = jax.jit(fun)
    start = time.perf_counter()
    lowered = jitted.lower(argument)
    lower_seconds = time.perf_counter() - start

    start = time.perf_counter()
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - start
    return _PreparedCallable(compiled, lower_seconds, compile_seconds)


def _sample(compiled: Any, argument: Any) -> float:
    start = time.perf_counter()
    ready_tree(compiled(argument))
    return time.perf_counter() - start


def _record(prepared: _PreparedCallable, samples: list[float]) -> TimingRecord:
    values = np.asarray(samples, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return TimingRecord(
        lower_seconds=prepared.lower_seconds,
        compile_seconds=prepared.compile_seconds,
        warm_seconds=tuple(samples),
        median_warm_seconds=median,
        mad_warm_seconds=mad,
        minimum_warm_seconds=float(np.min(values)),
        maximum_warm_seconds=float(np.max(values)),
    )


def _require_repeats(repeats: int) -> None:
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")


def measure_callable(
    fun: Callable[[Any], Any],
    argument: Any,
    *,
    repeats: int = 21,
) -> TimingRecord:
    """Measure lowering, compilation, and synchronized warm execution separately."""
    _require_repeats(repeats)
    prepared = _prepare(fun, argument)
    ready_tree(prepared.compiled(argument))
    samples = [_sample(prepared.compiled, argument) for _ in range(repeats)]
    return _record(prepared, samples)


def measure_pair_interleaved(
    functions: Mapping[str, Callable[[Any], Any]],
    argument: Any,
    *,
    repeats: int = 21,
) -> dict[str, TimingRecord]:
    """Compile first, then alternate execution order to limit temporal bias."""
    _require_repeats(repeats)
    if len(functions) != 2:
        raise ValueError("interleaved timing requires exactly two callables")
    names = tuple(functions)
    prepared = {name: _prepare(functions[name], argument) for name in names}
    for name in names:
        ready_tree(prepared[name].compiled(argument))
    samples = {name: [] for name in names}
    for repetition in range(repeats):
        order = names if repetition % 2 == 0 else names[::-1]
        for name in order:
            samples[name].append(_sample(prepared[name].compiled, argument))
    return {name: _record(prepared[name], samples[name]) for name in names}


def make_jvp_kernel(raw_callable: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Return a kernel retaining raw primal evidence and the value tangent."""

    def kernel(theta):
        primal, tangent = jax.jvp(
            raw_callable,
            (theta,),
            (jnp.ones_like(theta),),
        )
        return primal, tangent.value

    return kernel


def make_grad_kernel(raw_callable: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Return a scalar-value reverse-mode kernel with raw auxiliary evidence."""

    def objective(theta):
        result = raw_callable(theta)
        return jnp.real(result.value), result

    transformed = jax.value_and_grad(objective, has_aux=True)

    def kernel(theta):
        (value, auxiliary), gradient = transformed(theta)
        return value, auxiliary, gradient

    return kernel


def make_vmap_kernel(raw_callable: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Return a batched benchmark kernel over the parameter axis."""
    return jax.vmap(raw_callable)


__all__ = [
    "TimingRecord",
    "make_grad_kernel",
    "make_jvp_kernel",
    "make_vmap_kernel",
    "measure_callable",
    "measure_pair_interleaved",
    "ready_tree",
]
