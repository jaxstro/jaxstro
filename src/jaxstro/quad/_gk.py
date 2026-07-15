"""Netlib-derived Gauss-Kronrod data and local embedded-rule estimates."""

from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array

from ._gk_data import GK_POSITIVE_DATA
from .methods import GaussKronrod

_DEGREES = {
    15: (13, 23),
    21: (19, 31),
    31: (29, 47),
    41: (39, 61),
    51: (49, 77),
    61: (59, 91),
}


class GaussKronrodData(NamedTuple):
    """Expanded symmetric Kronrod rule and aligned embedded Gauss weights."""

    nodes: Array
    kronrod_weights: Array
    gauss_weights: Array
    gauss_degree: int
    kronrod_degree: int


class GaussKronrodEstimate(NamedTuple):
    """Payload estimate and QUADPACK-style local error evidence."""

    value: Array
    error: Array
    resabs: Array
    resasc: Array
    nonfinite: Array
    roundoff_floor: Array


def _symmetric(values: tuple[float, ...]) -> tuple[float, ...]:
    positive = values[:-1]
    return tuple(-value for value in positive) + (values[-1],) + positive[::-1]


def _symmetric_weights(values: tuple[float, ...]) -> tuple[float, ...]:
    positive = values[:-1]
    return positive + (values[-1],) + positive[::-1]


def gauss_kronrod_data(method: GaussKronrod, *, dtype=None) -> GaussKronrodData:
    """Expand one canonical positive-half QUADPACK table."""
    selected_dtype = jnp.asarray(0.0).dtype if dtype is None else jnp.dtype(dtype)
    if selected_dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise TypeError("Gauss-Kronrod rule dtype must be float32 or float64")
    positive = GK_POSITIVE_DATA[method.pair]
    xgk = tuple(positive["xgk"])
    wgk = tuple(positive["wgk"])
    wg = tuple(positive["wg"])
    aligned_gauss = [0.0] * len(xgk)
    for one_based_index, weight in enumerate(wg, start=1):
        aligned_gauss[2 * one_based_index - 1] = weight
    gauss_degree, kronrod_degree = _DEGREES[method.pair]
    return GaussKronrodData(
        nodes=jnp.asarray(_symmetric(xgk), dtype=selected_dtype),
        kronrod_weights=jnp.asarray(_symmetric_weights(wgk), dtype=selected_dtype),
        gauss_weights=jnp.asarray(
            _symmetric_weights(tuple(aligned_gauss)), dtype=selected_dtype
        ),
        gauss_degree=gauss_degree,
        kronrod_degree=kronrod_degree,
    )


def _weighted_sum(values: Array, weights: Array) -> Array:
    shape = (weights.shape[0],) + (1,) * (values.ndim - 1)
    return jnp.sum(values * jnp.reshape(weights, shape), axis=0)


def gauss_kronrod_estimate(
    fun: Callable[[Array], Array],
    method: GaussKronrod,
    *,
    dtype=None,
) -> GaussKronrodEstimate:
    """Evaluate one embedded pair and its stabilized local error indicator."""
    data = gauss_kronrod_data(method, dtype=dtype)
    values = jnp.asarray(fun(data.nodes))
    if values.ndim == 0 or values.shape[0] != method.pair:
        raise ValueError("Gauss-Kronrod integrand output must have a leading node axis")
    if jnp.issubdtype(values.dtype, jnp.complexfloating):
        target_dtype = (
            jnp.complex64 if data.nodes.dtype == jnp.float32 else jnp.complex128
        )
    else:
        target_dtype = data.nodes.dtype
    values = values.astype(target_dtype)

    value = _weighted_sum(values, data.kronrod_weights)
    gauss_value = _weighted_sum(values, data.gauss_weights)
    magnitudes = jnp.abs(values)
    resabs = _weighted_sum(magnitudes, data.kronrod_weights)
    mean = value / 2.0
    resasc = _weighted_sum(jnp.abs(values - mean), data.kronrod_weights)
    raw_error = jnp.abs(value - gauss_value)
    safe_resasc = jnp.where(resasc != 0.0, resasc, 1.0)
    rescaled = resasc * jnp.minimum(1.0, (200.0 * raw_error / safe_resasc) ** 1.5)
    stabilized = jnp.where((resasc != 0.0) & (raw_error != 0.0), rescaled, raw_error)

    real_dtype = data.nodes.dtype
    machine = jnp.finfo(real_dtype)
    floor = jnp.where(
        resabs > machine.tiny / (50.0 * machine.eps),
        50.0 * machine.eps * resabs,
        0.0,
    )
    error = jnp.maximum(stabilized, floor)
    nonfinite = ~(
        jnp.all(jnp.isfinite(values))
        & jnp.all(jnp.isfinite(value))
        & jnp.all(jnp.isfinite(gauss_value))
        & jnp.all(jnp.isfinite(raw_error))
        & jnp.all(jnp.isfinite(rescaled))
        & jnp.all(jnp.isfinite(stabilized))
        & jnp.all(jnp.isfinite(floor))
        & jnp.all(jnp.isfinite(error))
        & jnp.all(jnp.isfinite(resabs))
        & jnp.all(jnp.isfinite(resasc))
    )
    return GaussKronrodEstimate(
        value=value,
        error=error,
        resabs=resabs,
        resasc=resasc,
        nonfinite=nonfinite,
        roundoff_floor=jnp.any(floor > stabilized),
    )


__all__ = ["gauss_kronrod_data", "gauss_kronrod_estimate"]
