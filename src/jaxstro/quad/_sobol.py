"""Joe-Kuo Sobol direction recurrence and exact power-of-two prefixes."""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ._sobol_data import (
    MAX_SOBOL_DIMENSION,
    SOBOL_INITIAL_DIRECTIONS,
    SOBOL_POLYNOMIALS,
)


def _validate_dimension(dimension: int) -> None:
    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or not 1 <= dimension <= MAX_SOBOL_DIMENSION
    ):
        raise ValueError(
            f"Sobol dimension must lie between 1 and {MAX_SOBOL_DIMENSION}"
        )


def _validate_bits(bits: int) -> None:
    if isinstance(bits, bool) or not isinstance(bits, int) or not 1 <= bits <= 64:
        raise ValueError("Sobol bits must be an integer between 1 and 64")


def _validate_level(level: int, bits: int) -> None:
    if isinstance(level, bool) or not isinstance(level, int) or level < 0:
        raise ValueError("Sobol level must be a nonnegative integer")
    if level > bits:
        raise ValueError("Sobol requires level <= bits")


def _target_float_dtype(dtype) -> jnp.dtype:
    selected = jnp.dtype(dtype)
    if selected not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise TypeError("Sobol output dtype must be float32 or float64")
    return selected


def _require_x64(reason: str) -> None:
    if not jax.config.read("jax_enable_x64"):
        raise ValueError(f"{reason} requires jax_enable_x64=True")


def _integer_dtype(bits: int):
    return np.uint32 if bits <= 32 else np.uint64


@lru_cache(maxsize=None)
def _direction_numbers_host(dimension: int, bits: int) -> np.ndarray:
    _validate_dimension(dimension)
    _validate_bits(bits)
    dtype = _integer_dtype(bits)
    directions = np.zeros((dimension, bits), dtype=dtype)
    for bit in range(1, bits + 1):
        directions[0, bit - 1] = dtype(1 << (bits - bit))

    for axis in range(1, dimension):
        degree, coefficient = SOBOL_POLYNOMIALS[axis - 1]
        initial = SOBOL_INITIAL_DIRECTIONS[axis - 1]
        initial_count = min(degree, bits)
        for bit in range(1, initial_count + 1):
            directions[axis, bit - 1] = dtype(initial[bit - 1] << (bits - bit))
        for bit in range(degree + 1, bits + 1):
            value = int(directions[axis, bit - degree - 1])
            value ^= value >> degree
            for offset in range(1, degree):
                if (coefficient >> (degree - 1 - offset)) & 1:
                    value ^= int(directions[axis, bit - offset - 1])
            directions[axis, bit - 1] = dtype(value)
    return directions


def direction_numbers(
    dimension: int,
    bits: int,
    dtype=None,
) -> Array:
    """Return the first ``bits`` direction integers for each coordinate."""
    _validate_dimension(dimension)
    _validate_bits(bits)
    if bits > 32:
        _require_x64("Sobol direction arithmetic above 32 bits")
    if dtype is not None:
        selected = _target_float_dtype(dtype)
        if selected == jnp.dtype(jnp.float64):
            _require_x64("float64 Sobol output")
        limit = 24 if selected == jnp.dtype(jnp.float32) else 53
        if bits > limit:
            raise ValueError(
                f"{selected.name} Sobol coordinates retain at most {limit} "
                "distinct digital bits"
            )
    return jnp.asarray(_direction_numbers_host(dimension, bits))


@lru_cache(maxsize=None)
def _sobol_integer_points_host(
    level: int,
    dimension: int,
    bits: int,
) -> np.ndarray:
    _validate_dimension(dimension)
    _validate_bits(bits)
    _validate_level(level, bits)
    point_count = 1 << level
    directions = _direction_numbers_host(dimension, bits)
    points = np.zeros((point_count, dimension), dtype=directions.dtype)
    for point in range(1, point_count):
        least_significant = (point & -point).bit_length() - 1
        points[point] = points[point - 1] ^ directions[:, least_significant]
    return points


def sobol_integer_points(level: int, dimension: int, *, bits: int) -> Array:
    """Return one exact integer Sobol prefix in Gray-code order."""
    _validate_dimension(dimension)
    _validate_bits(bits)
    _validate_level(level, bits)
    if bits > 32:
        _require_x64("Sobol integer arithmetic above 32 bits")
    return jnp.asarray(_sobol_integer_points_host(level, dimension, bits))


def resolve_sobol_bits(
    level: int,
    dtype,
    *,
    bits: int | None = None,
) -> int:
    """Validate output precision and return the static digital bit count."""
    selected = _target_float_dtype(dtype)
    if selected == jnp.dtype(jnp.float64):
        _require_x64("float64 Sobol output")
    limit = 24 if selected == jnp.dtype(jnp.float32) else 53
    resolved_bits = limit if bits is None else bits
    _validate_bits(resolved_bits)
    _validate_level(level, resolved_bits)
    if resolved_bits > limit:
        raise ValueError(
            f"{selected.name} Sobol coordinates retain at most {limit} "
            "distinct digital bits"
        )
    return resolved_bits


def sobol_points(
    level: int,
    dimension: int,
    dtype,
    *,
    bits: int | None = None,
) -> Array:
    """Return one power-of-two Sobol prefix on the unit hyperrectangle."""
    selected = _target_float_dtype(dtype)
    resolved_bits = resolve_sobol_bits(level, selected, bits=bits)
    integers = sobol_integer_points(level, dimension, bits=resolved_bits)
    scale = jnp.asarray(2.0**resolved_bits, dtype=selected)
    return integers.astype(selected) / scale


__all__ = [
    "direction_numbers",
    "resolve_sobol_bits",
    "sobol_integer_points",
    "sobol_points",
]
