"""Correctly named digital randomizations for integer Sobol nets."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ._sobol import _require_x64, _validate_bits

_MATRIX_TAG = 0x4C4D5301
_SHIFT_TAG = 0x53484946
_OWEN_TAG = 0x4F57454E


@dataclass(frozen=True)
class DigitalShift:
    """One independent coordinatewise digital XOR shift."""


@dataclass(frozen=True)
class LinearMatrixScramble:
    """Random nonsingular lower-triangular matrix plus digital shift."""


@dataclass(frozen=True)
class OwenScramble:
    """True nested uniform bit permutations conditioned on scrambled prefixes."""


def _validate_scramble_inputs(points: Array, *, bits: int, key) -> jnp.dtype:
    _validate_bits(bits)
    if key is None:
        raise TypeError("Sobol randomization requires an explicit JAX key")
    if points.ndim != 2:
        raise ValueError("Sobol integer points must have shape (points, dimension)")
    expected = jnp.dtype(jnp.uint32 if bits <= 32 else jnp.uint64)
    if jnp.dtype(points.dtype) != expected:
        raise TypeError(
            f"{bits}-bit Sobol randomization requires {expected.name} integer points"
        )
    if bits > 32:
        _require_x64("Sobol randomization above 32 bits")
    return expected


def _bit_mask(bits: int, dtype: jnp.dtype) -> Array:
    if bits == jnp.iinfo(dtype).bits:
        return jnp.asarray(jnp.iinfo(dtype).max, dtype=dtype)
    return jnp.asarray((1 << bits) - 1, dtype=dtype)


def _random_words(key, *, shape: tuple[int, ...], bits: int, dtype: jnp.dtype):
    return jax.random.bits(key, shape=shape, dtype=dtype) & _bit_mask(bits, dtype)


def _digital_shift(
    points: Array,
    *,
    key,
    bits: int,
    dtype: jnp.dtype,
) -> Array:
    shift_key = jax.random.fold_in(key, _SHIFT_TAG)
    shifts = _random_words(
        shift_key,
        shape=(points.shape[1],),
        bits=bits,
        dtype=dtype,
    )
    return jnp.bitwise_xor(points, shifts[None, :])


def _linear_matrix_scramble(
    points: Array,
    *,
    key,
    bits: int,
    dtype: jnp.dtype,
) -> Array:
    dimension = points.shape[1]
    matrix_key = jax.random.fold_in(key, _MATRIX_TAG)
    lower = jnp.tril(
        jax.random.bernoulli(
            matrix_key,
            shape=(dimension, bits, bits),
        )
    )
    diagonal = jnp.arange(bits)
    lower = lower.at[:, diagonal, diagonal].set(True).astype(jnp.uint32)
    shifts = jnp.arange(bits - 1, -1, -1, dtype=dtype)
    point_bits = (
        jnp.right_shift(points[..., None], shifts) & jnp.asarray(1, dtype=dtype)
    ).astype(jnp.uint32)
    transformed_bits = (
        jnp.einsum("dbk,ndk->ndb", lower, point_bits) & jnp.uint32(1)
    ).astype(dtype)
    place_values = jnp.left_shift(
        jnp.ones((bits,), dtype=dtype),
        shifts,
    )
    transformed = jnp.sum(transformed_bits * place_values, axis=-1, dtype=dtype)
    return _digital_shift(
        transformed,
        key=key,
        bits=bits,
        dtype=dtype,
    )


def _owen_permutation_key(key, coordinate, bit, prefix: Array):
    """Derive one nested-permutation key from both halves of a prefix."""
    coordinate_key = jax.random.fold_in(
        jax.random.fold_in(key, _OWEN_TAG),
        coordinate,
    )
    bit_key = jax.random.fold_in(coordinate_key, bit)
    if jnp.dtype(prefix.dtype) == jnp.dtype(jnp.uint64):
        prefix_high = jnp.asarray(prefix >> jnp.uint64(32), dtype=jnp.uint32)
        prefix_low = jnp.asarray(
            prefix & jnp.uint64(0xFFFFFFFF),
            dtype=jnp.uint32,
        )
    else:
        prefix_high = jnp.asarray(0, dtype=jnp.uint32)
        prefix_low = jnp.asarray(prefix, dtype=jnp.uint32)
    return jax.random.fold_in(
        jax.random.fold_in(bit_key, prefix_high),
        prefix_low,
    )


def _owen_scramble(
    points: Array,
    *,
    key,
    bits: int,
    dtype: jnp.dtype,
) -> Array:
    one = jnp.asarray(1, dtype=dtype)

    def scramble_coordinate(column, coordinate):
        def scramble_point(point):
            def scramble_bit(prefix, bit):
                source = (
                    jnp.right_shift(
                        point,
                        jnp.asarray(bits - 1, dtype=dtype) - bit.astype(dtype),
                    )
                    & one
                )
                permutation_key = _owen_permutation_key(
                    key,
                    coordinate,
                    bit,
                    prefix,
                )
                flip = jax.random.bernoulli(permutation_key).astype(dtype)
                scrambled = jnp.bitwise_xor(source, flip)
                updated_prefix = jnp.left_shift(prefix, one) | scrambled
                return updated_prefix, scrambled

            prefix, _ = jax.lax.scan(
                scramble_bit,
                jnp.asarray(0, dtype=dtype),
                jnp.arange(bits, dtype=jnp.uint32),
            )
            return prefix

        return jax.vmap(scramble_point)(column)

    coordinates = jnp.arange(points.shape[1], dtype=jnp.uint32)
    return jax.vmap(
        scramble_coordinate,
        in_axes=(1, 0),
        out_axes=1,
    )(points, coordinates)


def scramble_integers(
    points: Array,
    *,
    method: DigitalShift | LinearMatrixScramble | OwenScramble,
    key,
    bits: int,
) -> Array:
    """Apply one reproducible randomization to exact integer Sobol points."""
    dtype = _validate_scramble_inputs(points, bits=bits, key=key)
    if isinstance(method, DigitalShift):
        return _digital_shift(points, key=key, bits=bits, dtype=dtype)
    if isinstance(method, LinearMatrixScramble):
        return _linear_matrix_scramble(points, key=key, bits=bits, dtype=dtype)
    if isinstance(method, OwenScramble):
        return _owen_scramble(points, key=key, bits=bits, dtype=dtype)
    raise TypeError(f"unsupported Sobol randomization {type(method).__name__}")


__all__ = [
    "DigitalShift",
    "LinearMatrixScramble",
    "OwenScramble",
    "scramble_integers",
]
