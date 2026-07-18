import jax
import jax.numpy as jnp
import pytest

from jaxstro import quad
from jaxstro.quad._scramble import (
    _MATRIX_TAG,
    _SHIFT_TAG,
    _owen_permutation_key,
    scramble_integers,
)
from jaxstro.quad._sobol import sobol_integer_points


def test_replicate_fold_in_is_stable_when_capacity_grows():
    key = jax.random.key(7)
    first = [jax.random.fold_in(key, index) for index in range(8)]
    grown = [jax.random.fold_in(key, index) for index in range(16)]
    assert all(jnp.array_equal(a, b) for a, b in zip(first, grown))


def test_each_scramble_is_reproducible_and_prefix_preserving():
    points = sobol_integer_points(6, 3, bits=24)
    key = jax.random.key(11)
    for method in (
        quad.DigitalShift(),
        quad.LinearMatrixScramble(),
        quad.OwenScramble(),
    ):
        first = scramble_integers(points, method=method, key=key, bits=24)
        second = scramble_integers(points, method=method, key=key, bits=24)
        prefix = scramble_integers(
            points[:16],
            method=method,
            key=key,
            bits=24,
        )
        assert jnp.array_equal(first, second)
        assert jnp.array_equal(first[:16], prefix)


@pytest.mark.parametrize(
    "method",
    (
        quad.DigitalShift(),
        quad.LinearMatrixScramble(),
        quad.OwenScramble(),
    ),
)
def test_independent_keys_change_randomization(method):
    points = sobol_integer_points(4, 2, bits=24)
    first = scramble_integers(
        points,
        method=method,
        key=jax.random.key(1),
        bits=24,
    )
    second = scramble_integers(
        points,
        method=method,
        key=jax.random.key(2),
        bits=24,
    )
    assert not jnp.array_equal(first, second)


def test_digital_shift_is_one_coordinatewise_xor_translation():
    points = sobol_integer_points(5, 4, bits=24)
    scrambled = scramble_integers(
        points,
        method=quad.DigitalShift(),
        key=jax.random.key(3),
        bits=24,
    )
    translations = jnp.bitwise_xor(points, scrambled)
    assert jnp.all(translations == translations[0])


def test_lms_plus_shift_preserves_affine_gf2_identity():
    dtype = jnp.uint32
    x = jnp.asarray((0x123456, 0x654321), dtype=dtype)
    y = jnp.asarray((0x0F0F0F, 0x333333), dtype=dtype)
    points = jnp.stack((jnp.zeros_like(x), x, y, jnp.bitwise_xor(x, y)))
    scrambled = scramble_integers(
        points,
        method=quad.LinearMatrixScramble(),
        key=jax.random.key(5),
        bits=24,
    )
    assert jnp.array_equal(
        jnp.bitwise_xor(
            jnp.bitwise_xor(scrambled[1], scrambled[2]),
            scrambled[0],
        ),
        scrambled[3],
    )


def test_lms_recovers_the_tagged_random_unit_lower_triangular_matrix():
    bits = 8
    key = jax.random.key(29)
    basis = jnp.left_shift(
        jnp.ones((bits,), dtype=jnp.uint32),
        jnp.arange(bits - 1, -1, -1, dtype=jnp.uint32),
    )
    points = jnp.concatenate((jnp.zeros((1,), dtype=jnp.uint32), basis))[:, None]
    scrambled = scramble_integers(
        points,
        method=quad.LinearMatrixScramble(),
        key=key,
        bits=bits,
    )[:, 0]
    shift = scrambled[0]
    transformed_basis = jnp.bitwise_xor(scrambled[1:], shift)
    recovered = (
        jnp.right_shift(
            transformed_basis[:, None],
            jnp.arange(bits - 1, -1, -1, dtype=jnp.uint32),
        )
        & jnp.uint32(1)
    ).T

    expected = jnp.tril(
        jax.random.bernoulli(
            jax.random.fold_in(key, _MATRIX_TAG),
            shape=(1, bits, bits),
        )
    )[0]
    diagonal = jnp.arange(bits)
    expected = expected.at[diagonal, diagonal].set(True).astype(jnp.uint32)
    expected_shift = jax.random.bits(
        jax.random.fold_in(key, _SHIFT_TAG),
        shape=(1,),
        dtype=jnp.uint32,
    )[0] & jnp.uint32((1 << bits) - 1)

    assert _MATRIX_TAG != _SHIFT_TAG
    assert shift == expected_shift
    assert jnp.array_equal(recovered, expected)
    assert jnp.array_equal(recovered, jnp.tril(recovered))
    assert jnp.all(jnp.diag(recovered) == 1)
    assert jnp.any(jnp.tril(recovered, k=-1) == 1)

    other = scramble_integers(
        points,
        method=quad.LinearMatrixScramble(),
        key=jax.random.key(30),
        bits=bits,
    )[:, 0]
    other_basis = jnp.bitwise_xor(other[1:], other[0])
    assert not jnp.array_equal(transformed_basis, other_basis)


def _owen_oracle(points, *, key, bits, use_scrambled_prefix):
    outputs = []
    for point in (int(value) for value in points):
        scrambled_prefix = jnp.uint32(0)
        source_prefix = jnp.uint32(0)
        for bit in range(bits):
            source = jnp.uint32((point >> (bits - 1 - bit)) & 1)
            prefix = scrambled_prefix if use_scrambled_prefix else source_prefix
            flip = jax.random.bernoulli(
                _owen_permutation_key(key, 0, bit, prefix)
            ).astype(jnp.uint32)
            scrambled = jnp.bitwise_xor(source, flip)
            scrambled_prefix = (scrambled_prefix << jnp.uint32(1)) | scrambled
            source_prefix = (source_prefix << jnp.uint32(1)) | source
        outputs.append(scrambled_prefix)
    return jnp.asarray(outputs, dtype=jnp.uint32)


def test_owen_scan_conditions_on_already_scrambled_prefix():
    points = jnp.arange(16, dtype=jnp.uint32)
    key = jax.random.key(23)
    actual = scramble_integers(
        points[:, None],
        method=quad.OwenScramble(),
        key=key,
        bits=4,
    )[:, 0]
    expected = _owen_oracle(
        points,
        key=key,
        bits=4,
        use_scrambled_prefix=True,
    )
    source_prefix_mutation = _owen_oracle(
        points,
        key=key,
        bits=4,
        use_scrambled_prefix=False,
    )
    assert jnp.array_equal(actual, expected)
    assert not jnp.array_equal(actual, source_prefix_mutation)


def test_owen_permutation_key_owns_all_53_prefix_bits():
    key = jax.random.key(17)
    low_prefix = jnp.asarray(0x0000000100000000, dtype=jnp.uint64)
    high_prefix = jnp.asarray(0x0000000200000000, dtype=jnp.uint64)
    low_key = _owen_permutation_key(key, 2, 40, low_prefix)
    high_key = _owen_permutation_key(key, 2, 40, high_prefix)
    replay_key = _owen_permutation_key(key, 2, 40, low_prefix)
    assert not jnp.array_equal(
        jax.random.key_data(low_key), jax.random.key_data(high_key)
    )
    assert jnp.array_equal(
        jax.random.key_data(low_key), jax.random.key_data(replay_key)
    )


def test_scrambles_retain_requested_integer_width():
    points = sobol_integer_points(4, 3, bits=24)
    for method in (
        quad.DigitalShift(),
        quad.LinearMatrixScramble(),
        quad.OwenScramble(),
    ):
        scrambled = scramble_integers(
            points,
            method=method,
            key=jax.random.key(19),
            bits=24,
        )
        assert scrambled.dtype == jnp.uint32
        assert jnp.all(scrambled < (1 << 24))


@pytest.mark.parametrize("bits", (0, 65, True))
def test_scramble_rejects_invalid_bit_controls(bits):
    with pytest.raises(ValueError, match="bits"):
        scramble_integers(
            jnp.zeros((4, 2), dtype=jnp.uint32),
            method=quad.DigitalShift(),
            key=jax.random.key(1),
            bits=bits,
        )
