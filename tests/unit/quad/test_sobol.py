import os
import subprocess
import sys

import jax.numpy as jnp
import pytest

from jaxstro.quad._sobol import (
    direction_numbers,
    sobol_integer_points,
    sobol_points,
)


def test_first_ten_three_dimensional_points_match_joe_kuo_example():
    expected = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.75, 0.25, 0.25],
            [0.25, 0.75, 0.75],
            [0.375, 0.375, 0.625],
            [0.875, 0.875, 0.125],
            [0.625, 0.125, 0.875],
            [0.125, 0.625, 0.375],
            [0.1875, 0.3125, 0.9375],
            [0.6875, 0.8125, 0.4375],
        ],
        dtype=jnp.float64,
    )
    assert jnp.array_equal(sobol_points(4, 3, jnp.float64)[:10], expected)


def test_prefixes_are_exactly_nested_in_integer_and_float_space():
    assert jnp.array_equal(
        sobol_integer_points(4, 7, bits=24),
        sobol_integer_points(5, 7, bits=24)[:16],
    )
    assert jnp.array_equal(
        sobol_points(4, 7, jnp.float64),
        sobol_points(5, 7, jnp.float64)[:16],
    )


def test_dimension_one_direction_numbers_are_descending_binary_units():
    expected = jnp.asarray(
        [1 << shift for shift in range(7, -1, -1)],
        dtype=jnp.uint32,
    )
    assert jnp.array_equal(direction_numbers(1, 8), expected[None, :])


def test_direction_recurrence_reaches_the_vendored_dimension_boundary():
    directions = direction_numbers(21201, 8)
    assert directions.shape == (21201, 8)
    assert jnp.all(directions[-1] > 0)


def test_direction_recurrence_matches_independent_high_bit_sentinels():
    # Frozen from SciPy 1.16.0's independent Joe-Kuo recurrence implementation.
    expected = jnp.asarray(
        [
            (
                8388608,
                12582912,
                2097152,
                5242880,
                16252928,
                7602176,
                10616832,
                9633792,
                14188544,
                2441216,
                5890048,
                15126528,
                7866368,
                11799552,
                8520192,
                12780800,
                2133888,
                5326656,
                16509472,
                7723312,
                10519944,
                9522772,
                14411678,
                2481005,
            ),
            (
                8388608,
                12582912,
                14680064,
                5242880,
                6815744,
                7077888,
                16646144,
                4259840,
                3768320,
                11976704,
                9396224,
                1159168,
                186368,
                14545920,
                1257984,
                14613248,
                4584320,
                16702400,
                1365920,
                8487376,
                9273592,
                14021708,
                15675962,
                11074341,
            ),
            (
                8388608,
                4194304,
                14680064,
                11534336,
                7864320,
                1835008,
                4849664,
                15663104,
                11042816,
                4014080,
                12754944,
                15077376,
                15067136,
                9870336,
                14011904,
                6878464,
                14669184,
                5524288,
                7184864,
                2883632,
                7471144,
                1245204,
                5406770,
                11157529,
            ),
        ],
        dtype=jnp.uint32,
    )
    directions = direction_numbers(21201, 24)
    assert jnp.array_equal(directions[jnp.asarray((3, 9999, 21200))], expected)


def test_float64_sobol_fails_closed_when_jax_x64_is_disabled():
    script = """
import jax.numpy as jnp
from jaxstro.quad._sobol import sobol_points

try:
    sobol_points(3, 2, jnp.float64, bits=53)
except ValueError as error:
    assert "jax_enable_x64=True" in str(error)
else:
    raise AssertionError("float64 Sobol must reject disabled x64")
"""
    environment = os.environ.copy()
    environment["JAX_ENABLE_X64"] = "0"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("dtype", "bits"),
    ((jnp.float32, 25), (jnp.float64, 54)),
)
def test_distinct_coordinate_bit_limit_is_eager(dtype, bits):
    with pytest.raises(ValueError, match="distinct"):
        sobol_points(3, 2, dtype, bits=bits)


@pytest.mark.parametrize("dimension", (0, 21202))
def test_dimension_limit_is_eager(dimension):
    with pytest.raises(ValueError, match="between 1 and 21201"):
        direction_numbers(dimension, 8)


@pytest.mark.parametrize(
    ("level", "bits"),
    ((-1, 8), (9, 8), (True, 8), (3, True)),
)
def test_integer_prefix_rejects_invalid_static_controls(level, bits):
    with pytest.raises(ValueError):
        sobol_integer_points(level, 2, bits=bits)
