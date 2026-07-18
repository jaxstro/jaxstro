import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstro.quad._cubature import (
    genz_malik_data,
    genz_malik_estimate,
    select_split_axis,
)
from jaxstro.quad.cubature import GenzMalik
from jaxstro.quad.tolerance import L1Norm, MaxNorm


def _total_degree_exponents(dimension: int, degree: int) -> np.ndarray:
    def fixed_total(total: int, remaining_axes: int):
        if remaining_axes == 1:
            yield (total,)
            return
        for power in range(total + 1):
            for suffix in fixed_total(total - power, remaining_axes - 1):
                yield (power, *suffix)

    return np.asarray(
        [
            exponent
            for total in range(degree + 1)
            for exponent in fixed_total(total, dimension)
        ],
        dtype=np.int32,
    )


def _monomial_moments(data, degree: int) -> tuple[np.ndarray, np.ndarray]:
    exponents = _total_degree_exponents(data.dimension, degree)
    observed_chunks = []
    for start in range(0, exponents.shape[0], 256):
        chunk = jnp.asarray(exponents[start : start + 256])
        values = jnp.prod(data.points[:, None, :] ** chunk[None, :, :], axis=-1)
        observed_chunks.append(np.asarray(data.high_weights @ values))
    observed = np.concatenate(observed_chunks)
    expected = np.prod(1.0 / (exponents + 1.0), axis=-1)
    return observed, expected


def _embedded_monomial_moments(data, degree: int) -> tuple[np.ndarray, np.ndarray]:
    exponents = _total_degree_exponents(data.dimension, degree)
    observed_chunks = []
    for start in range(0, exponents.shape[0], 256):
        chunk = jnp.asarray(exponents[start : start + 256])
        values = jnp.prod(data.points[:, None, :] ** chunk[None, :, :], axis=-1)
        observed_chunks.append(np.asarray(data.low_weights @ values))
    observed = np.concatenate(observed_chunks)
    expected = np.prod(1.0 / (exponents + 1.0), axis=-1)
    return observed, expected


@pytest.mark.parametrize("dimension", range(2, 9))
def test_orbit_counts_slices_and_order_are_deterministic(dimension):
    data = genz_malik_data(dimension, jnp.float64)
    axis_count = 2 * dimension
    pair_count = 2 * dimension * (dimension - 1)
    corner_count = 2**dimension
    point_count = 1 + 2 * axis_count + pair_count + corner_count

    assert data.dimension == dimension
    assert data.point_count == point_count
    assert data.points.shape == (point_count, dimension)
    assert data.center_slice == slice(0, 1)
    assert data.lambda2_axis_slice == slice(1, 1 + axis_count)
    assert data.lambda4_axis_slice == slice(
        1 + axis_count,
        1 + 2 * axis_count,
    )
    assert data.lambda4_pair_slice == slice(
        1 + 2 * axis_count,
        1 + 2 * axis_count + pair_count,
    )
    assert data.lambda5_corner_slice == slice(
        1 + 2 * axis_count + pair_count,
        point_count,
    )

    assert jnp.array_equal(
        data.lambda2_axis_indices,
        jnp.arange(1, 1 + axis_count, dtype=jnp.int32).reshape(dimension, 2),
    )
    assert jnp.array_equal(
        data.lambda4_axis_indices,
        jnp.arange(
            1 + axis_count,
            1 + 2 * axis_count,
            dtype=jnp.int32,
        ).reshape(dimension, 2),
    )

    center = data.points[data.center_slice]
    assert jnp.array_equal(center, jnp.full((1, dimension), 0.5))

    lambda2 = math.sqrt(9.0 / 70.0)
    lambda4 = math.sqrt(9.0 / 10.0)
    for axis in range(dimension):
        expected_lambda2 = np.full((2, dimension), 0.5)
        expected_lambda2[:, axis] += 0.5 * lambda2 * np.asarray([-1.0, 1.0])
        expected_lambda4 = np.full((2, dimension), 0.5)
        expected_lambda4[:, axis] += 0.5 * lambda4 * np.asarray([-1.0, 1.0])
        assert jnp.allclose(
            data.points[data.lambda2_axis_indices[axis]],
            expected_lambda2,
            rtol=0.0,
            atol=2e-16,
        )
        assert jnp.allclose(
            data.points[data.lambda4_axis_indices[axis]],
            expected_lambda4,
            rtol=0.0,
            atol=2e-16,
        )

    pair_points = data.points[data.lambda4_pair_slice]
    pair_cursor = 0
    for first, second in itertools.combinations(range(dimension), 2):
        expected = np.full((4, dimension), 0.5)
        for row, signs in enumerate(itertools.product((-1.0, 1.0), repeat=2)):
            expected[row, first] += 0.5 * lambda4 * signs[0]
            expected[row, second] += 0.5 * lambda4 * signs[1]
        assert jnp.allclose(
            pair_points[pair_cursor : pair_cursor + 4],
            expected,
            rtol=0.0,
            atol=2e-16,
        )
        pair_cursor += 4

    lambda5 = math.sqrt(9.0 / 19.0)
    expected_corners = 0.5 + 0.5 * lambda5 * np.asarray(
        list(itertools.product((-1.0, 1.0), repeat=dimension))
    )
    assert jnp.allclose(
        data.points[data.lambda5_corner_slice],
        expected_corners,
        rtol=0.0,
        atol=2e-16,
    )


@pytest.mark.parametrize("dimension", range(2, 9))
def test_every_orbit_is_reflection_symmetric(dimension):
    data = genz_malik_data(dimension, jnp.float64)

    for orbit in (
        data.lambda2_axis_slice,
        data.lambda4_axis_slice,
        data.lambda4_pair_slice,
        data.lambda5_corner_slice,
    ):
        points = np.asarray(data.points[orbit])
        reflected = 1.0 - points
        assert all(
            np.any(np.all(np.isclose(points, point, rtol=0.0, atol=2e-16), axis=1))
            for point in reflected
        )


@pytest.mark.parametrize("dimension", range(2, 9))
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_target_dtype_and_both_rules_have_unit_mass(dimension, dtype):
    data = genz_malik_data(dimension, dtype)
    tolerance = 16 * np.finfo(np.dtype(dtype)).eps

    assert data.points.dtype == dtype
    assert data.high_weights.dtype == dtype
    assert data.low_weights.dtype == dtype
    assert data.lambda2_axis_indices.dtype == jnp.int32
    assert data.lambda4_axis_indices.dtype == jnp.int32
    assert jnp.allclose(
        jnp.sum(data.high_weights),
        jnp.asarray(1.0, dtype=dtype),
        rtol=0.0,
        atol=tolerance,
    )
    assert jnp.allclose(
        jnp.sum(data.low_weights),
        jnp.asarray(1.0, dtype=dtype),
        rtol=0.0,
        atol=tolerance,
    )


@pytest.mark.parametrize("dimension", range(2, 9))
def test_published_weights_are_attached_to_the_correct_orbits(dimension):
    data = genz_malik_data(dimension, jnp.float64)
    high_expected = (
        (12824.0 - 9120.0 * dimension + 400.0 * dimension**2) / 19683.0,
        980.0 / 6561.0,
        (1820.0 - 400.0 * dimension) / 19683.0,
        200.0 / 19683.0,
        6859.0 / (19683.0 * 2.0**dimension),
    )
    low_expected = (
        (729.0 - 950.0 * dimension + 50.0 * dimension**2) / 729.0,
        245.0 / 486.0,
        (265.0 - 100.0 * dimension) / 1458.0,
        25.0 / 729.0,
        0.0,
    )
    slices = (
        data.center_slice,
        data.lambda2_axis_slice,
        data.lambda4_axis_slice,
        data.lambda4_pair_slice,
        data.lambda5_corner_slice,
    )

    for orbit, high_weight, low_weight in zip(
        slices,
        high_expected,
        low_expected,
        strict=True,
    ):
        assert jnp.array_equal(
            data.high_weights[orbit],
            jnp.full(data.high_weights[orbit].shape, high_weight),
        )
        assert jnp.array_equal(
            data.low_weights[orbit],
            jnp.full(data.low_weights[orbit].shape, low_weight),
        )


@pytest.mark.parametrize("dimension", range(2, 9))
def test_degree_seven_rule_matches_complete_multivariate_moment_matrix(dimension):
    data = genz_malik_data(dimension, jnp.float64)

    observed, expected = _monomial_moments(data, degree=7)

    assert observed.shape == (math.comb(dimension + 7, 7),)
    assert np.allclose(observed, expected, rtol=0.0, atol=3e-13)


@pytest.mark.parametrize("dimension", range(2, 9))
def test_embedded_rule_matches_complete_degree_five_moment_matrix(dimension):
    data = genz_malik_data(dimension, jnp.float64)

    observed, expected = _embedded_monomial_moments(data, degree=5)

    assert observed.shape == (math.comb(dimension + 5, 5),)
    assert np.allclose(observed, expected, rtol=0.0, atol=3e-13)


@pytest.mark.parametrize(
    ("payload", "expected_shape"),
    [
        ("scalar", ()),
        ("vector", (2,)),
        ("complex", (2,)),
    ],
)
def test_local_estimate_weights_scalar_vector_and_complex_payloads(
    payload,
    expected_shape,
):
    data = genz_malik_data(3, jnp.float64)
    base = (
        1.0 + data.points[:, 0] ** 6 + data.points[:, 0] ** 2 * data.points[:, 1] ** 4
    )
    if payload == "scalar":
        values = base
    elif payload == "vector":
        values = jnp.stack((base, 2.0 * base - data.points[:, 2]), axis=-1)
    else:
        values = jnp.stack(
            (
                base + 2.0j * data.points[:, 1],
                -0.5j * base + data.points[:, 2] ** 2,
            ),
            axis=-1,
        )

    estimate = genz_malik_estimate(values, data)
    broadcast_shape = (data.point_count,) + (1,) * (values.ndim - 1)
    expected_high = jnp.sum(
        values * data.high_weights.reshape(broadcast_shape),
        axis=0,
    )
    expected_low = jnp.sum(
        values * data.low_weights.reshape(broadcast_shape),
        axis=0,
    )

    assert estimate.value.shape == expected_shape
    assert estimate.error.shape == expected_shape
    assert jnp.allclose(estimate.value, expected_high, rtol=0.0, atol=2e-14)
    assert jnp.allclose(
        estimate.error,
        jnp.abs(expected_high - expected_low),
        rtol=0.0,
        atol=2e-14,
    )
    assert estimate.axis_difference.shape == (3,)
    assert estimate.nonfinite.shape == ()
    assert not estimate.nonfinite


@pytest.mark.parametrize("bad_value", [jnp.nan, jnp.inf, -jnp.inf])
def test_local_estimate_reports_nonfinite_payloads_without_sanitizing(bad_value):
    data = genz_malik_data(2, jnp.float64)
    values = jnp.ones((data.point_count, 2), dtype=jnp.float64)
    values = values.at[data.lambda4_pair_slice.start, 1].set(bad_value)

    estimate = genz_malik_estimate(values, data)

    assert estimate.nonfinite
    assert not jnp.all(jnp.isfinite(estimate.value))
    assert not jnp.all(jnp.isfinite(estimate.error))


def test_axis_fourth_differences_match_the_five_point_formula():
    data = genz_malik_data(3, jnp.float64)
    centered = data.points - 0.5
    values = centered[:, 0] ** 4 + 3.0 * centered[:, 2] ** 4

    estimate = genz_malik_estimate(values, data)

    outer_squared = (9.0 / 10.0) / 4.0
    inner_squared = (9.0 / 70.0) / 4.0
    quartic_scale = 2.0 * outer_squared * (outer_squared - inner_squared)
    assert jnp.allclose(
        estimate.axis_difference,
        jnp.asarray([quartic_scale, 0.0, 3.0 * quartic_scale]),
        rtol=0.0,
        atol=2e-16,
    )
    assert select_split_axis(estimate.axis_difference) == 2


def test_axis_fourth_differences_reduce_payload_with_the_configured_norm():
    data = genz_malik_data(2, jnp.float64)
    centered = data.points - 0.5
    values = jnp.stack(
        (
            centered[:, 0] ** 4 + centered[:, 1] ** 4,
            centered[:, 0] ** 4 - centered[:, 1] ** 4,
        ),
        axis=-1,
    )

    max_estimate = genz_malik_estimate(values, data, error_norm=MaxNorm())
    l1_estimate = genz_malik_estimate(values, data, error_norm=L1Norm())

    assert jnp.allclose(
        l1_estimate.axis_difference,
        2.0 * max_estimate.axis_difference,
        rtol=0.0,
        atol=2e-16,
    )


def test_axis_selection_uses_the_lowest_axis_on_ties():
    assert select_split_axis(jnp.asarray([4.0, 4.0, 1.0])) == 0
    assert select_split_axis(jnp.asarray([1.0, 4.0, 4.0])) == 1


def test_genz_malik_declaration_and_rule_data_have_stable_pytree_metadata():
    method = GenzMalik()
    method_leaves, method_structure = jax.tree.flatten(method)
    rebuilt_method = jax.tree.unflatten(method_structure, method_leaves)
    data = genz_malik_data(4, jnp.float32)
    data_leaves, data_structure = jax.tree.flatten(data)
    rebuilt_data = jax.tree.unflatten(data_structure, data_leaves)

    assert method_leaves == []
    assert rebuilt_method == method
    assert len(data_leaves) == 5
    assert rebuilt_data.dimension == data.dimension
    assert rebuilt_data.point_count == data.point_count
    assert rebuilt_data.center_slice == data.center_slice
    assert rebuilt_data.lambda2_axis_slice == data.lambda2_axis_slice
    assert rebuilt_data.lambda4_axis_slice == data.lambda4_axis_slice
    assert rebuilt_data.lambda4_pair_slice == data.lambda4_pair_slice
    assert rebuilt_data.lambda5_corner_slice == data.lambda5_corner_slice
    assert all(
        jnp.array_equal(rebuilt, original)
        for rebuilt, original in zip(data_leaves, jax.tree.leaves(rebuilt_data))
    )


def test_local_estimate_composes_with_jit_and_vmap():
    data = genz_malik_data(3, jnp.float64)
    points = data.points
    values = jnp.stack(
        (
            points[:, 0] ** 2 + points[:, 1] ** 4,
            points[:, 0] * points[:, 2] + 1.0j * points[:, 1],
        ),
        axis=-1,
    )
    eager = genz_malik_estimate(values, data)
    compiled = jax.jit(genz_malik_estimate)(values, data)
    batched_values = jnp.stack((values, 2.0 * values), axis=0)
    batched = jax.jit(jax.vmap(genz_malik_estimate, in_axes=(0, None)))(
        batched_values, data
    )

    assert jax.tree.all(
        jax.tree.map(
            lambda actual, expected: jnp.allclose(
                actual,
                expected,
                rtol=0.0,
                atol=2e-14,
                equal_nan=True,
            ),
            compiled,
            eager,
        )
    )
    assert jnp.allclose(batched.value[0], eager.value, rtol=0.0, atol=2e-14)
    assert jnp.allclose(
        batched.value[1],
        2.0 * eager.value,
        rtol=0.0,
        atol=2e-14,
    )
    assert jnp.allclose(
        batched.axis_difference[1],
        2.0 * eager.axis_difference,
        rtol=0.0,
        atol=2e-14,
    )
    assert not jnp.any(batched.nonfinite)


@pytest.mark.parametrize("dimension", [1, 9])
def test_rule_rejects_dimensions_outside_the_validated_envelope(dimension):
    with pytest.raises(
        ValueError,
        match="Phase B1 deterministic methods require dimension 2 through 8",
    ):
        genz_malik_data(dimension, jnp.float64)


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.complex64])
def test_rule_rejects_non_real_floating_target_dtypes(dtype):
    with pytest.raises(TypeError, match="real floating dtype"):
        genz_malik_data(2, dtype)
