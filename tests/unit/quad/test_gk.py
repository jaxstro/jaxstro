"""Gauss-Kronrod rule data and local estimator contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstro.quad import GaussKronrod
from jaxstro.quad._gk import gauss_kronrod_data, gauss_kronrod_estimate


def _scalar_quadpack_error(values, kronrod_weights, gauss_weights, dtype):
    values = np.asarray(values)
    kronrod_weights = np.asarray(kronrod_weights)
    gauss_weights = np.asarray(gauss_weights)
    result = np.sum(kronrod_weights * values)
    gauss = np.sum(gauss_weights * values)
    resabs = np.sum(kronrod_weights * np.abs(values))
    resasc = np.sum(kronrod_weights * np.abs(values - 0.5 * result))
    error = abs(result - gauss)
    if resasc != 0.0 and error != 0.0:
        error = resasc * min(1.0, (200.0 * error / resasc) ** 1.5)
    machine = np.finfo(dtype)
    if resabs > machine.tiny / (50.0 * machine.eps):
        error = max(error, 50.0 * machine.eps * resabs)
    return result, error, resabs, resasc


@pytest.mark.parametrize("pair", [15, 21, 31, 41, 51, 61])
def test_gauss_kronrod_data_are_symmetric_aligned_probability_free_rules(pair) -> None:
    data = gauss_kronrod_data(GaussKronrod(pair=pair), dtype=jnp.float64)
    assert data.nodes.shape == (pair,)
    assert data.kronrod_weights.shape == (pair,)
    assert data.gauss_weights.shape == (pair,)
    assert jnp.all(jnp.diff(data.nodes) > 0.0)
    assert jnp.array_equal(data.nodes, -data.nodes[::-1])
    assert jnp.array_equal(data.kronrod_weights, data.kronrod_weights[::-1])
    assert jnp.array_equal(data.gauss_weights, data.gauss_weights[::-1])
    assert jnp.all(data.kronrod_weights > 0.0)
    assert jnp.all(data.gauss_weights >= 0.0)
    assert jnp.count_nonzero(data.gauss_weights) == (pair - 1) // 2
    assert jnp.allclose(jnp.sum(data.kronrod_weights), 2.0, rtol=0.0, atol=5e-16)
    assert jnp.allclose(jnp.sum(data.gauss_weights), 2.0, rtol=0.0, atol=5e-16)


def test_gauss_kronrod_estimate_supports_vector_and_complex_payloads() -> None:
    estimate = gauss_kronrod_estimate(
        lambda x: jnp.stack((jnp.exp(1j * x), x**4), axis=-1),
        GaussKronrod(pair=21),
        dtype=jnp.float64,
    )
    expected = jnp.asarray([2.0 * jnp.sin(1.0) + 0j, 2.0 / 5.0 + 0j])
    assert estimate.value.shape == (2,)
    assert jnp.allclose(estimate.value, expected, rtol=2e-14, atol=2e-14)
    assert estimate.error.shape == (2,)
    assert jnp.all(jnp.isreal(estimate.error))
    assert jnp.all(estimate.error >= 0.0)
    assert jnp.all(estimate.resabs >= 0.0)
    assert jnp.all(estimate.resasc >= 0.0)
    assert not estimate.nonfinite


def test_gauss_kronrod_estimate_supports_matrix_payloads() -> None:
    estimate = gauss_kronrod_estimate(
        lambda x: jnp.stack(
            (
                jnp.stack((jnp.ones_like(x), x), axis=-1),
                jnp.stack((x**2, x**3), axis=-1),
            ),
            axis=-1,
        ),
        GaussKronrod(pair=15),
        dtype=jnp.float64,
    )
    assert estimate.value.shape == (2, 2)
    assert estimate.error.shape == (2, 2)
    assert jnp.allclose(estimate.value, jnp.asarray([[2.0, 2.0 / 3.0], [0.0, 0.0]]))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_gauss_kronrod_quadrature_roundoff_floor_branches(dtype) -> None:
    zero = gauss_kronrod_estimate(
        lambda x: jnp.zeros_like(x), GaussKronrod(), dtype=dtype
    )
    constant = gauss_kronrod_estimate(
        lambda x: jnp.ones_like(x), GaussKronrod(), dtype=dtype
    )
    assert jnp.array_equal(zero.error, jnp.asarray(0.0, dtype=dtype))
    assert not zero.roundoff_floor
    expected_floor = jnp.asarray(100.0, dtype=dtype) * jnp.finfo(dtype).eps
    assert jnp.allclose(constant.error, expected_floor, rtol=4e-6, atol=0.0)
    assert constant.roundoff_floor


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize(
    "fun",
    [
        lambda x: jnp.exp(x),
        lambda x: jnp.abs(x - 0.1),
        lambda x: jnp.exp(1j * x),
    ],
)
def test_gauss_kronrod_estimator_matches_independent_scalar_translation(
    dtype, fun
) -> None:
    method = GaussKronrod(pair=21)
    data = gauss_kronrod_data(method, dtype=dtype)
    values = fun(data.nodes)
    expected = _scalar_quadpack_error(
        values, data.kronrod_weights, data.gauss_weights, np.dtype(dtype)
    )
    estimate = gauss_kronrod_estimate(fun, method, dtype=dtype)
    tolerance = 3e-6 if dtype == jnp.float32 else 3e-14
    assert jnp.allclose(estimate.value, expected[0], rtol=tolerance, atol=tolerance)
    assert jnp.allclose(estimate.error, expected[1], rtol=tolerance, atol=tolerance)
    assert jnp.allclose(estimate.resabs, expected[2], rtol=tolerance, atol=tolerance)
    assert jnp.allclose(estimate.resasc, expected[3], rtol=tolerance, atol=tolerance)


def test_gauss_kronrod_safe_underflow_branch_does_not_force_a_floor() -> None:
    tiny_value = 10.0 * jnp.finfo(jnp.float64).tiny
    data = gauss_kronrod_data(GaussKronrod(), dtype=jnp.float64)
    estimate = gauss_kronrod_estimate(
        lambda x: jnp.full_like(x, tiny_value),
        GaussKronrod(),
        dtype=jnp.float64,
    )
    values = jnp.full_like(data.nodes, tiny_value)
    raw_difference = jnp.abs(
        jnp.sum(data.kronrod_weights * values) - jnp.sum(data.gauss_weights * values)
    )
    assert estimate.resabs > 0.0
    assert jnp.array_equal(estimate.error, raw_difference)
    assert not estimate.roundoff_floor


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.bool_, jnp.complex64])
def test_gauss_kronrod_rejects_non_real_rule_dtypes(dtype) -> None:
    with pytest.raises(TypeError, match="float32 or float64"):
        gauss_kronrod_data(GaussKronrod(), dtype=dtype)


def test_gauss_kronrod_promotes_integer_payload_to_rule_precision() -> None:
    estimate = gauss_kronrod_estimate(
        lambda x: jnp.ones(x.shape, dtype=jnp.int32),
        GaussKronrod(),
        dtype=jnp.float64,
    )
    assert estimate.value.dtype == jnp.float64
    assert jnp.allclose(estimate.value, 2.0)
    assert jnp.isfinite(estimate.error)


def test_gauss_kronrod_uses_selected_precision_for_mixed_payload() -> None:
    method = GaussKronrod()
    data = gauss_kronrod_data(method, dtype=jnp.float64)

    def fun(x):
        return jnp.asarray(jnp.exp(x), dtype=jnp.float32)

    values = np.asarray(fun(data.nodes), dtype=np.float64)
    expected = _scalar_quadpack_error(
        values,
        np.asarray(data.kronrod_weights),
        np.asarray(data.gauss_weights),
        np.dtype("float64"),
    )
    estimate = gauss_kronrod_estimate(fun, method, dtype=jnp.float64)
    assert estimate.value.dtype == jnp.float64
    assert jnp.allclose(estimate.error, expected[1], rtol=3e-14, atol=3e-14)


def test_gauss_kronrod_flags_embedded_gauss_overflow_from_finite_values() -> None:
    method = GaussKronrod(pair=21)
    data = gauss_kronrod_data(method, dtype=jnp.float64)
    huge = 0.6 * jnp.finfo(jnp.float64).max
    values = jnp.where(data.gauss_weights > 0.0, huge, 0.0)
    assert jnp.all(jnp.isfinite(values))
    estimate = gauss_kronrod_estimate(lambda _nodes: values, method, dtype=jnp.float64)
    assert estimate.nonfinite


def test_gauss_kronrod_nonfinite_and_payload_axis_contracts() -> None:
    estimate = gauss_kronrod_estimate(
        lambda x: jnp.where(x > 0.0, jnp.nan, x),
        GaussKronrod(),
        dtype=jnp.float64,
    )
    assert estimate.nonfinite
    with pytest.raises(ValueError, match="leading node axis"):
        gauss_kronrod_estimate(
            lambda _x: jnp.asarray(1.0),
            GaussKronrod(),
            dtype=jnp.float64,
        )


@pytest.mark.parametrize("pair", [15, 21, 31, 41, 51, 61])
def test_gauss_kronrod_estimator_has_one_vectorized_integrand_trace(pair) -> None:
    method = GaussKronrod(pair=pair)
    jaxpr = jax.make_jaxpr(
        lambda scale: (
            gauss_kronrod_estimate(
                lambda x: jnp.exp(scale * x), method, dtype=jnp.float64
            ).value
        )
    )(jnp.asarray(0.2))
    primitives = [equation.primitive.name for equation in jaxpr.jaxpr.eqns]
    assert primitives.count("exp") == 1
