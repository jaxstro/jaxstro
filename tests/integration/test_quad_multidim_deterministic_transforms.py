from __future__ import annotations

import gc

import jax
import jax.numpy as jnp
import pytest

from jaxstro import quad

TRANSFORM_CONTROLS = {
    "fixed_tensor": {
        "method": quad.TensorProduct(quad.GaussianRule(4)),
        "max_evaluations": 16,
        "expected_evaluations": 16,
        "expected_status": quad.QuadStatus.ERROR_ESTIMATE_UNAVAILABLE,
    },
    "adaptive_tensor": {
        "method": quad.AdaptiveTensorClenshawCurtis(initial_level=2),
        "max_evaluations": 65,
        "expected_evaluations": 65,
        "expected_status": quad.QuadStatus.CONVERGED,
    },
    "adaptive_cubature": {
        "method": quad.AdaptiveCubature(quad.GenzMalik()),
        "max_evaluations": 17,
        "max_regions": 1,
        "expected_evaluations": 17,
        "expected_status": quad.QuadStatus.CONVERGED,
    },
}


@pytest.fixture(autouse=True)
def _bounded_transform_cache_lifetime():
    yield
    jax.clear_caches()
    gc.collect()


def _integrand(payload: str):
    def fun(x, scale):
        ones = scale * jnp.ones(x.shape[0], dtype=x.dtype)
        if payload == "scalar_real":
            return ones
        if payload == "array_real":
            return jnp.stack((ones, 2.0 * ones), axis=-1)
        if payload == "scalar_complex":
            return ones * jnp.asarray(1.0 + 2.0j)
        raise ValueError(f"unknown payload: {payload}")

    return fun


def _expected(payload: str, scale):
    if payload == "scalar_real":
        return scale
    if payload == "array_real":
        return jnp.asarray((scale, 2.0 * scale))
    if payload == "scalar_complex":
        return scale * jnp.asarray(1.0 + 2.0j)
    raise ValueError(f"unknown payload: {payload}")


def _solver(method_name: str, dtype, payload: str):
    controls = TRANSFORM_CONTROLS[method_name]

    def solve(scale):
        kwargs = {
            "method": controls["method"],
            "epsabs": jnp.asarray(2.0e-5, dtype=dtype),
            "epsrel": jnp.asarray(0.0, dtype=dtype),
            "max_evaluations": controls["max_evaluations"],
            "gradient": "stop",
        }
        if method_name == "adaptive_cubature":
            kwargs["max_regions"] = controls["max_regions"]
        return quad.integrate(
            _integrand(payload),
            quad.Hyperrectangle(
                jnp.zeros(2, dtype=dtype),
                jnp.ones(2, dtype=dtype),
            ),
            args=scale,
            **kwargs,
        )

    return solve


def _assert_result(method_name: str, payload: str, scale, result) -> None:
    controls = TRANSFORM_CONTROLS[method_name]
    assert int(result.status) == controls["expected_status"]
    assert int(result.work.evaluations) == controls["expected_evaluations"]
    assert jnp.allclose(result.value, _expected(payload, scale), rtol=2e-5, atol=2e-5)
    assert int(result.work.replicates) == 0
    if method_name == "adaptive_cubature":
        assert int(result.work.active_regions) == 1
        assert int(result.work.refinements) == 0
    else:
        assert int(result.work.active_regions) == 0


@pytest.mark.parametrize("method_name", tuple(TRANSFORM_CONTROLS))
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
@pytest.mark.parametrize(
    "payload",
    ("scalar_real", "array_real", "scalar_complex"),
)
def test_stop_mode_composes_eager_and_jit_across_dtype_and_payload(
    method_name: str,
    dtype,
    payload: str,
):
    solve = _solver(method_name, dtype, payload)
    scale = jnp.asarray(1.25, dtype=dtype)
    eager = solve(scale)
    compiled = jax.jit(solve)(scale)
    _assert_result(method_name, payload, scale, eager)
    _assert_result(method_name, payload, scale, compiled)
    for eager_leaf, compiled_leaf in zip(
        jax.tree.leaves(eager),
        jax.tree.leaves(compiled),
        strict=True,
    ):
        assert jnp.allclose(eager_leaf, compiled_leaf, equal_nan=True)


@pytest.mark.parametrize("method_name", tuple(TRANSFORM_CONTROLS))
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
@pytest.mark.parametrize(
    "payload",
    ("scalar_real", "array_real", "scalar_complex"),
)
def test_stop_mode_composes_with_jit_of_vmap(
    method_name: str,
    dtype,
    payload: str,
):
    solve = _solver(method_name, dtype, payload)
    scales = jnp.asarray((0.5, 1.5), dtype=dtype)
    batched = jax.jit(jax.vmap(solve))(scales)
    assert jnp.array_equal(
        batched.status,
        jnp.full((2,), TRANSFORM_CONTROLS[method_name]["expected_status"]),
    )
    assert jnp.array_equal(
        batched.work.evaluations,
        jnp.full(
            (2,),
            TRANSFORM_CONTROLS[method_name]["expected_evaluations"],
        ),
    )
    assert jnp.allclose(
        batched.value,
        jax.vmap(lambda scale: _expected(payload, scale))(scales),
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.parametrize("method_name", tuple(TRANSFORM_CONTROLS))
def test_stop_mode_has_exact_zero_grad_and_jvp(method_name: str):
    solve = _solver(method_name, jnp.float64, "scalar_real")
    assert jax.grad(lambda scale: solve(scale).value)(1.25) == 0.0
    _, tangent = jax.jvp(
        lambda scale: solve(scale).value,
        (jnp.asarray(1.25),),
        (jnp.asarray(1.0),),
    )
    assert tangent == 0.0


@pytest.mark.parametrize(
    ("method", "message"),
    (
        (
            quad.TensorProduct(quad.GaussianRule(4)),
            'TensorProduct supports only gradient="stop" in Phase B1; '
            'gradient="replay" is introduced in Phase B4',
        ),
        (
            quad.AdaptiveTensorClenshawCurtis(initial_level=2),
            'AdaptiveTensorClenshawCurtis supports only gradient="stop" in Phase B1; '
            'gradient="replay" is introduced in Phase B4',
        ),
        (
            quad.AdaptiveCubature(quad.GenzMalik()),
            'AdaptiveCubature supports only gradient="stop" in Phase B1; '
            'gradient="replay" is introduced in Phase B4',
        ),
    ),
)
@pytest.mark.parametrize("gradient", ("replay", "Replay", "forward", ""))
def test_every_non_stop_mode_is_rejected_with_exact_capability_message(
    method,
    message: str,
    gradient: str,
):
    kwargs = {}
    if isinstance(method, quad.AdaptiveCubature):
        kwargs["max_regions"] = 1
    with pytest.raises(ValueError) as exc_info:
        quad.integrate(
            lambda x: jnp.ones(x.shape[0]),
            quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
            method=method,
            epsabs=1.0e-5,
            epsrel=0.0,
            max_evaluations=65,
            gradient=gradient,
            **kwargs,
        )
    assert str(exc_info.value) == message
