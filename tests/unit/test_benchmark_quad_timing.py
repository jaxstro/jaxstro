from __future__ import annotations

import jax
import jax.numpy as jnp
from scripts.quad_benchmark_adapters import RawBenchmarkResult
from scripts.quad_benchmark_timing import (
    make_grad_kernel,
    make_jvp_kernel,
    make_vmap_kernel,
    measure_callable,
    measure_pair_interleaved,
)


def test_measure_callable_separates_compile_and_warm_samples() -> None:
    record = measure_callable(
        lambda x: (jnp.sin(x), jnp.asarray(0, dtype=jnp.int32)),
        jax.device_put(jnp.asarray(0.5)),
        repeats=5,
    )
    assert record.lower_seconds >= 0.0
    assert record.compile_seconds >= 0.0
    assert len(record.warm_seconds) == 5
    assert record.median_warm_seconds > 0.0
    assert record.mad_warm_seconds >= 0.0


def test_interleaved_measurement_preserves_both_sample_counts() -> None:
    records = measure_pair_interleaved(
        {"jaxstro": lambda x: jnp.exp(x), "quadax": lambda x: jnp.exp(x)},
        jax.device_put(jnp.asarray(0.5)),
        repeats=5,
    )
    assert set(records) == {"jaxstro", "quadax"}
    assert all(len(record.warm_seconds) == 5 for record in records.values())


def _raw(theta):
    value = theta**2
    integer = jnp.asarray(0, dtype=jnp.int32)
    return RawBenchmarkResult(
        value=value,
        error=jnp.asarray(0.0),
        status=integer,
        reported_evaluations=integer,
        normalized_evaluations=integer,
        refinements=integer,
        active_regions=integer,
        levels=integer,
    )


def test_transform_factories_preserve_diagnostics_and_value_derivatives() -> None:
    theta = jnp.asarray(3.0)
    primal, tangent = make_jvp_kernel(_raw)(theta)
    assert float(primal.value) == 9.0
    assert float(tangent) == 6.0

    value, auxiliary, gradient = make_grad_kernel(_raw)(theta)
    assert float(value) == 9.0
    assert float(auxiliary.value) == 9.0
    assert float(gradient) == 6.0

    batched = make_vmap_kernel(_raw)(jnp.asarray([1.0, 2.0, 3.0]))
    assert batched.value.shape == (3,)
    assert jnp.array_equal(batched.value, jnp.asarray([1.0, 4.0, 9.0]))
