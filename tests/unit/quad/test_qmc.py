import importlib

import jax
import jax.numpy as jnp
import pytest

from jaxstro import quad


def _deterministic_options(**overrides):
    options = dict(
        method=quad.Sobol(level=10),
        epsabs=1.0e-4,
        epsrel=1.0e-4,
        max_evaluations=1024,
        gradient="stop",
    )
    options.update(overrides)
    return options


def test_deterministic_sobol_result_has_unavailable_error():
    result = quad.integrate(
        lambda x: jnp.prod(x, axis=-1),
        quad.Hyperrectangle(jnp.zeros(4), jnp.ones(4)),
        **_deterministic_options(),
    )
    assert result.work.evaluations == 1024
    assert result.work.levels == 10
    assert result.work.replicates == 0
    assert result.error.kind == quad.ErrorKind.UNAVAILABLE
    assert result.status == quad.QuadStatus.ERROR_ESTIMATE_UNAVAILABLE
    assert jnp.isnan(result.error.confidence_level)


def test_deterministic_sobol_supports_array_and_complex_payloads():
    array_result = quad.integrate(
        lambda x: jnp.stack((jnp.sum(x, axis=-1), jnp.prod(x, axis=-1)), axis=-1),
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        **_deterministic_options(method=quad.Sobol(level=4), max_evaluations=16),
    )
    complex_result = quad.integrate(
        lambda x: (1.0 + 2.0j) * jnp.sum(x, axis=-1),
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        **_deterministic_options(method=quad.Sobol(level=4), max_evaluations=16),
    )
    assert array_result.value.shape == (2,)
    assert jnp.iscomplexobj(complex_result.value)


def test_deterministic_sobol_rejects_insufficient_budget_before_generation():
    with pytest.raises(ValueError, match="1024 evaluations"):
        quad.integrate(
            lambda x: jnp.prod(x, axis=-1),
            quad.Hyperrectangle(jnp.zeros(4), jnp.ones(4)),
            **_deterministic_options(max_evaluations=1023),
        )


def test_deterministic_sobol_zero_volume_does_no_physical_work():
    calls = 0

    def integrand(x):
        def record(_value):
            nonlocal calls
            calls += 1

        jax.debug.callback(record, x[0, 0])
        return jnp.sum(x, axis=-1)

    result = quad.integrate(
        integrand,
        quad.Hyperrectangle(jnp.asarray((0.0, 1.0)), jnp.asarray((2.0, 1.0))),
        **_deterministic_options(method=quad.Sobol(level=4), max_evaluations=16),
    )
    assert calls == 0
    assert result.value == 0.0
    assert result.work.evaluations == 0


def test_deterministic_sobol_zero_volume_skips_point_generation(monkeypatch):
    qmc_module = importlib.import_module("jaxstro.quad.qmc")
    generation_calls = 0
    original = qmc_module.sobol_points

    def record_generation(*args, **kwargs):
        nonlocal generation_calls
        generation_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(qmc_module, "sobol_points", record_generation)
    result = quad.integrate(
        lambda x: jnp.sum(x, axis=-1),
        quad.Hyperrectangle(jnp.asarray((0.0, 1.0)), jnp.asarray((2.0, 1.0))),
        **_deterministic_options(method=quad.Sobol(level=4), max_evaluations=16),
    )
    assert generation_calls == 0
    assert result.work.evaluations == 0


def test_deterministic_sobol_rejects_int32_work_overflow_before_generation():
    with pytest.raises(ValueError, match="int32 work-accounting limit"):
        quad.integrate(
            lambda x: jnp.sum(x, axis=-1),
            quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
            **_deterministic_options(
                method=quad.Sobol(level=31, bits=53),
                max_evaluations=1 << 31,
            ),
        )


def test_traced_nonfinite_bound_precedes_zero_volume():
    @jax.jit
    def solve(upper):
        return quad.integrate(
            lambda x: jnp.sum(x, axis=-1),
            quad.Hyperrectangle(jnp.zeros(2), upper),
            **_deterministic_options(method=quad.Sobol(level=4), max_evaluations=16),
        )

    result = solve(jnp.asarray((0.0, jnp.inf)))
    assert result.status == quad.QuadStatus.INVALID_INPUT
    assert result.work.evaluations == 0


def test_nonfinite_integrand_returns_typed_failure():
    result = quad.integrate(
        lambda x: jnp.where(x[:, 0] == 0.0, jnp.nan, 1.0),
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        **_deterministic_options(method=quad.Sobol(level=4), max_evaluations=16),
    )
    assert result.status == quad.QuadStatus.NONFINITE_INTEGRAND
    assert jnp.isnan(result.value)


@pytest.mark.parametrize("level", (-1, True, 2.5))
def test_sobol_declaration_rejects_invalid_level(level):
    with pytest.raises(ValueError, match="nonnegative integer"):
        quad.Sobol(level=level)


def test_sobol_declaration_accepts_and_rejects_exact_bit_boundary():
    assert quad.Sobol(level=53, bits=53).level == 53
    with pytest.raises(ValueError, match="level <= bits"):
        quad.Sobol(level=54, bits=53)


def test_sobol_replay_fails_closed_at_b4_boundary():
    with pytest.raises(ValueError, match='supports only gradient="stop" in Phase B3'):
        quad.integrate(
            lambda x: jnp.sum(x, axis=-1),
            quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
            **_deterministic_options(
                method=quad.Sobol(level=4),
                max_evaluations=16,
                gradient="replay",
            ),
        )
