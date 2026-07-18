import importlib

import jax
import jax.numpy as jnp
import pytest

from jaxstro import quad
from jaxstro.quantity import dimensionless


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


def test_sobol_replay_differentiates_the_accepted_prefix():
    def objective(scale):
        return quad.integrate(
            lambda x, live_scale: live_scale * jnp.sum(x, axis=-1),
            quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
            args=scale,
            **_deterministic_options(
                method=quad.Sobol(level=4),
                max_evaluations=16,
                gradient="replay",
            ),
        ).value

    assert jax.grad(objective)(2.0) == pytest.approx(objective(1.0))


def test_scrambled_sobol_returns_one_fixed_look_interval():
    result = quad.integrate(
        lambda x: jnp.prod(x, axis=-1),
        quad.Hyperrectangle(jnp.zeros(3), jnp.ones(3)),
        method=quad.ScrambledSobol(level=8, replicates=8),
        key=jax.random.key(19),
        epsabs=0.02,
        epsrel=0.0,
        max_evaluations=8 * 256,
        gradient="stop",
    )
    assert result.error.kind == quad.ErrorKind.CONFIDENCE_INTERVAL_HALF_WIDTH
    assert result.error.confidence_level == 0.95
    assert result.work.evaluations == 8 * 256
    assert result.work.replicates == 8
    assert result.work.levels == 8
    assert result.status in (
        quad.QuadStatus.CONVERGED,
        quad.QuadStatus.MAX_EVALUATIONS,
    )


def test_scrambled_sobol_is_reproducible_and_key_sensitive():
    options = dict(
        method=quad.ScrambledSobol(level=5, replicates=8),
        epsabs=0.0,
        epsrel=0.0,
        max_evaluations=8 * 32,
        gradient="stop",
    )
    domain = quad.Hyperrectangle(jnp.zeros(3), jnp.ones(3))
    first = quad.integrate(
        lambda x: jnp.exp(jnp.sum(x, axis=-1)),
        domain,
        key=jax.random.key(31),
        **options,
    )
    replay = quad.integrate(
        lambda x: jnp.exp(jnp.sum(x, axis=-1)),
        domain,
        key=jax.random.key(31),
        **options,
    )
    independent = quad.integrate(
        lambda x: jnp.exp(jnp.sum(x, axis=-1)),
        domain,
        key=jax.random.key(32),
        **options,
    )
    assert jnp.array_equal(first.value, replay.value)
    assert jnp.array_equal(first.error.estimate, replay.error.estimate)
    assert not jnp.array_equal(first.value, independent.value)


def test_scrambled_sobol_requires_key_scalar_real_payload_and_exact_budget():
    domain = quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2))
    method = quad.ScrambledSobol(level=4, replicates=8)
    options = dict(
        method=method,
        epsabs=0.1,
        epsrel=0.0,
        max_evaluations=128,
        gradient="stop",
    )
    with pytest.raises(TypeError, match="explicit JAX key"):
        quad.integrate(lambda x: jnp.sum(x, axis=-1), domain, **options)
    with pytest.raises(ValueError, match="scalar real"):
        quad.integrate(
            lambda x: x,
            domain,
            key=jax.random.key(1),
            **options,
        )
    with pytest.raises(ValueError, match="128 evaluations"):
        quad.integrate(
            lambda x: jnp.sum(x, axis=-1),
            domain,
            key=jax.random.key(1),
            **(options | {"max_evaluations": 127}),
        )


def test_scrambled_sobol_zero_volume_returns_zero_work():
    result = quad.integrate(
        lambda x: jnp.sum(x, axis=-1),
        quad.Hyperrectangle(jnp.asarray((0.0, 1.0)), jnp.asarray((2.0, 1.0))),
        method=quad.ScrambledSobol(level=4, replicates=8),
        key=jax.random.key(37),
        epsabs=0.1,
        epsrel=0.0,
        max_evaluations=128,
        gradient="stop",
    )
    assert result.value == 0.0
    assert result.work.evaluations == 0
    assert result.work.replicates == 0


def test_scrambled_sobol_zero_volume_skips_integer_point_generation(monkeypatch):
    qmc_module = importlib.import_module("jaxstro.quad.qmc")
    generation_calls = 0
    original = qmc_module.sobol_integer_points

    def record_generation(*args, **kwargs):
        nonlocal generation_calls
        generation_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(qmc_module, "sobol_integer_points", record_generation)
    result = quad.integrate(
        lambda x: jnp.sum(x, axis=-1),
        quad.Hyperrectangle(jnp.asarray((0.0, 1.0)), jnp.asarray((2.0, 1.0))),
        method=quad.ScrambledSobol(level=4, replicates=8),
        key=jax.random.key(41),
        epsabs=0.1,
        epsrel=0.0,
        max_evaluations=128,
        gradient="stop",
    )
    assert generation_calls == 0
    assert result.work.evaluations == 0


def test_scrambled_sobol_is_jittable_over_the_explicit_key():
    @jax.jit
    def solve(key):
        return quad.integrate(
            lambda x: jnp.exp(jnp.sum(x, axis=-1)),
            quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
            method=quad.ScrambledSobol(level=4, replicates=8),
            key=key,
            epsabs=0.1,
            epsrel=0.0,
            max_evaluations=128,
            gradient="stop",
        )

    result = solve(jax.random.key(43))
    assert jnp.isfinite(result.value)
    assert jnp.isfinite(result.error.estimate)
    assert result.work.evaluations == 128


def test_scrambled_sobol_nonfinite_replicate_returns_typed_failure():
    result = quad.integrate(
        lambda x: jnp.full(x.shape[0], jnp.nan),
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        method=quad.ScrambledSobol(level=4, replicates=8),
        key=jax.random.key(47),
        epsabs=0.1,
        epsrel=0.0,
        max_evaluations=128,
        gradient="stop",
    )
    assert result.status == quad.QuadStatus.NONFINITE_INTEGRAND
    assert jnp.isnan(result.value)
    assert jnp.isnan(result.error.estimate)


@pytest.mark.parametrize("nonfinite_integrand", (False, True))
def test_scrambled_sobol_invalid_interval_precedes_nonfinite_integrand(
    nonfinite_integrand,
):
    def integrand(x):
        values = jnp.sum(x, axis=-1)
        return jnp.full_like(values, jnp.nan) if nonfinite_integrand else values

    result = quad.integrate(
        integrand,
        quad.Hyperrectangle(
            jnp.zeros(2, dtype=jnp.float32),
            jnp.ones(2, dtype=jnp.float32),
        ),
        method=quad.ScrambledSobol(
            level=4,
            replicates=8,
            confidence_level=1.0e-12,
        ),
        key=jax.random.key(53),
        epsabs=0.1,
        epsrel=0.0,
        max_evaluations=128,
        gradient="stop",
    )
    assert result.status == quad.QuadStatus.INVALID_INPUT


def test_adaptive_schedule_requires_replicate_and_point_level_growth():
    with pytest.raises(ValueError, match="replicate growth"):
        quad.AdaptiveScrambledSobol(
            schedule=((6, 8), (7, 8)),
            estimate_bounds=(0.0, 1.0),
        )
    with pytest.raises(ValueError, match="point-level growth"):
        quad.AdaptiveScrambledSobol(
            schedule=((6, 8), (6, 16)),
            estimate_bounds=(0.0, 1.0),
        )


@pytest.mark.parametrize(
    "schedule",
    (
        ((6.0, 8), (7, 16)),
        ((6, 8.0), (7, 16)),
        ((6, 8), (5, 16)),
        ((6, 16), (7, 8)),
        ((6, 8), (6, 8), (7, 16)),
    ),
)
def test_adaptive_schedule_rejects_noninteger_or_nonprogressing_rows(schedule):
    with pytest.raises(ValueError, match="schedule"):
        quad.AdaptiveScrambledSobol(
            schedule=schedule,
            estimate_bounds=(0.0, 1.0),
        )


@pytest.mark.parametrize(
    "method",
    (
        lambda: quad.ScrambledSobol(level=4.0, replicates=8),
        lambda: quad.ScrambledSobol(level=4, replicates=8.0),
        lambda: quad.AdaptiveScrambledSobol(
            schedule=((3.0, 8), (4, 16)),
            estimate_bounds=(0.0, 1.0),
        ),
        lambda: quad.AdaptiveScrambledSobol(
            schedule=((3, 8.0), (4, 16)),
            estimate_bounds=(0.0, 1.0),
        ),
    ),
)
def test_randomized_qmc_declarations_reject_float_controls(method):
    with pytest.raises(ValueError):
        method()


def test_adaptive_declaration_requires_exactly_one_bound_contract():
    with pytest.raises(ValueError, match="exactly one"):
        quad.AdaptiveScrambledSobol(schedule=((3, 8), (4, 16)))
    with pytest.raises(ValueError, match="exactly one"):
        quad.AdaptiveScrambledSobol(
            schedule=((3, 8), (4, 16)),
            estimate_bounds=(0.0, 1.0),
            integrand_bounds=(0.0, 1.0),
        )


def test_adaptive_scrambled_sobol_reuses_points_and_replicates_physically():
    physical_points = 0

    def integrand(x):
        def record(_):
            nonlocal physical_points
            physical_points += x.shape[0]

        jax.debug.callback(record, x[0, 0])
        return jnp.exp(jnp.sum(x, axis=-1))

    result = quad.integrate(
        integrand,
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        method=quad.AdaptiveScrambledSobol(
            schedule=((3, 8), (4, 8), (4, 12), (5, 12)),
            estimate_bounds=(0.0, 10.0),
        ),
        key=jax.random.key(59),
        epsabs=0.0,
        epsrel=0.0,
        max_evaluations=12 * 32,
        gradient="stop",
    )
    assert result.status == quad.QuadStatus.MAX_EVALUATIONS
    assert result.work.evaluations == 12 * 32
    assert result.work.refinements == 3
    assert result.work.levels == 5
    assert result.work.replicates == 12
    assert physical_points == 12 * 32


def test_adaptive_scrambled_sobol_stops_physical_work_at_first_inspection():
    physical_points = 0

    def integrand(x):
        def record(_):
            nonlocal physical_points
            physical_points += x.shape[0]

        jax.debug.callback(record, x[0, 0])
        return jnp.exp(jnp.sum(x, axis=-1))

    result = quad.integrate(
        integrand,
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        method=quad.AdaptiveScrambledSobol(
            schedule=((3, 8), (4, 8), (4, 12), (5, 12)),
            estimate_bounds=(0.0, 10.0),
        ),
        key=jax.random.key(73),
        epsabs=100.0,
        epsrel=0.0,
        max_evaluations=12 * 32,
        gradient="stop",
    )
    assert result.status == quad.QuadStatus.CONVERGED
    assert result.work.evaluations == 8 * 8
    assert result.work.refinements == 0
    assert result.work.levels == 3
    assert result.work.replicates == 8
    assert physical_points == 8 * 8


def test_adaptive_zero_volume_skips_final_prefix_generation(monkeypatch):
    qmc_module = importlib.import_module("jaxstro.quad.qmc")
    generation_calls = 0
    original = qmc_module.sobol_integer_points

    def record_generation(*args, **kwargs):
        nonlocal generation_calls
        generation_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(qmc_module, "sobol_integer_points", record_generation)
    result = quad.integrate(
        lambda x: jnp.sum(x, axis=-1),
        quad.Hyperrectangle(jnp.asarray((0.0, 1.0)), jnp.asarray((2.0, 1.0))),
        method=quad.AdaptiveScrambledSobol(
            schedule=((3, 8), (4, 16)),
            estimate_bounds=(-10.0, 10.0),
        ),
        key=jax.random.key(79),
        epsabs=1.0,
        epsrel=0.0,
        max_evaluations=16 * 16,
        gradient="stop",
    )
    assert generation_calls == 0
    assert result.work.evaluations == 0


@pytest.mark.parametrize(
    ("lower", "upper", "expected_sign"),
    ((0.0, 1.0, 1.0), (1.0, 0.0, -1.0)),
)
def test_integrand_bounds_derive_valid_oriented_lebesgue_estimate_bounds(
    lower,
    upper,
    expected_sign,
):
    result = quad.integrate(
        lambda x: 1.0 + 0.5 * x[:, 0],
        quad.Hyperrectangle(jnp.asarray((lower,)), jnp.asarray((upper,))),
        method=quad.AdaptiveScrambledSobol(
            schedule=((3, 8), (4, 16)),
            integrand_bounds=(1.0, 1.5),
        ),
        key=jax.random.key(61),
        epsabs=10.0,
        epsrel=0.0,
        max_evaluations=16 * 16,
        gradient="stop",
    )
    assert result.status == quad.QuadStatus.CONVERGED
    assert jnp.sign(result.value) == expected_sign


def test_integrand_bound_derivation_rejects_weighted_measure():
    with pytest.raises(ValueError, match="direct estimate_bounds"):
        quad.integrate(
            lambda x: jnp.ones(x.shape[0]),
            quad.Hyperrectangle(jnp.zeros(1), jnp.ones(1)),
            method=quad.AdaptiveScrambledSobol(
                schedule=((3, 8), (4, 16)),
                integrand_bounds=(0.0, 1.0),
            ),
            measure=quad.WeightedMeasure(
                lambda x, _args: jnp.ones(x.shape[0]),
                density_unit=dimensionless,
            ),
            key=jax.random.key(67),
            epsabs=1.0,
            epsrel=0.0,
            max_evaluations=16 * 16,
            gradient="stop",
        )


def test_direct_estimate_bounds_support_a_signed_weighted_measure():
    result = quad.integrate(
        lambda x: jnp.ones(x.shape[0]),
        quad.Hyperrectangle(jnp.zeros(1), jnp.ones(1)),
        method=quad.AdaptiveScrambledSobol(
            schedule=((3, 8), (4, 16)),
            estimate_bounds=(-1.0, -1.0),
        ),
        measure=quad.WeightedMeasure(
            lambda x, _args: -jnp.ones(x.shape[0]),
            density_unit=dimensionless,
        ),
        key=jax.random.key(83),
        epsabs=1.0,
        epsrel=0.0,
        max_evaluations=16 * 16,
        gradient="stop",
    )
    assert result.status == quad.QuadStatus.CONVERGED
    assert result.value == -1.0


def test_adaptive_scrambled_sobol_fails_closed_on_bound_violation():
    result = quad.integrate(
        lambda x: 2.0 + jnp.sum(x, axis=-1),
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        method=quad.AdaptiveScrambledSobol(
            schedule=((3, 8), (4, 16)),
            integrand_bounds=(0.0, 1.0),
        ),
        key=jax.random.key(71),
        epsabs=1.0,
        epsrel=0.0,
        max_evaluations=16 * 16,
        gradient="stop",
    )
    assert result.status == quad.QuadStatus.INVALID_INPUT
    assert jnp.isnan(result.value)
