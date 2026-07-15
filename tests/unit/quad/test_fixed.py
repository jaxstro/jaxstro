"""Public fixed quadrature evaluator contracts."""

import jax.numpy as jnp
import pytest

from jaxstro import quantity
from jaxstro.quad import (
    ClenshawCurtisRule,
    GaussianRule,
    Infinite,
    Interval,
    LaguerreMeasure,
    LebesgueMeasure,
    RightInfinite,
    StandardNormalMeasure,
    TanhSinhRule,
    WeightedMeasure,
    fixed,
)


def test_fixed_gaussian_integrates_polynomial() -> None:
    got = fixed(lambda x: x**4, Interval(-1.0, 1.0), rule=GaussianRule(3))
    assert jnp.allclose(got, 2.0 / 5.0, rtol=1e-12, atol=1e-12)


def test_fixed_vector_payload_and_explicit_args() -> None:
    def fun(x, args):
        return jnp.stack((args * x, x * x), axis=-1)

    got = fixed(
        fun,
        Interval(-1.0, 1.0),
        args=3.0,
        rule=GaussianRule(8),
    )
    assert jnp.allclose(got, jnp.asarray([0.0, 2.0 / 3.0]), atol=2e-12)


def test_fixed_breakpoints_sum_vectorized_segments() -> None:
    domain = Interval(0.0, 1.0, breakpoints=(0.25, 0.75))
    got = fixed(lambda x: x**3, domain, rule=GaussianRule(2))
    assert jnp.allclose(got, 0.25, rtol=1e-12, atol=1e-12)


def test_fixed_preserves_reversed_orientation() -> None:
    forward = fixed(lambda x: x**2, Interval(0.0, 2.0), rule=GaussianRule(4))
    reverse = fixed(lambda x: x**2, Interval(2.0, 0.0), rule=GaussianRule(4))
    assert jnp.allclose(reverse, -forward)


def test_zero_width_returns_exact_zero_for_nonfinite_integrand() -> None:
    got = fixed(
        lambda x: jnp.full_like(x, jnp.nan),
        Interval(2.0, 2.0),
        rule=GaussianRule(4),
    )
    assert jnp.array_equal(got, 0.0)


def test_weighted_measure_is_applied_exactly_once() -> None:
    measure = WeightedMeasure(
        lambda x, args: args * x,
        density_unit=quantity.dimensionless,
    )
    got = fixed(
        lambda x, _args: x,
        Interval(0.0, 1.0),
        args=2.0,
        rule=ClenshawCurtisRule(9),
        measure=measure,
    )
    assert jnp.allclose(got, 2.0 / 3.0, rtol=2e-11, atol=2e-11)


def test_gaussian_laguerre_and_normal_use_natural_supports() -> None:
    laguerre = fixed(
        lambda x: jnp.ones_like(x),
        RightInfinite(3.0),
        rule=GaussianRule(8),
        measure=LaguerreMeasure(),
    )
    normal = fixed(
        lambda x: x**2,
        Infinite(),
        rule=GaussianRule(8),
        measure=StandardNormalMeasure(),
    )
    assert jnp.allclose(laguerre, 1.0, rtol=2e-12, atol=2e-12)
    assert jnp.allclose(normal, 1.0, rtol=2e-12, atol=2e-12)


def test_fixed_tanh_sinh_handles_infinite_domain() -> None:
    got = fixed(
        lambda x: jnp.exp(-(x**2)),
        Infinite(),
        rule=TanhSinhRule(6),
    )
    assert jnp.allclose(got, jnp.sqrt(jnp.pi), rtol=2e-9, atol=2e-9)


def test_fixed_rejects_unsupported_structural_pairings() -> None:
    with pytest.raises(TypeError, match="finite Interval"):
        fixed(
            lambda x: x,
            Infinite(),
            rule=GaussianRule(4),
            measure=LebesgueMeasure(),
        )
    with pytest.raises(TypeError, match="WeightedMeasure"):
        fixed(
            lambda x: x,
            Interval(-1.0, 1.0),
            rule=ClenshawCurtisRule(5),
            measure=StandardNormalMeasure(),
        )


def test_fixed_rejects_payload_without_node_axis() -> None:
    with pytest.raises(ValueError, match="leading node axis"):
        fixed(lambda _x: jnp.asarray(1.0), Interval(0.0, 1.0), rule=GaussianRule(3))


def test_value_invalid_interval_returns_nan() -> None:
    got = fixed(
        lambda x: x,
        Interval(0.0, 1.0, breakpoints=(2.0,)),
        rule=GaussianRule(3),
    )
    assert jnp.isnan(got)
