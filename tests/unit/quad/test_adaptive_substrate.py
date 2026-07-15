"""Shared adaptive reference-partition and transformed-integrand contracts."""

import jax
import jax.numpy as jnp
import pytest

from jaxstro import quantity
from jaxstro.quad import (
    Infinite,
    Interval,
    JacobiMeasure,
    LeftInfinite,
    RightInfinite,
    WeightedMeasure,
)
from jaxstro.quad._adaptive import (
    infer_payload_zero,
    reference_partition,
    transformed_integrand,
)


def test_reference_partition_normalizes_breakpoints_independently_of_orientation() -> (
    None
):
    forward = reference_partition(Interval(0.0, 10.0, breakpoints=(7.0, 2.0)))
    reverse = reference_partition(Interval(10.0, 0.0, breakpoints=(2.0, 7.0)))
    expected_lower = jnp.asarray([-1.0, -0.6, 0.4])
    expected_upper = jnp.asarray([-0.6, 0.4, 1.0])
    assert jnp.allclose(forward.lower, expected_lower)
    assert jnp.allclose(forward.upper, expected_upper)
    assert jnp.allclose(reverse.lower, expected_lower)
    assert jnp.allclose(reverse.upper, expected_upper)
    assert forward.valid
    assert reverse.valid


def test_reference_partition_stops_breakpoint_motion() -> None:
    derivative = jax.grad(
        lambda breakpoint: jnp.sum(
            reference_partition(Interval(0.0, 1.0, breakpoints=(breakpoint,))).upper
        )
    )(jnp.asarray(0.3))
    assert jnp.array_equal(derivative, 0.0)


@pytest.mark.parametrize(
    "domain",
    [RightInfinite(2), LeftInfinite(2), Infinite()],
)
def test_improper_reference_partition_has_one_normalized_region(domain) -> None:
    partition = reference_partition(domain)
    assert jnp.array_equal(partition.lower, jnp.asarray([-1.0]))
    assert jnp.array_equal(partition.upper, jnp.asarray([1.0]))
    assert jnp.issubdtype(partition.lower.dtype, jnp.inexact)
    assert partition.valid


@pytest.mark.parametrize(
    "domain",
    [
        Interval(0.0, 1.0, breakpoints=(0.5, 0.5)),
        Interval(0.0, 1.0, breakpoints=(2.0,)),
        RightInfinite(jnp.inf),
        LeftInfinite(jnp.nan),
    ],
)
def test_reference_partition_reports_dynamic_invalid_domains(domain) -> None:
    assert not reference_partition(domain).valid


def test_transformed_integrand_composes_local_and_global_finite_maps() -> None:
    result = transformed_integrand(
        lambda x: x**2,
        Interval(2.0, 6.0),
        jnp.asarray([-1.0, 0.0, 1.0]),
        region_lower=-1.0,
        region_upper=0.0,
    )
    assert jnp.allclose(result.reference, jnp.asarray([-1.0, -0.5, 0.0]))
    assert jnp.allclose(result.x, jnp.asarray([2.0, 3.0, 4.0]))
    assert jnp.allclose(result.jacobian, 1.0)
    assert jnp.allclose(result.values, result.x**2)
    assert result.valid
    assert not result.nonfinite


def test_transformed_integrand_preserves_reversed_orientation() -> None:
    nodes = jnp.asarray([-0.5, 0.5])
    forward = transformed_integrand(lambda x: x**2, Interval(0.0, 2.0), nodes)
    reverse = transformed_integrand(lambda x: x**2, Interval(2.0, 0.0), nodes)
    assert jnp.array_equal(reverse.x, forward.x)
    assert jnp.array_equal(reverse.jacobian, forward.jacobian)
    assert jnp.array_equal(reverse.values, -forward.values)


def test_transformed_integrand_applies_density_once_and_preserves_payload() -> None:
    measure = WeightedMeasure(
        lambda x, args: args * (1.0 + x),
        density_unit=quantity.dimensionless,
    )

    def fun(x, args):
        return jnp.stack((x + 1j * args, x**2), axis=-1)

    result = transformed_integrand(
        fun,
        Interval(0.0, 1.0),
        jnp.asarray([-1.0, 0.0, 1.0]),
        args=2.0,
        measure=measure,
    )
    expected_integrand = fun(result.x, 2.0)
    expected_density = 2.0 * (1.0 + result.x)
    assert result.values.shape == (3, 2)
    assert jnp.allclose(
        result.values, expected_integrand * expected_density[:, None] * 0.5
    )


@pytest.mark.parametrize(
    ("domain", "fun"),
    [
        (RightInfinite(0.0), lambda x: jnp.exp(-x)),
        (LeftInfinite(0.0), lambda x: jnp.exp(x)),
        (Infinite(), lambda x: jnp.exp(-(x**2))),
    ],
)
def test_transformed_integrand_supports_every_improper_domain(domain, fun) -> None:
    result = transformed_integrand(fun, domain, jnp.asarray([-0.75, 0.0, 0.75]))
    assert result.valid
    assert not result.nonfinite
    assert jnp.all(jnp.isfinite(result.values))
    assert jnp.all(result.jacobian > 0.0)


def test_transformed_integrand_reports_nonfinite_contributions() -> None:
    result = transformed_integrand(
        lambda x: jnp.where(x > 0.0, jnp.nan, x),
        Interval(-1.0, 1.0),
        jnp.asarray([-0.5, 0.5]),
    )
    assert result.nonfinite


def test_transformed_integrand_reports_nonfinite_weighted_density() -> None:
    measure = WeightedMeasure(
        lambda x, _args: jnp.where(x > 0.0, jnp.inf, 1.0),
        density_unit=quantity.dimensionless,
    )
    result = transformed_integrand(
        lambda x: jnp.ones_like(x),
        Interval(-1.0, 1.0),
        jnp.asarray([-0.5, 0.5]),
        measure=measure,
    )
    assert result.valid
    assert result.nonfinite


def test_invalid_domain_remains_distinct_from_nonfinite_mapped_data() -> None:
    result = transformed_integrand(
        lambda x: jnp.exp(-x),
        RightInfinite(jnp.inf),
        jnp.asarray([-0.5, 0.5]),
    )
    assert not result.valid
    assert result.nonfinite


def test_transformed_integrand_rejects_classical_measure_reinterpretation() -> None:
    with pytest.raises(TypeError, match="LebesgueMeasure or WeightedMeasure"):
        transformed_integrand(
            lambda x: x,
            Interval(-1.0, 1.0),
            jnp.asarray([0.0]),
            measure=JacobiMeasure(0.0, 0.0),
        )


def test_transformed_integrand_rejects_payload_without_node_axis() -> None:
    with pytest.raises(ValueError, match="leading node axis"):
        transformed_integrand(
            lambda _x: jnp.asarray(1.0),
            Interval(-1.0, 1.0),
            jnp.asarray([-0.5, 0.5]),
        )


def test_zero_width_payload_inference_does_not_evaluate_nonfinite_values() -> None:
    zero = infer_payload_zero(
        lambda x: jnp.full((x.shape[0], 2, 3), jnp.nan),
        args=(),
        node_count=5,
        node_dtype=jnp.float64,
    )
    assert zero.shape == (2, 3)
    assert jnp.array_equal(zero, jnp.zeros((2, 3)))


def test_transformed_integrand_jit_supports_dynamic_bounds_regions_and_args() -> None:
    evaluate = jax.jit(
        lambda lower, upper, split, scale: (
            transformed_integrand(
                lambda x, args: args * x,
                Interval(lower, upper),
                jnp.asarray([-0.5, 0.5]),
                region_lower=-1.0,
                region_upper=split,
                args=scale,
            ).values
        )
    )
    got = evaluate(0.0, 2.0, 0.0, 3.0)
    assert jnp.all(jnp.isfinite(got))
