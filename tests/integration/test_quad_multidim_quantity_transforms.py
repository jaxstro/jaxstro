import jax
import jax.numpy as jnp
import pytest

from jaxstro import quad
from jaxstro.quantity import (
    Msun,
    Myr,
    Quantity,
    cm,
    dimensionless,
    pc,
    s,
)
from jaxstro.quantity.errors import DimensionError


def _constant_mass_integral(length_value, length_unit, time_value, time_unit):
    domain = quad.Hyperrectangle.from_axes(
        (
            quad.Axis(Quantity(0.0, length_unit), Quantity(length_value, length_unit)),
            quad.Axis(Quantity(0.0, time_unit), Quantity(time_value, time_unit)),
        )
    )
    return quad.integrate(
        lambda x: Quantity(jnp.ones(x.shape[:-1]), Msun),
        domain,
        method=quad.TensorProduct(quad.GaussianRule(2)),
        epsabs=Quantity(1e-10, Msun * length_unit * time_unit),
        epsrel=1e-10,
        max_evaluations=4,
    )


def test_heterogeneous_quantity_result_is_representation_invariant():
    parsec_myr = _constant_mass_integral(2.0, pc, 3.0, Myr)
    cgs = _constant_mass_integral(
        Quantity(2.0, pc).to_value(cm),
        cm,
        Quantity(3.0, Myr).to_value(s),
        s,
    )
    target = Msun * pc * Myr

    assert jnp.allclose(
        parsec_myr.value.to_value(target),
        cgs.value.to_value(target),
        rtol=2e-12,
        atol=2e-12,
    )
    assert parsec_myr.status == cgs.status
    assert parsec_myr.work == cgs.work


def test_reversed_quantity_axis_preserves_oriented_integral():
    forward = _constant_mass_integral(2.0, pc, 3.0, Myr)
    reversed_domain = quad.Hyperrectangle.from_axes(
        (
            quad.Axis(Quantity(2.0, pc), Quantity(0.0, pc)),
            quad.Axis(Quantity(0.0, Myr), Quantity(3.0, Myr)),
        )
    )
    reversed_result = quad.integrate(
        lambda x: Quantity(jnp.ones(x.shape[:-1]), Msun),
        reversed_domain,
        method=quad.TensorProduct(quad.GaussianRule(2)),
        epsabs=Quantity(1e-10, Msun * pc * Myr),
        epsrel=1e-10,
        max_evaluations=4,
    )

    assert jnp.allclose(reversed_result.value.value, -forward.value.value)


def test_dimensionless_hyperrectangle_quantity_mode_uses_coordinate_point():
    result = quad.integrate(
        lambda x: Quantity(
            jnp.ones(x.shape[:-1]) + 0.0 * x.axis(0).to_value(dimensionless),
            Msun,
        ),
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        method=quad.TensorProduct(quad.GaussianRule(2)),
        epsabs=Quantity(1e-10, Msun),
        epsrel=1e-10,
        max_evaluations=4,
    )

    assert result.value.unit == Msun
    assert jnp.allclose(result.value.value, 1.0)


def test_full_weighted_measure_receives_heterogeneous_coordinate_point():
    domain = quad.Hyperrectangle.from_axes(
        (
            quad.Axis(Quantity(0.0, pc), Quantity(2.0, pc)),
            quad.Axis(Quantity(0.0, Myr), Quantity(3.0, Myr)),
        )
    )
    measure = quad.WeightedMeasure(
        lambda x, _args: Quantity(
            1.0 + x.axis(0).to_value(pc) / 2.0,
            dimensionless,
        ),
        density_unit=dimensionless,
    )
    result = quad.integrate(
        lambda x: Quantity(jnp.ones(x.shape[:-1]), Msun),
        domain,
        measure=measure,
        method=quad.TensorProduct(quad.GaussianRule(3)),
        epsabs=Quantity(1e-10, Msun * pc * Myr),
        epsrel=1e-10,
        max_evaluations=9,
    )

    assert result.value.unit == Msun * pc * Myr
    assert jnp.allclose(result.value.value, 9.0, rtol=2e-12, atol=2e-12)


def test_product_measure_wraps_each_axis_and_multiplies_density_units():
    domain = quad.Hyperrectangle.from_axes(
        (
            quad.Axis(Quantity(0.0, pc), Quantity(2.0, pc)),
            quad.Axis(Quantity(0.0, Myr), Quantity(3.0, Myr)),
        )
    )
    measure = quad.ProductMeasure(
        (
            quad.WeightedMeasure(
                lambda x, _args: Quantity(
                    jnp.ones_like(x.value) / 2.0,
                    dimensionless / pc,
                ),
                density_unit=dimensionless / pc,
                normalized=True,
            ),
            quad.WeightedMeasure(
                lambda x, _args: Quantity(
                    jnp.ones_like(x.value) / 3.0,
                    dimensionless / Myr,
                ),
                density_unit=dimensionless / Myr,
                normalized=True,
            ),
        )
    )
    result = quad.integrate(
        lambda x: Quantity(jnp.ones(x.shape[:-1]), Msun),
        domain,
        measure=measure,
        method=quad.AdaptiveCubature(),
        epsabs=Quantity(1e-8, Msun),
        epsrel=1e-8,
        max_evaluations=256,
        max_regions=8,
    )

    assert result.value.unit == Msun
    assert jnp.allclose(result.value.value, 1.0, rtol=2e-10, atol=2e-10)


def test_product_measure_density_and_integrand_share_live_args():
    domain = quad.Hyperrectangle.from_axes(
        (
            quad.Axis(Quantity(0.0, pc), Quantity(2.0, pc)),
            quad.Axis(Quantity(0.0, Myr), Quantity(3.0, Myr)),
        )
    )
    measure = quad.ProductMeasure(
        (
            quad.WeightedMeasure(
                lambda x, live: Quantity(
                    live["rate"] * jnp.ones_like(x.value),
                    dimensionless / pc,
                ),
                density_unit=dimensionless / pc,
            ),
            quad.LebesgueMeasure(),
        )
    )

    def objective(parameters):
        return quad.integrate(
            lambda x, live: Quantity(
                live["mass"] * jnp.ones(x.shape[:-1]),
                Msun,
            ),
            domain,
            args=parameters,
            measure=measure,
            method=quad.TensorProduct(quad.GaussianRule(2)),
            epsabs=Quantity(1e-10, Msun * Myr),
            epsrel=1e-10,
            max_evaluations=4,
        ).value.to_value(Msun * Myr)

    parameters = {"mass": jnp.asarray(2.0), "rate": jnp.asarray(0.5)}
    derivative = jax.grad(objective)(parameters)

    assert objective(parameters) == pytest.approx(6.0)
    assert derivative["mass"] == pytest.approx(3.0)
    assert derivative["rate"] == pytest.approx(12.0)


def test_moving_quantity_bound_replay_differentiates_raw_magnitude():
    output_unit = Msun * pc * Myr

    def objective(upper_pc):
        domain = quad.Hyperrectangle.from_axes(
            (
                quad.Axis(Quantity(0.0, pc), Quantity(upper_pc, pc)),
                quad.Axis(Quantity(0.0, Myr), Quantity(3.0, Myr)),
            )
        )
        return quad.integrate(
            lambda x: Quantity(jnp.ones(x.shape[:-1]), Msun),
            domain,
            method=quad.TensorProduct(quad.GaussianRule(2)),
            epsabs=Quantity(1e-10, output_unit),
            epsrel=1e-10,
            max_evaluations=4,
        ).value.to_value(output_unit)

    derivative = jax.jit(jax.grad(objective))(jnp.asarray(2.0))
    batched = jax.jit(jax.vmap(jax.grad(objective)))(jnp.asarray([1.0, 2.0, 4.0]))

    assert derivative == pytest.approx(3.0)
    assert jnp.allclose(batched, 3.0)


def test_adaptive_randomized_quantity_bounds_are_normalized():
    result_unit = Msun * pc * Myr
    domain = quad.Hyperrectangle.from_axes(
        (
            quad.Axis(Quantity(0.0, pc), Quantity(2.0, pc)),
            quad.Axis(Quantity(0.0, Myr), Quantity(3.0, Myr)),
        )
    )
    method = quad.AdaptiveScrambledSobol(
        schedule=((3, 8), (4, 16)),
        estimate_bounds=(
            Quantity(0.0, result_unit),
            Quantity(10.0, result_unit),
        ),
    )
    result = quad.integrate(
        lambda x: Quantity(jnp.ones(x.shape[:-1]), Msun),
        domain,
        method=method,
        key=jax.random.key(7),
        epsabs=Quantity(0.1, result_unit),
        epsrel=0.0,
        max_evaluations=256,
    )

    assert result.value.unit == result_unit
    assert jnp.allclose(result.value.value, 6.0, rtol=2e-12, atol=2e-12)


def test_adaptive_randomized_quantity_bounds_must_match_expected_unit():
    domain = quad.Hyperrectangle.from_axes(
        (
            quad.Axis(Quantity(0.0, pc), Quantity(2.0, pc)),
            quad.Axis(Quantity(0.0, Myr), Quantity(3.0, Myr)),
        )
    )
    method = quad.AdaptiveScrambledSobol(
        schedule=((3, 8), (4, 16)),
        estimate_bounds=(Quantity(0.0, Msun), Quantity(10.0, Msun)),
    )
    with pytest.raises(DimensionError):
        quad.integrate(
            lambda x: Quantity(jnp.ones(x.shape[:-1]), Msun),
            domain,
            method=method,
            key=jax.random.key(7),
            epsabs=Quantity(0.1, Msun * pc * Myr),
            epsrel=0.0,
            max_evaluations=256,
        )
