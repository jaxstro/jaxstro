import jax.numpy as jnp
import pytest

from jaxstro import quad
from jaxstro.quantity import Msun, Myr, Quantity, cm, pc, s
from jaxstro.quantity.errors import DimensionError


def test_coordinate_point_exposes_each_static_axis_unit():
    point = quad.CoordinatePoint(
        values=jnp.array([[2.0, 3.0]]),
        units=(pc, Myr),
    )

    assert point.shape == (1, 2)
    assert point.dimension == 2
    assert point.axis(0).unit == pc
    assert point.axis(1).unit == Myr


def test_coordinate_point_can_stack_compatible_axes_as_one_quantity():
    point = quad.CoordinatePoint(
        values=jnp.array([[2.0, 3.0]]),
        units=(pc, pc),
    )

    stacked = point.as_quantity(pc)

    assert stacked.unit == pc
    assert jnp.array_equal(stacked.value, point.values)


def test_axis_requires_scalar_compatible_quantity_bounds():
    with pytest.raises(DimensionError, match="compatible units"):
        quad.Axis(Quantity(0.0, pc), Quantity(1.0, Myr))
    with pytest.raises(ValueError, match="scalar"):
        quad.Axis(Quantity(jnp.zeros(2), pc), Quantity(1.0, pc))
    with pytest.raises(TypeError, match="Quantity"):
        quad.Axis(0.0, Quantity(1.0, pc))


def test_hyperrectangle_converts_each_upper_bound_to_its_axis_unit():
    domain = quad.Hyperrectangle.from_axes(
        (
            quad.Axis(Quantity(0.0, pc), Quantity(2.0, pc).to(cm)),
            quad.Axis(Quantity(0.0, Myr), Quantity(3.0, Myr)),
        )
    )

    assert domain.axis_units == (pc, Myr)
    assert jnp.allclose(domain.lower, jnp.asarray([0.0, 0.0]))
    assert jnp.allclose(domain.upper, jnp.asarray([2.0, 3.0]))


def test_heterogeneous_axes_produce_product_result_unit():
    domain = quad.Hyperrectangle.from_axes(
        (
            quad.Axis(Quantity(0.0, pc), Quantity(2.0, pc)),
            quad.Axis(Quantity(0.0, Myr), Quantity(3.0, Myr)),
        )
    )
    result = quad.integrate(
        lambda x: Quantity(jnp.ones(x.shape[:-1]), Msun),
        domain,
        method=quad.TensorProduct(quad.GaussianRule(2)),
        epsabs=Quantity(1e-10, Msun * pc * Myr),
        epsrel=1e-10,
        max_evaluations=4,
    )

    assert result.value.unit == Msun * pc * Myr
    assert jnp.allclose(result.value.value, 6.0)


def test_product_measure_requires_finite_one_dimensional_components():
    with pytest.raises(ValueError, match="at least one"):
        quad.ProductMeasure(())
    with pytest.raises(TypeError, match="LebesgueMeasure or WeightedMeasure"):
        quad.ProductMeasure((quad.StandardNormalMeasure(),))


def test_product_measure_component_count_matches_domain_dimension():
    domain = quad.Hyperrectangle.from_axes(
        (
            quad.Axis(Quantity(0.0, pc), Quantity(1.0, pc)),
            quad.Axis(Quantity(0.0, s), Quantity(1.0, s)),
        )
    )
    with pytest.raises(ValueError, match="component per coordinate axis"):
        quad.integrate(
            lambda x: Quantity(jnp.ones(x.shape[:-1]), Msun),
            domain,
            measure=quad.ProductMeasure((quad.LebesgueMeasure(),)),
            method=quad.TensorProduct(quad.GaussianRule(2)),
            epsabs=Quantity(1e-10, Msun * pc * s),
            epsrel=1e-10,
            max_evaluations=4,
        )
