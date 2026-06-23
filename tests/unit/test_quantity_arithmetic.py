"""Tests for quantity arithmetic and conversion."""

import jax.numpy as jnp
import pytest

from jaxstro import quantity as q
from jaxstro.quantity.errors import DimensionError


def test_quantity_constructor_and_value_times_unit():
    length = q.Quantity(jnp.array([1.0, 2.0]), q.cm)
    via_unit = 3.0 * q.cm

    assert length.unit is q.cm
    assert jnp.all(length.value == jnp.array([1.0, 2.0]))
    assert via_unit.unit is q.cm
    assert via_unit.value == 3.0


def test_addition_and_subtraction_convert_right_to_left_unit():
    length = 1.0 * q.m
    extra = 25.0 * q.cm

    total = length + extra
    diff = length - extra

    assert total.unit is q.m
    assert total.value == pytest.approx(1.25)
    assert diff.unit is q.m
    assert diff.value == pytest.approx(0.75)


def test_incompatible_addition_raises_structured_dimension_error():
    with pytest.raises(DimensionError) as exc:
        (1.0 * q.cm) + (1.0 * q.s)

    assert exc.value.operation == "add"
    assert exc.value.expected == q.cm.dimensions
    assert exc.value.actual == q.s.dimensions


def test_multiplication_division_and_powers_combine_units():
    speed = (10.0 * q.km) / (2.0 * q.s)
    area = (2.0 * q.cm) ** 2
    density = (2.0 * q.g) / (4.0 * q.cm**3)

    assert speed.unit == q.km / q.s
    assert speed.value == 5.0
    assert area.unit == q.cm**2
    assert area.value == 4.0
    assert density.unit == q.g / q.cm**3
    assert density.value == 0.5


def test_raw_scalar_rules():
    length = 2.0 * q.cm
    dimensionless = q.Quantity(2.0, q.dimensionless)

    assert (3.0 * length).value == 6.0
    assert (length * 3.0).unit is q.cm
    assert (dimensionless + 1.0).unit is q.dimensionless
    assert (dimensionless + 1.0).value == 3.0

    with pytest.raises(DimensionError):
        length + 1.0


def test_conversion_helpers_and_one_unit_per_array():
    values = jnp.array([100.0, 250.0]) * q.cm
    meters = values.to(q.m)

    assert meters.unit is q.m
    assert jnp.allclose(meters.value, jnp.array([1.0, 2.5]))
    assert jnp.allclose(values.to_value(q.m), jnp.array([1.0, 2.5]))
    assert values.to_cgs().unit is q.cm
    assert jnp.allclose(values.to_cgs_value(), jnp.array([100.0, 250.0]))


def test_incompatible_conversion_raises():
    with pytest.raises(DimensionError) as exc:
        (1.0 * q.cm).to(q.s)

    assert exc.value.operation == "convert"
