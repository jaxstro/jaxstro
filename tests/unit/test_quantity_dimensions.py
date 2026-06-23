"""Tests for exact quantity dimensions."""

from fractions import Fraction

import pytest

from jaxstro.quantity import dimensions as d
from jaxstro.quantity.errors import DimensionError


def test_base_dimensions_have_exact_vectors():
    assert d.mass.exponents == (
        Fraction(1),
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
    )
    assert d.length.exponents == (
        Fraction(0),
        Fraction(1),
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
    )
    assert d.time.exponents == (
        Fraction(0),
        Fraction(0),
        Fraction(1),
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
    )
    assert d.temperature.exponents == (
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(1),
        Fraction(0),
        Fraction(0),
        Fraction(0),
    )
    assert d.current.exponents == (
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(1),
        Fraction(0),
        Fraction(0),
    )
    assert d.amount.exponents == (
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(1),
        Fraction(0),
    )
    assert d.luminosity.exponents == (
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(1),
    )


def test_dimensionless_singleton_is_stable():
    assert d.dimensionless.is_dimensionless
    assert d.Dimension.dimensionless() is d.dimensionless
    assert d.dimensionless * d.length == d.length
    assert d.length / d.length == d.dimensionless


def test_dimension_arithmetic_and_rational_powers():
    velocity = d.length / d.time
    acceleration = velocity / d.time
    area_root = (d.length**2) ** Fraction(1, 2)

    assert velocity == d.velocity
    assert acceleration == d.acceleration
    assert area_root == d.length
    assert (d.energy ** Fraction(1, 2)).exponents == (
        Fraction(1, 2),
        Fraction(1),
        Fraction(-1),
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
    )


def test_readable_derived_dimensions():
    assert d.velocity == d.Dimension.from_powers(length=1, time=-1)
    assert d.acceleration == d.Dimension.from_powers(length=1, time=-2)
    assert d.energy == d.Dimension.from_powers(mass=1, length=2, time=-2)
    assert d.power == d.Dimension.from_powers(mass=1, length=2, time=-3)
    assert d.power.is_compatible_with(d.energy / d.time)


def test_hash_and_equality_stability():
    dims = {d.length / d.time, d.velocity, d.mass}
    assert len(dims) == 2
    assert hash(d.velocity) == hash(d.Dimension.from_powers(length=1, time=-1))


def test_float_exponents_are_rejected():
    with pytest.raises(DimensionError, match="exact rational"):
        d.length**0.5

    with pytest.raises(DimensionError, match="exact rational"):
        d.Dimension.from_powers(length=0.5)
