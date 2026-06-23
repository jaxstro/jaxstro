"""Tests for astrophysical quantity bases."""

import pytest

from jaxstro import quantity as q
from jaxstro.quantity.errors import UnitRegistryError


def test_named_bases_and_aliases_exist():
    for name in (
        "CGS",
        "SI",
        "STELLAR",
        "PLANETARY",
        "STAR_CLUSTER",
        "CLOSE_BINARY",
        "WIDE_BINARY",
        "COMPACT_BINARY",
    ):
        assert hasattr(q.bases, name)

    assert q.bases.DYNAMICAL is q.bases.STAR_CLUSTER
    assert q.bases.EXOPLANET is q.bases.PLANETARY


def test_role_aware_stellar_basis_choices():
    basis = q.bases.STELLAR

    assert basis.unit_for(role="stellar_mass") is q.Msun
    assert basis.unit_for(role="stellar_radius") is q.Rsun
    assert basis.unit_for(role="luminosity") is q.Lsun
    assert basis.unit_for(role="time") is q.Myr
    assert basis.unit_for(role="velocity") == q.km / q.s


def test_role_aware_planetary_basis_choices():
    basis = q.bases.PLANETARY

    assert basis.unit_for(role="stellar_mass") is q.Msun
    assert basis.unit_for(role="planet_mass") is q.g
    assert basis.unit_for(role="orbit") is q.AU
    assert basis.unit_for(role="orbital_separation") is q.AU
    assert basis.unit_for(role="time") is q.yr
    assert basis.unit_for(role="velocity") == q.km / q.s


def test_dimension_defaults_when_role_is_omitted():
    assert q.bases.CGS.unit_for(q.Quantity(1.0, q.km)) is q.cm
    assert q.bases.SI.unit_for(q.Quantity(1.0, q.cm)) is q.m
    assert q.bases.STAR_CLUSTER.unit_for(q.Quantity(1.0, q.s)) is q.Myr


def test_quantity_to_basis():
    radius = q.Quantity(2.0 * q.Rsun.scale_to_cgs, q.cm)
    converted = radius.to_basis(q.bases.STELLAR, role="stellar_radius")

    assert converted.unit is q.Rsun
    assert converted.value == pytest.approx(2.0)


def test_unknown_role_errors_with_suggestions():
    with pytest.raises(UnitRegistryError) as exc:
        q.bases.STELLAR.unit_for(role="stellar_mas")

    assert "stellar_mass" in exc.value.expected
    assert "Did you mean" in str(exc.value)
