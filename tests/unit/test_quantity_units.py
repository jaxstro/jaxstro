"""Tests for quantity unit algebra and documented units."""

from fractions import Fraction

import pytest

from jaxstro import constants as C
from jaxstro import quantity as q
from jaxstro.quantity import dimensions as d
from jaxstro.quantity.errors import DimensionError


def test_core_units_have_expected_dimensions_and_cgs_scales():
    assert q.cm.dimensions == d.length
    assert q.cm.scale_to_cgs == 1.0
    assert q.m.scale_to_cgs == 100.0
    assert q.km.scale_to_cgs == C.KM_CM
    assert q.g.dimensions == d.mass
    assert q.kg.scale_to_cgs == 1000.0
    assert q.s.dimensions == d.time
    assert q.day.scale_to_cgs == 86400.0
    assert q.yr.scale_to_cgs == C.YR_S
    assert q.K.dimensions == d.temperature


def test_derived_and_angle_units():
    assert q.erg.dimensions == d.energy
    assert q.erg.scale_to_cgs == 1.0
    assert q.Hz.dimensions == d.time**-1
    assert q.rad.dimensions == d.dimensionless
    assert q.rad.metadata.get("semantic") == "angle"
    assert q.deg.scale_to_cgs == pytest.approx(3.141592653589793 / 180.0)


def test_angle_semantics_are_part_of_unit_identity():
    assert q.rad != q.dimensionless
    assert len({q.rad, q.dimensionless}) == 2
    assert q.rad.is_compatible_with(q.deg)
    assert not q.rad.is_compatible_with(q.dimensionless)
    with pytest.raises(TypeError):
        q.rad.metadata["semantic"] = "other"
    with pytest.raises(TypeError, match="JSON scalar"):
        q.Unit("mutable", 1.0, d.dimensionless, metadata={"semantic": []})
    with pytest.raises(ValueError, match="Angle units"):
        q.Unit("bad_angle", 1.0, d.length, metadata={"semantic": "angle"})


def test_canonicalization_preserves_near_unity_scales():
    near_one = q.Unit("near_one", 1.0000000001, d.dimensionless)
    near_erg = q.Unit("near_erg", 1.0000000001, d.energy)

    assert (near_one / q.dimensionless).scale_to_cgs == near_one.scale_to_cgs
    assert (near_erg * q.dimensionless).scale_to_cgs == near_erg.scale_to_cgs


def test_angle_algebra_preserves_tag_only_under_dimensionless_scaling():
    percent = q.Unit("percent", 0.01, d.dimensionless)

    for unit in (q.rad * percent, percent * q.rad, q.rad / percent, q.rad**1):
        assert unit.metadata.get("semantic") == "angle"
    for unit in (q.rad * q.rad, q.rad / q.rad, q.rad**2, q.rad**0):
        assert unit.metadata.get("semantic") is None
    assert (q.rad * q.dimensionless) != q.dimensionless


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_unit_scale_must_be_finite_and_positive(scale):
    with pytest.raises(ValueError, match="finite and positive"):
        q.Unit("invalid", scale, d.length)


def test_documented_wavelength_and_astro_units():
    assert q.nm.scale_to_cgs == pytest.approx(1.0e-7)
    assert q.micron.scale_to_cgs == pytest.approx(1.0e-4)
    assert q.um is q.micron
    assert q.AU.scale_to_cgs == C.AU_CM
    assert q.pc.scale_to_cgs == C.PC_CM
    assert q.Msun.scale_to_cgs == C.MSUN_G
    assert q.Rsun.scale_to_cgs == C.RSUN_CM
    assert q.Lsun.scale_to_cgs == C.LSUN_ERG_S


def test_unit_algebra_combines_dimensions_and_scales():
    velocity = q.km / q.s
    energy = q.g * q.cm**2 / q.s**2
    flux_density = q.erg / q.s / q.cm**2 / q.Hz

    assert velocity.dimensions == d.velocity
    assert velocity.scale_to_cgs == C.KM_CM
    assert energy == q.erg
    assert flux_density.dimensions == d.power / d.length**2 / (d.time**-1)


def test_exact_rational_powers():
    root = (q.cm**2) ** Fraction(1, 2)

    assert root.dimensions == d.length
    assert root.scale_to_cgs == 1.0
    assert str(q.cm ** Fraction(1, 2)) == "cm^(1/2)"


def test_strict_symbols_and_repr_are_stable():
    assert str(q.cm) == "cm"
    assert repr(q.cm) == "Unit('cm')"
    assert str(q.km / q.s) == "km/s"
    assert str(q.g * q.cm**2 / q.s**2) == "erg"


def test_float_powers_are_rejected_until_rationalized():
    with pytest.raises(DimensionError, match="exact rational"):
        q.cm**0.5
