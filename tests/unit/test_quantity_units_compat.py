"""Compatibility tests between legacy UnitSystem and quantity units."""

import warnings

from jaxstro import quantity as q
from jaxstro import units as U


def test_existing_jaxstro_units_imports_still_work():
    assert U.CGS.mass_unit == "g"
    assert U.STELLAR is U.ASTRO_DYNAMICAL
    assert U.STAR is U.ASTRO_STELLAR
    assert U.BINARY is U.ASTRO_PLANETARY
    assert U.DEFAULT is U.CGS


def test_unit_system_quantity_unit_bridge():
    assert U.CGS.quantity_units == (q.g, q.cm, q.s)
    assert U.STELLAR.quantity_units == (q.Msun, q.pc, q.Myr)
    assert U.STAR.quantity_units == (q.Msun, q.Rsun, q.Myr)
    assert U.BINARY.quantity_units == (q.Msun, q.AU, q.yr)


def test_unit_system_quantity_scale_bridge():
    scales = U.STELLAR.quantity_scales

    assert scales["mass"] == q.Msun.scale_to_cgs
    assert scales["length"] == q.pc.scale_to_cgs
    assert scales["time"] == q.Myr.scale_to_cgs


def test_no_deprecation_warnings_for_legacy_units():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = U.STELLAR.G
        _ = U.STELLAR.convert_length(1.0, to=U.CGS)
        _ = U.STELLAR.quantity_units

    assert not caught
