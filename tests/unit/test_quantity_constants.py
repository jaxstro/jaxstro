"""Tests for versioned quantity constants."""

import pytest

from jaxstro import constants as C
from jaxstro import quantity as q


def test_constants_are_quantities():
    assert isinstance(q.constants.G, q.Quantity)
    assert isinstance(q.constants.c, q.Quantity)
    assert isinstance(q.constants.k_B, q.Quantity)
    assert isinstance(q.constants.Msun, q.Quantity)


def test_source_metadata_is_inspectable():
    meta = q.constants.metadata("G")

    assert meta.name == "Newtonian constant of gravitation"
    assert meta.source == "CODATA"
    assert meta.version == "2018"
    assert meta.checked_against == "CODATA 2022"
    assert meta.accessed == "2026-06-23"


def test_raw_value_helpers_return_cgs_values():
    assert q.constants.raw_value_cgs("G") == C.G_CGS
    assert q.constants.raw_value_cgs("c") == C.C_CGS
    assert q.constants.raw_value_cgs("k_B") == C.K_B
    assert q.constants.raw_value_cgs("Msun") == C.MSUN_G


def test_quantity_constants_match_legacy_constants():
    assert q.constants.G.to_value(q.cm**3 / q.g / q.s**2) == C.G_CGS
    assert q.constants.c.to_value(q.cm / q.s) == C.C_CGS
    assert q.constants.h.to_value(q.erg * q.s) == C.H_CGS
    assert q.constants.sigma_sb.to_value(
        q.erg / q.s / q.cm**2 / q.K**4
    ) == pytest.approx(C.SIGMA_SB)
    assert q.constants.AU.to_value(q.cm) == C.AU_CM
    assert q.constants.Rsun.to_value(q.cm) == C.RSUN_CM
    assert q.constants.Lsun.to_value(q.erg / q.s) == C.LSUN_ERG_S


def test_default_constant_set_documents_compatibility_posture():
    assert q.constants.default_set.name == "jaxstro-legacy-compatible"
    assert "jaxstro.constants" in q.constants.default_set.description
