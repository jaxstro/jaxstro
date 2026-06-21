"""Tests for TLUSTY raw metadata and numeric parsing."""

from __future__ import annotations

import pytest

from jaxstro.atmospheres.tlusty import (
    parse_tlusty_float,
    parse_tlusty_flux_filename,
)


def test_parse_tlusty_filename_extracts_axes_and_cn_flag():
    metadata = parse_tlusty_flux_filename("BC15000g175v10CN.flux.gz")

    assert metadata.prefix == "BC"
    assert metadata.teff == 15000.0
    assert metadata.logg == 1.75
    assert metadata.vturb_km_s == 10.0
    assert metadata.cn_altered is True


def test_parse_tlusty_filename_rejects_non_flux_name():
    with pytest.raises(ValueError, match="TLUSTY flux"):
        parse_tlusty_flux_filename("BC15000g175v10.11.gz")


def test_parse_tlusty_float_accepts_fortran_and_bare_exponents():
    assert parse_tlusty_float("6.67943694D+16") == 6.67943694e16
    assert parse_tlusty_float("1.4363-100") == 1.4363e-100
