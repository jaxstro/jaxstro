"""Tests for Sonora 2024 raw metadata parsing."""

from __future__ import annotations

import math

import pytest

from jaxstro.atmospheres.sonora import parse_sonora_2024_filename


def test_parse_sonora_filename_preserves_source_gravity_and_derives_cgs_logg():
    metadata = parse_sonora_2024_filename("spectra/t1300g3160f1_m+0.5_co1.0.spec")

    assert metadata.teff == 1300.0
    assert metadata.g_m_s2 == 3160.0
    assert metadata.cloud_label == "f1"
    assert metadata.m_h == 0.5
    assert metadata.c_o == 1.0
    assert math.isclose(metadata.logg, math.log10(3160.0 * 100.0))


def test_parse_sonora_filename_accepts_cloud_free_label():
    metadata = parse_sonora_2024_filename("t900g31nc_m-0.5_co1.0.spec")

    assert metadata.cloud_label == "nc"
    assert metadata.g_m_s2 == 31.0
    assert math.isclose(metadata.logg, math.log10(31.0 * 100.0))


def test_parse_sonora_filename_rejects_non_sonora_name():
    with pytest.raises(ValueError, match="Sonora 2024"):
        parse_sonora_2024_filename("not-a-spectrum.txt")
