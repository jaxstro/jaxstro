"""Tests for atmosphere overlap diagnostics."""

from __future__ import annotations

import numpy as np

from jaxstro.atmospheres import Spectrum
from jaxstro.atmospheres.overlap import validate_spectrum_overlap


def test_overlap_validation_reports_domain_and_normalized_difference():
    left = Spectrum(
        wavelength=np.array([1000.0, 1500.0, 2000.0]),
        flux_lambda=np.array([1.0, 2.0, 3.0]),
        wavelength_unit="nm",
        flux_unit="left",
    )
    right = Spectrum(
        wavelength=np.array([10000.0, 15000.0, 20000.0]),
        flux_lambda=np.array([2.0, 4.0, 8.0]),
        wavelength_unit="angstrom",
        flux_unit="right",
    )

    diagnostic = validate_spectrum_overlap("left", left, "right", right)

    assert diagnostic.left_dataset == "left"
    assert diagnostic.right_dataset == "right"
    assert diagnostic.overlap_min_nm == 1000.0
    assert diagnostic.overlap_max_nm == 2000.0
    assert diagnostic.n_overlap_left == 3
    assert diagnostic.n_overlap_right == 3
    assert diagnostic.finite_left is True
    assert diagnostic.finite_right is True
    assert diagnostic.max_abs_normalized_difference > 0.0
    assert diagnostic.passed is True


def test_overlap_validation_fails_closed_without_wavelength_intersection():
    left = Spectrum(
        wavelength=np.array([1.0, 2.0]),
        flux_lambda=np.array([1.0, 2.0]),
        wavelength_unit="nm",
    )
    right = Spectrum(
        wavelength=np.array([3.0, 4.0]),
        flux_lambda=np.array([1.0, 2.0]),
        wavelength_unit="nm",
    )

    diagnostic = validate_spectrum_overlap("left", left, "right", right)

    assert diagnostic.passed is False
    assert diagnostic.n_overlap_left == 0
    assert diagnostic.max_abs_normalized_difference is None
