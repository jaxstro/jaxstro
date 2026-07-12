"""Tests for atmosphere overlap diagnostics."""

from __future__ import annotations

import numpy as np

from jaxstro.atmospheres.overlap import validate_spectrum_overlap
from jaxstro.spectra import (
    SpectralAxis,
    SpectralCoordinate,
    SpectralSemantic,
    Spectrum,
    SpectrumProvenance,
)


def _spectrum(wavelength, values, *, unit="nm", source_id="fixture") -> Spectrum:
    return Spectrum(
        axis=SpectralAxis.points(
            np.asarray(wavelength),
            coordinate=SpectralCoordinate.WAVELENGTH,
            unit=unit,
        ),
        values=np.asarray(values),
        semantic=SpectralSemantic.SURFACE_FLUX_LAMBDA,
        provenance=SpectrumProvenance(
            source_id=source_id,
            product_id=source_id,
            native_coordinate=f"wavelength_{unit}",
            native_density="F_lambda",
            native_unit=f"fixture per {unit}",
            canonical_conversion="fixture identity",
            citations=("fixture:overlap",),
        ),
    )


def test_overlap_validation_reports_domain_and_normalized_difference():
    left = _spectrum([1000.0, 1500.0, 2000.0], [1.0, 2.0, 3.0], source_id="left")
    right = _spectrum(
        [10000.0, 15000.0, 20000.0],
        [2.0, 4.0, 8.0],
        unit="angstrom",
        source_id="right",
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
    left = _spectrum([1.0, 2.0], [1.0, 2.0], source_id="left")
    right = _spectrum([3.0, 4.0], [1.0, 2.0], source_id="right")

    diagnostic = validate_spectrum_overlap("left", left, "right", right)

    assert diagnostic.passed is False
    assert diagnostic.n_overlap_left == 0
    assert diagnostic.max_abs_normalized_difference is None
