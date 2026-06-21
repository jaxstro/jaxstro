"""Diagnostic overlap checks for atmosphere spectra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .spectra import Spectrum


@dataclass(frozen=True)
class OverlapDiagnostic:
    """Diagnostic result for one pair of spectra."""

    left_dataset: str
    right_dataset: str
    overlap_min_nm: float | None
    overlap_max_nm: float | None
    n_overlap_left: int
    n_overlap_right: int
    finite_left: bool
    finite_right: bool
    max_abs_normalized_difference: float | None
    passed: bool
    message: str


def validate_spectrum_overlap(
    left_dataset: str,
    left: Spectrum,
    right_dataset: str,
    right: Spectrum,
) -> OverlapDiagnostic:
    """Compare wavelength overlap and normalized SED shape diagnostics.

    This is intentionally not a strict physical-equality test across atmosphere
    model families.
    """
    left_wave_nm = _wavelength_to_nm(left.wavelength, left.wavelength_unit)
    right_wave_nm = _wavelength_to_nm(right.wavelength, right.wavelength_unit)
    left_flux = np.asarray(left.flux_lambda, dtype=np.float64)
    right_flux = np.asarray(right.flux_lambda, dtype=np.float64)

    finite_left = bool(
        np.all(np.isfinite(left_wave_nm)) and np.all(np.isfinite(left_flux))
    )
    finite_right = bool(
        np.all(np.isfinite(right_wave_nm)) and np.all(np.isfinite(right_flux))
    )
    overlap_min = max(float(np.min(left_wave_nm)), float(np.min(right_wave_nm)))
    overlap_max = min(float(np.max(left_wave_nm)), float(np.max(right_wave_nm)))
    if overlap_min > overlap_max:
        return OverlapDiagnostic(
            left_dataset=left_dataset,
            right_dataset=right_dataset,
            overlap_min_nm=None,
            overlap_max_nm=None,
            n_overlap_left=0,
            n_overlap_right=0,
            finite_left=finite_left,
            finite_right=finite_right,
            max_abs_normalized_difference=None,
            passed=False,
            message="no wavelength-domain overlap",
        )

    left_mask = (left_wave_nm >= overlap_min) & (left_wave_nm <= overlap_max)
    right_mask = (right_wave_nm >= overlap_min) & (right_wave_nm <= overlap_max)
    n_left = int(np.count_nonzero(left_mask))
    n_right = int(np.count_nonzero(right_mask))
    if n_left == 0 or n_right == 0:
        return OverlapDiagnostic(
            left_dataset=left_dataset,
            right_dataset=right_dataset,
            overlap_min_nm=overlap_min,
            overlap_max_nm=overlap_max,
            n_overlap_left=n_left,
            n_overlap_right=n_right,
            finite_left=finite_left,
            finite_right=finite_right,
            max_abs_normalized_difference=None,
            passed=False,
            message="overlap interval contains no sampled points in one spectrum",
        )

    left_x, left_y = _sorted_xy(left_wave_nm[left_mask], left_flux[left_mask])
    right_x, right_y = _sorted_xy(right_wave_nm[right_mask], right_flux[right_mask])
    right_on_left = np.interp(left_x, right_x, right_y)
    diff = _normalize(left_y) - _normalize(right_on_left)
    max_diff = float(np.max(np.abs(diff)))
    passed = bool(finite_left and finite_right and np.isfinite(max_diff))
    return OverlapDiagnostic(
        left_dataset=left_dataset,
        right_dataset=right_dataset,
        overlap_min_nm=overlap_min,
        overlap_max_nm=overlap_max,
        n_overlap_left=n_left,
        n_overlap_right=n_right,
        finite_left=finite_left,
        finite_right=finite_right,
        max_abs_normalized_difference=max_diff,
        passed=passed,
        message=(
            "diagnostic overlap computed; normalized SED difference is not a "
            "model-family equality criterion"
        ),
    )


def _wavelength_to_nm(wavelength: Any, unit: str) -> np.ndarray:
    values = np.asarray(wavelength, dtype=np.float64)
    normalized = unit.lower()
    if normalized in {"nm", "nanometer", "nanometers"}:
        return values
    if normalized in {"angstrom", "angstroms", "a"}:
        return values * 0.1
    if normalized in {"micron", "microns", "um"}:
        return values * 1000.0
    raise ValueError(f"Unsupported wavelength unit for overlap diagnostic: {unit}")


def _sorted_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    return x[order], y[order]


def _normalize(values: np.ndarray) -> np.ndarray:
    scale = np.nanmax(np.abs(values))
    if not np.isfinite(scale) or scale == 0.0:
        return values
    return values / scale


__all__ = [
    "OverlapDiagnostic",
    "validate_spectrum_overlap",
]
