"""Exact coordinate, density, and geometric spectral transformations."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax.numpy as jnp

from jaxstro.constants import C_CGS
from jaxstro.numerics.checks import try_concrete_bool

from .types import (
    SpectralAxis,
    SpectralCoordinate,
    SpectralSampling,
    SpectralSemantic,
    Spectrum,
    SpectrumProvenance,
)

_LAMBDA_TO_NU = {
    SpectralSemantic.SURFACE_FLUX_LAMBDA: SpectralSemantic.SURFACE_FLUX_NU,
    SpectralSemantic.LUMINOSITY_LAMBDA: SpectralSemantic.LUMINOSITY_NU,
    SpectralSemantic.OBSERVER_FLUX_LAMBDA: SpectralSemantic.OBSERVER_FLUX_NU,
}
_NU_TO_LAMBDA = {target: source for source, target in _LAMBDA_TO_NU.items()}
_SURFACE_TO_LUMINOSITY = {
    SpectralSemantic.SURFACE_FLUX_LAMBDA: SpectralSemantic.LUMINOSITY_LAMBDA,
    SpectralSemantic.SURFACE_FLUX_NU: SpectralSemantic.LUMINOSITY_NU,
}
_SURFACE_TO_OBSERVER = {
    SpectralSemantic.SURFACE_FLUX_LAMBDA: SpectralSemantic.OBSERVER_FLUX_LAMBDA,
    SpectralSemantic.SURFACE_FLUX_NU: SpectralSemantic.OBSERVER_FLUX_NU,
}


def _with_operation(
    provenance: SpectrumProvenance, operation: str
) -> SpectrumProvenance:
    return replace(provenance, operations=(*provenance.operations, operation))


def _require_scalar_positive(value: Any, name: str):
    array = jnp.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar")
    positive = try_concrete_bool(array > 0.0)
    if positive is False:
        raise ValueError(f"{name} must be positive")
    return array


def to_frequency(axis: SpectralAxis) -> SpectralAxis:
    """Convert a canonical wavelength axis in cm to increasing frequency in Hz."""
    if axis.coordinate is SpectralCoordinate.FREQUENCY:
        if axis.unit != "Hz":
            raise ValueError("frequency axes must use Hz for canonical transforms")
        return axis
    if axis.unit != "cm":
        raise ValueError("wavelength axes must use cm for canonical transforms")
    if axis.sampling is SpectralSampling.POINTS:
        return SpectralAxis.points(
            C_CGS / axis.values[::-1],
            coordinate=SpectralCoordinate.FREQUENCY,
            unit="Hz",
            resolving_power=axis.resolving_power,
        )
    if axis.edges is None:  # protected by SpectralAxis invariants
        raise ValueError("binned wavelength axes require edges")
    return SpectralAxis.bins(
        C_CGS / axis.edges[::-1],
        coordinate=SpectralCoordinate.FREQUENCY,
        unit="Hz",
        sampling=axis.sampling,
        resolving_power=axis.resolving_power,
    )


def to_wavelength(axis: SpectralAxis) -> SpectralAxis:
    """Convert a canonical frequency axis in Hz to increasing wavelength in cm."""
    if axis.coordinate is SpectralCoordinate.WAVELENGTH:
        if axis.unit != "cm":
            raise ValueError("wavelength axes must use cm for canonical transforms")
        return axis
    if axis.unit != "Hz":
        raise ValueError("frequency axes must use Hz for canonical transforms")
    if axis.sampling is SpectralSampling.POINTS:
        return SpectralAxis.points(
            C_CGS / axis.values[::-1],
            coordinate=SpectralCoordinate.WAVELENGTH,
            unit="cm",
            resolving_power=axis.resolving_power,
        )
    if axis.edges is None:  # protected by SpectralAxis invariants
        raise ValueError("binned frequency axes require edges")
    return SpectralAxis.bins(
        C_CGS / axis.edges[::-1],
        coordinate=SpectralCoordinate.WAVELENGTH,
        unit="cm",
        sampling=axis.sampling,
        resolving_power=axis.resolving_power,
    )


def to_flux_nu(spectrum: Spectrum) -> Spectrum:
    """Convert a point-sampled wavelength density to the equivalent F_nu."""
    if spectrum.semantic in _NU_TO_LAMBDA:
        return spectrum
    target_semantic = _LAMBDA_TO_NU.get(spectrum.semantic)
    if target_semantic is None:
        raise ValueError("spectrum does not carry a wavelength-density semantic")
    if spectrum.axis.sampling is not SpectralSampling.POINTS:
        raise ValueError("F_lambda/F_nu conversion requires point-sampled spectra")
    wavelength = spectrum.axis.values
    values = (spectrum.values * wavelength**2 / C_CGS)[::-1]
    return Spectrum(
        axis=to_frequency(spectrum.axis),
        values=values,
        semantic=target_semantic,
        provenance=_with_operation(spectrum.provenance, "density:F_lambda->F_nu"),
    )


def to_flux_lambda(spectrum: Spectrum) -> Spectrum:
    """Convert a point-sampled frequency density to the equivalent F_lambda."""
    if spectrum.semantic in _LAMBDA_TO_NU:
        return spectrum
    target_semantic = _NU_TO_LAMBDA.get(spectrum.semantic)
    if target_semantic is None:
        raise ValueError("spectrum does not carry a frequency-density semantic")
    if spectrum.axis.sampling is not SpectralSampling.POINTS:
        raise ValueError("F_nu/F_lambda conversion requires point-sampled spectra")
    wavelength_axis = to_wavelength(spectrum.axis)
    values = spectrum.values[::-1] * C_CGS / wavelength_axis.values**2
    return Spectrum(
        axis=wavelength_axis,
        values=values,
        semantic=target_semantic,
        provenance=_with_operation(spectrum.provenance, "density:F_nu->F_lambda"),
    )


def surface_flux_to_luminosity(
    spectrum: Spectrum,
    *,
    radius_cm: Any,
) -> Spectrum:
    """Convert surface flux density to luminosity density for a spherical source."""
    target_semantic = _SURFACE_TO_LUMINOSITY.get(spectrum.semantic)
    if target_semantic is None:
        raise ValueError("luminosity conversion requires a surface-flux semantic")
    radius = _require_scalar_positive(radius_cm, "radius_cm")
    return Spectrum(
        axis=spectrum.axis,
        values=4.0 * jnp.pi * radius**2 * spectrum.values,
        semantic=target_semantic,
        provenance=_with_operation(
            spectrum.provenance,
            "geometry:surface_flux_to_luminosity:spherical_isotropic",
        ),
    )


def surface_flux_to_observer_flux(
    spectrum: Spectrum,
    *,
    radius_cm: Any,
    distance_cm: Any,
) -> Spectrum:
    """Dilute surface flux density to an observer for a spherical isotropic source."""
    target_semantic = _SURFACE_TO_OBSERVER.get(spectrum.semantic)
    if target_semantic is None:
        raise ValueError("observer conversion requires a surface-flux semantic")
    radius = _require_scalar_positive(radius_cm, "radius_cm")
    distance = _require_scalar_positive(distance_cm, "distance_cm")
    return Spectrum(
        axis=spectrum.axis,
        values=(radius / distance) ** 2 * spectrum.values,
        semantic=target_semantic,
        provenance=_with_operation(
            spectrum.provenance,
            "geometry:surface_flux_to_observer_flux:spherical_isotropic",
        ),
    )


__all__ = [
    "surface_flux_to_luminosity",
    "surface_flux_to_observer_flux",
    "to_flux_lambda",
    "to_flux_nu",
    "to_frequency",
    "to_wavelength",
]
