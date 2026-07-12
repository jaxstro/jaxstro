"""Analytic tests for exact spectral and geometric transformations."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstro.constants import C_CGS
from jaxstro.jaxconfig import enable_high_precision

enable_high_precision()

from jaxstro.spectra import (  # noqa: E402
    SpectralAxis,
    SpectralCoordinate,
    SpectralSampling,
    SpectralSemantic,
    Spectrum,
    SpectrumProvenance,
    surface_flux_to_luminosity,
    surface_flux_to_observer_flux,
    to_flux_lambda,
    to_flux_nu,
    to_frequency,
    to_wavelength,
)


def _provenance() -> SpectrumProvenance:
    return SpectrumProvenance(
        source_id="synthetic",
        product_id="synthetic-grid",
        native_coordinate="wavelength_cm",
        native_density="F_lambda",
        native_unit="erg s^-1 cm^-2 cm^-1",
        canonical_conversion="identity",
        citations=("synthetic:test",),
    )


def _surface_flux_lambda() -> Spectrum:
    return Spectrum(
        axis=SpectralAxis.points(
            jnp.array([1.0e-5, 2.0e-5, 4.0e-5]),
            coordinate=SpectralCoordinate.WAVELENGTH,
            unit="cm",
        ),
        values=jnp.array([2.0, 3.0, 5.0]),
        semantic=SpectralSemantic.SURFACE_FLUX_LAMBDA,
        provenance=_provenance(),
    )


def test_coordinate_transforms_reverse_samples_to_remain_increasing() -> None:
    wavelength = _surface_flux_lambda().axis

    frequency = to_frequency(wavelength)
    rebuilt = to_wavelength(frequency)

    np.testing.assert_allclose(
        frequency.values,
        C_CGS / np.array([4.0e-5, 2.0e-5, 1.0e-5]),
        rtol=1.0e-15,
    )
    assert frequency.coordinate is SpectralCoordinate.FREQUENCY
    assert frequency.unit == "Hz"
    assert bool(jnp.all(jnp.diff(frequency.values) > 0.0))
    np.testing.assert_allclose(rebuilt.values, wavelength.values, rtol=1.0e-15)
    assert rebuilt.coordinate is SpectralCoordinate.WAVELENGTH
    assert rebuilt.unit == "cm"


def test_binned_coordinate_transform_rebuilds_ordered_edges_and_centers() -> None:
    wavelength = SpectralAxis.bins(
        jnp.array([1.0e-5, 2.0e-5, 4.0e-5]),
        coordinate=SpectralCoordinate.WAVELENGTH,
        unit="cm",
        sampling=SpectralSampling.BIN_AVERAGES,
    )

    frequency = to_frequency(wavelength)

    expected_edges = C_CGS / np.array([4.0e-5, 2.0e-5, 1.0e-5])
    np.testing.assert_allclose(frequency.edges, expected_edges, rtol=1.0e-15)
    np.testing.assert_allclose(
        frequency.values,
        0.5 * (expected_edges[:-1] + expected_edges[1:]),
        rtol=1.0e-15,
    )


def test_flux_density_conversion_uses_jacobian_and_roundtrips() -> None:
    flux_lambda = _surface_flux_lambda()

    flux_nu = to_flux_nu(flux_lambda)
    rebuilt = to_flux_lambda(flux_nu)

    expected = (flux_lambda.values * flux_lambda.axis.values**2 / C_CGS)[::-1]
    np.testing.assert_allclose(flux_nu.values, expected, rtol=1.0e-15)
    assert flux_nu.semantic is SpectralSemantic.SURFACE_FLUX_NU
    assert flux_nu.value_unit == "erg s^-1 cm^-2 Hz^-1"
    np.testing.assert_allclose(rebuilt.axis.values, flux_lambda.axis.values)
    np.testing.assert_allclose(rebuilt.values, flux_lambda.values, rtol=1.0e-15)
    assert rebuilt.semantic is SpectralSemantic.SURFACE_FLUX_LAMBDA
    assert rebuilt.value_unit == "erg s^-1 cm^-2 cm^-1"
    assert flux_nu.provenance.operations[-1] == "density:F_lambda->F_nu"
    assert rebuilt.provenance.operations[-1] == "density:F_nu->F_lambda"


def test_transform_functions_are_idempotent_in_target_representation() -> None:
    flux_lambda = _surface_flux_lambda()
    flux_nu = to_flux_nu(flux_lambda)

    assert to_wavelength(flux_lambda.axis) is flux_lambda.axis
    assert to_frequency(flux_nu.axis) is flux_nu.axis
    assert to_flux_lambda(flux_lambda) is flux_lambda
    assert to_flux_nu(flux_nu) is flux_nu


def test_density_conversion_rejects_bin_center_jacobian_approximation() -> None:
    axis = SpectralAxis.bins(
        jnp.array([1.0e-5, 2.0e-5, 4.0e-5]),
        coordinate=SpectralCoordinate.WAVELENGTH,
        unit="cm",
    )
    spectrum = Spectrum(
        axis=axis,
        values=jnp.array([2.0, 3.0]),
        semantic=SpectralSemantic.SURFACE_FLUX_LAMBDA,
        provenance=_provenance(),
    )

    with pytest.raises(ValueError, match="point-sampled"):
        to_flux_nu(spectrum)


def test_surface_flux_to_luminosity_uses_spherical_surface_area() -> None:
    surface = _surface_flux_lambda()
    radius_cm = 7.0e10

    luminosity = surface_flux_to_luminosity(surface, radius_cm=radius_cm)

    np.testing.assert_allclose(
        luminosity.values,
        4.0 * np.pi * radius_cm**2 * surface.values,
        rtol=1.0e-15,
    )
    assert luminosity.semantic is SpectralSemantic.LUMINOSITY_LAMBDA
    assert luminosity.value_unit == "erg s^-1 cm^-1"
    assert luminosity.provenance.operations[-1] == (
        "geometry:surface_flux_to_luminosity:spherical_isotropic"
    )


def test_surface_flux_to_observer_flux_uses_inverse_square_dilution() -> None:
    surface = _surface_flux_lambda()
    radius_cm = 7.0e10
    distance_cm = 3.0e18

    observed = surface_flux_to_observer_flux(
        surface,
        radius_cm=radius_cm,
        distance_cm=distance_cm,
    )

    np.testing.assert_allclose(
        observed.values,
        (radius_cm / distance_cm) ** 2 * surface.values,
        rtol=1.0e-15,
    )
    assert observed.semantic is SpectralSemantic.OBSERVER_FLUX_LAMBDA
    assert observed.value_unit == "erg s^-1 cm^-2 cm^-1"
    assert observed.provenance.operations[-1] == (
        "geometry:surface_flux_to_observer_flux:spherical_isotropic"
    )


def test_geometric_transforms_reject_wrong_semantic_and_nonpositive_scales() -> None:
    surface = _surface_flux_lambda()
    luminosity = surface_flux_to_luminosity(surface, radius_cm=7.0e10)

    with pytest.raises(ValueError, match="surface-flux semantic"):
        surface_flux_to_luminosity(luminosity, radius_cm=7.0e10)
    with pytest.raises(ValueError, match="radius_cm must be positive"):
        surface_flux_to_luminosity(surface, radius_cm=0.0)
    with pytest.raises(ValueError, match="distance_cm must be positive"):
        surface_flux_to_observer_flux(
            surface,
            radius_cm=7.0e10,
            distance_cm=-1.0,
        )


def test_transform_stack_is_jittable() -> None:
    surface = _surface_flux_lambda()

    @jax.jit
    def transform(values: object) -> object:
        spectrum = Spectrum(
            axis=surface.axis,
            values=values,
            semantic=surface.semantic,
            provenance=surface.provenance,
        )
        return surface_flux_to_observer_flux(
            to_flux_nu(spectrum),
            radius_cm=7.0e10,
            distance_cm=3.0e18,
        ).values

    result = transform(surface.values)

    assert result.shape == surface.values.shape
    assert bool(jnp.all(jnp.isfinite(result)))
