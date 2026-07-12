"""AD-vs-FD audits for smooth spectral transformation parameters."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jaxstro.jaxconfig import enable_high_precision

enable_high_precision()

from jaxstro.spectra import (  # noqa: E402
    SpectralAxis,
    SpectralCoordinate,
    SpectralSemantic,
    Spectrum,
    SpectrumProvenance,
    surface_flux_to_luminosity,
    surface_flux_to_observer_flux,
    to_flux_nu,
)
from jaxstro.testing import Case, audit_entry_point  # noqa: E402

PROVENANCE = SpectrumProvenance(
    source_id="synthetic",
    product_id="gradient-audit",
    native_coordinate="wavelength_cm",
    native_density="F_lambda",
    native_unit="erg s^-1 cm^-2 cm^-1",
    canonical_conversion="identity",
    citations=("synthetic:test",),
)
BASE_WAVELENGTH_CM = jnp.array([1.0e-5, 2.0e-5, 4.0e-5])
BASE_FLUX = jnp.array([2.0, 3.0, 5.0])


def _spectrum(wavelength_scale: object = 1.0, flux_scale: object = 1.0) -> Spectrum:
    return Spectrum(
        axis=SpectralAxis.points(
            wavelength_scale * BASE_WAVELENGTH_CM,
            coordinate=SpectralCoordinate.WAVELENGTH,
            unit="cm",
        ),
        values=flux_scale * BASE_FLUX,
        semantic=SpectralSemantic.SURFACE_FLUX_LAMBDA,
        provenance=PROVENANCE,
    )


CASES = (
    Case(
        id="spectra-density-wavelength-scale",
        direction="spectrum->spectrum",
        fn=lambda scale: to_flux_nu(_spectrum(wavelength_scale=scale)).values,
        param="wavelength_scale",
        theta0=1.0,
        reduce=lambda values: jnp.sum(values / 1.0e-19),
        tol=2.0e-5,
        h_rel=1.0e-5,
        allowed_claim="smooth point-density wavelength Jacobian",
        forbidden_claims=("differentiability through sampling-policy changes",),
    ),
    Case(
        id="spectra-density-flux-scale",
        direction="spectrum->spectrum",
        fn=lambda scale: to_flux_nu(_spectrum(flux_scale=scale)).values,
        param="flux_scale",
        theta0=1.0,
        reduce=lambda values: jnp.sum(values / 1.0e-19),
        tol=2.0e-5,
        h_rel=1.0e-5,
        allowed_claim="smooth point-density flux sensitivity",
    ),
    Case(
        id="spectra-luminosity-radius-scale",
        direction="surface-spectrum->luminosity-spectrum",
        fn=lambda scale: (
            surface_flux_to_luminosity(_spectrum(), radius_cm=scale * 7.0e10).values
        ),
        param="radius_scale",
        theta0=1.0,
        reduce=lambda values: jnp.sum(values / 1.0e23),
        tol=2.0e-5,
        h_rel=1.0e-5,
        allowed_claim="smooth spherical-area radius sensitivity",
        forbidden_claims=("nonspherical or beamed emission",),
    ),
    Case(
        id="spectra-observer-distance-scale",
        direction="surface-spectrum->observer-spectrum",
        fn=lambda scale: (
            surface_flux_to_observer_flux(
                _spectrum(),
                radius_cm=7.0e10,
                distance_cm=scale * 3.0e18,
            ).values
        ),
        param="distance_scale",
        theta0=1.0,
        reduce=lambda values: jnp.sum(values / 1.0e-15),
        tol=2.0e-5,
        h_rel=1.0e-5,
        allowed_claim="smooth inverse-square distance sensitivity",
        forbidden_claims=("extinction, redshift, lensing, or instrumental response",),
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_spectral_transform_ad_matches_finite_difference(case: Case) -> None:
    result = audit_entry_point(case)

    assert result.finite
    assert result.status == "clean"
    assert abs(result.ratio - 1.0) < result.tol
    assert result.grad_contract == "smooth_pathwise"
