"""Downstream packages can consume spectra without atmosphere internals."""

from __future__ import annotations

import jax.numpy as jnp

from jaxstro.atmospheres import AtmosphereParams, AtmosphereQuery
from jaxstro.spectra import (
    SpectralAxis,
    SpectralCoordinate,
    SpectralPlan,
    SpectralSemantic,
    Spectrum,
    SpectrumProvenance,
)


def test_generic_consumer_contract_uses_jaxstro_spectra_public_types() -> None:
    axis = SpectralAxis.points(
        jnp.array([500.0, 600.0]),
        coordinate=SpectralCoordinate.WAVELENGTH,
        unit="nm",
    )
    spectrum = Spectrum(
        axis=axis,
        values=jnp.array([1.0, 2.0]),
        semantic=SpectralSemantic.SURFACE_FLUX_LAMBDA,
        provenance=SpectrumProvenance(
            source_id="consumer-fixture",
            product_id="consumer-product",
            native_coordinate="wavelength_nm",
            native_density="F_lambda",
            native_unit="erg s^-1 cm^-2 nm^-1",
            canonical_conversion="identity",
            citations=("fixture:consumer-contract",),
        ),
    )
    query = AtmosphereQuery(
        params=AtmosphereParams(teff=5000.0, logg=4.0),
        product_id="consumer-product",
        spectral_plan=SpectralPlan(axis),
    )

    assert spectrum.values.shape == query.spectral_plan.target_axis.values.shape
    assert spectrum.value_unit == "erg s^-1 cm^-2 nm^-1"
