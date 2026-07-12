"""Conservation and differentiation checks for spectral remapping."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxstro.spectra import (
    CoveragePolicy,
    SpectralAxis,
    SpectralCoordinate,
    SpectralPlan,
    SpectralSampling,
    SpectralSemantic,
    Spectrum,
    SpectrumProvenance,
    SpectrumStatusCode,
    resample_spectrum,
)

PROVENANCE = SpectrumProvenance(
    source_id="synthetic",
    product_id="conservation",
    native_coordinate="wavelength_nm",
    native_density="F_lambda",
    native_unit="erg s^-1 cm^-2 nm^-1",
    canonical_conversion="identity",
    citations=("synthetic:test",),
)


def test_bin_average_remap_preserves_integrated_spectral_density() -> None:
    source_axis = SpectralAxis.bins(
        jnp.array([100.0, 200.0, 400.0]),
        coordinate=SpectralCoordinate.WAVELENGTH,
        unit="nm",
        sampling=SpectralSampling.BIN_AVERAGES,
    )
    target_axis = SpectralAxis.bins(
        jnp.array([100.0, 150.0, 300.0, 400.0]),
        coordinate=SpectralCoordinate.WAVELENGTH,
        unit="nm",
        sampling=SpectralSampling.BIN_AVERAGES,
    )
    source = Spectrum(
        axis=source_axis,
        values=jnp.array([2.0, 5.0]),
        semantic=SpectralSemantic.SURFACE_FLUX_LAMBDA,
        provenance=PROVENANCE,
    )

    result = resample_spectrum(
        source,
        SpectralPlan(target_axis, CoveragePolicy.INTERSECTION),
    )

    source_integral = jnp.sum(source.values * jnp.diff(source_axis.edges))
    target_integral = jnp.sum(result.spectrum.values * jnp.diff(target_axis.edges))
    assert int(result.status.code) == SpectrumStatusCode.OK
    np.testing.assert_allclose(target_integral, source_integral, rtol=1.0e-14)
    assert result.spectrum.provenance.operations[-1] == (
        "resample:conservative-bin-average"
    )


def test_linear_point_resampling_has_exact_value_gradients() -> None:
    source_axis = SpectralAxis.points(
        jnp.array([100.0, 200.0, 400.0]),
        coordinate=SpectralCoordinate.WAVELENGTH,
        unit="nm",
    )
    target_axis = SpectralAxis.points(
        jnp.array([150.0, 300.0]),
        coordinate=SpectralCoordinate.WAVELENGTH,
        unit="nm",
    )
    plan = SpectralPlan(target_axis, CoveragePolicy.INTERSECTION)

    def total(values: object) -> object:
        source = Spectrum(
            axis=source_axis,
            values=values,
            semantic=SpectralSemantic.SURFACE_FLUX_LAMBDA,
            provenance=PROVENANCE,
        )
        return jnp.sum(resample_spectrum(source, plan).spectrum.values)

    gradient = jax.grad(total)(jnp.array([2.0, 4.0, 8.0]))

    np.testing.assert_allclose(gradient, [0.5, 1.0, 0.5], rtol=1.0e-14)
