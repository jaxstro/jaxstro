"""Unit tests for fail-closed spectral resampling."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstro.numerics import interpolation
from jaxstro.spectra import (
    CoveragePolicy,
    PointResamplingMethod,
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
    product_id="resampling",
    native_coordinate="wavelength_nm",
    native_density="F_lambda",
    native_unit="erg s^-1 cm^-2 nm^-1",
    canonical_conversion="identity",
    citations=("synthetic:test",),
)


def _points() -> Spectrum:
    return Spectrum(
        axis=SpectralAxis.points(
            jnp.array([100.0, 200.0, 400.0]),
            coordinate=SpectralCoordinate.WAVELENGTH,
            unit="nm",
        ),
        values=jnp.array([2.0, 4.0, 8.0]),
        semantic=SpectralSemantic.SURFACE_FLUX_LAMBDA,
        provenance=PROVENANCE,
    )


def _curved_points() -> Spectrum:
    return Spectrum(
        axis=SpectralAxis.points(
            jnp.array([100.0, 180.0, 280.0, 400.0]),
            coordinate=SpectralCoordinate.WAVELENGTH,
            unit="nm",
        ),
        values=jnp.array([1.0, 2.0, 1.5, 3.0]),
        semantic=SpectralSemantic.SURFACE_FLUX_LAMBDA,
        provenance=PROVENANCE,
    )


def _plan(axis: SpectralAxis) -> SpectralPlan:
    return SpectralPlan(axis, CoveragePolicy.INTERSECTION)


def test_identical_axis_returns_bit_identical_values() -> None:
    source = _points()

    result = resample_spectrum(source, _plan(source.axis))

    assert int(result.status.code) == SpectrumStatusCode.OK
    np.testing.assert_array_equal(result.spectrum.values, source.values)
    assert result.spectrum.values.dtype == source.values.dtype
    assert result.spectrum.provenance.operations[-1] == "resample:identity"


def test_point_samples_use_linear_interpolation_inside_coverage() -> None:
    target = SpectralAxis.points(
        jnp.array([150.0, 300.0]),
        coordinate=SpectralCoordinate.WAVELENGTH,
        unit="nm",
    )

    result = resample_spectrum(_points(), _plan(target))

    assert int(result.status.code) == SpectrumStatusCode.OK
    np.testing.assert_allclose(result.spectrum.values, [3.0, 6.0])
    assert result.spectrum.axis is target
    assert result.spectrum.provenance.operations[-1] == "resample:linear-points"


def test_linear_points_delegate_to_jaxstro_interp1d() -> None:
    source = _points()
    target = SpectralAxis.points(
        jnp.array([150.0, 300.0]),
        coordinate=SpectralCoordinate.WAVELENGTH,
        unit="nm",
    )

    result = resample_spectrum(source, SpectralPlan(target))
    expected = interpolation.interp1d(
        source.axis.values,
        source.values,
        target.values,
    )

    np.testing.assert_array_equal(result.spectrum.values, expected)
    assert result.spectrum.provenance.operations[-1] == "resample:linear-points"


def test_monotone_cubic_points_delegate_to_jaxstro_pchip() -> None:
    source = _curved_points()
    target = SpectralAxis.points(
        jnp.linspace(100.0, 400.0, 31),
        coordinate=SpectralCoordinate.WAVELENGTH,
        unit="nm",
    )
    plan = SpectralPlan(
        target,
        point_method=PointResamplingMethod.MONOTONE_CUBIC,
    )

    result = resample_spectrum(source, plan)
    expected = interpolation.monotone_cubic_interp(
        source.axis.values,
        source.values,
        target.values,
    )

    np.testing.assert_array_equal(result.spectrum.values, expected)
    assert bool(jnp.all(result.spectrum.values >= jnp.min(source.values)))
    assert bool(jnp.all(result.spectrum.values <= jnp.max(source.values)))
    assert result.spectrum.provenance.operations[-1] == (
        "resample:monotone-cubic-points"
    )


def test_identical_axis_ignores_selected_point_method() -> None:
    source = _curved_points()
    plan = SpectralPlan(
        source.axis,
        point_method=PointResamplingMethod.MONOTONE_CUBIC,
    )

    result = resample_spectrum(source, plan)

    np.testing.assert_array_equal(result.spectrum.values, source.values)
    assert result.spectrum.provenance.operations[-1] == "resample:identity"


def test_target_outside_source_coverage_fails_without_extrapolation() -> None:
    target = SpectralAxis.points(
        jnp.array([90.0, 150.0]),
        coordinate=SpectralCoordinate.WAVELENGTH,
        unit="nm",
    )

    result = resample_spectrum(_points(), _plan(target))

    assert int(result.status.code) == SpectrumStatusCode.UNSUPPORTED_SPECTRAL_WINDOW
    assert bool(jnp.all(jnp.isnan(result.spectrum.values)))


def test_bin_target_outside_source_coverage_fails_without_zero_fill() -> None:
    source_axis = SpectralAxis.bins(
        jnp.array([100.0, 200.0, 400.0]),
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
    target = SpectralAxis.bins(
        jnp.array([90.0, 150.0, 300.0]),
        coordinate=SpectralCoordinate.WAVELENGTH,
        unit="nm",
        sampling=SpectralSampling.BIN_AVERAGES,
    )

    result = resample_spectrum(source, _plan(target))

    assert int(result.status.code) == SpectrumStatusCode.UNSUPPORTED_SPECTRAL_WINDOW
    assert bool(jnp.all(jnp.isnan(result.spectrum.values)))


@pytest.mark.parametrize(
    "target",
    [
        SpectralAxis.points(
            jnp.array([1.0e14, 2.0e14]),
            coordinate=SpectralCoordinate.FREQUENCY,
            unit="Hz",
        ),
        SpectralAxis.points(
            jnp.array([1000.0, 2000.0]),
            coordinate=SpectralCoordinate.WAVELENGTH,
            unit="angstrom",
        ),
    ],
)
def test_resampling_rejects_coordinate_or_unit_changes(target: SpectralAxis) -> None:
    with pytest.raises(ValueError, match="same coordinate and unit"):
        resample_spectrum(_points(), _plan(target))


def test_resampling_rejects_point_to_bin_conversion() -> None:
    target = SpectralAxis.bins(
        jnp.array([100.0, 200.0, 400.0]),
        coordinate=SpectralCoordinate.WAVELENGTH,
        unit="nm",
        sampling=SpectralSampling.BIN_AVERAGES,
    )

    with pytest.raises(ValueError, match="matching sampling semantics"):
        resample_spectrum(_points(), _plan(target))


def test_resampling_rejects_bin_to_point_conversion() -> None:
    source_axis = SpectralAxis.bins(
        jnp.array([100.0, 200.0, 400.0]),
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

    with pytest.raises(ValueError, match="matching sampling semantics"):
        resample_spectrum(source, _plan(_points().axis))


def test_point_resampling_is_jittable() -> None:
    source = _points()
    target = SpectralAxis.points(
        jnp.array([150.0, 300.0]),
        coordinate=SpectralCoordinate.WAVELENGTH,
        unit="nm",
    )
    plan = _plan(target)

    result = jax.jit(lambda spectrum: resample_spectrum(spectrum, plan))(source)

    np.testing.assert_allclose(result.spectrum.values, [3.0, 6.0])
    assert int(result.status.code) == SpectrumStatusCode.OK


def test_outside_coverage_status_and_nan_payload_survive_jit() -> None:
    source = _points()
    target = SpectralAxis.points(
        jnp.array([90.0, 150.0]),
        coordinate=SpectralCoordinate.WAVELENGTH,
        unit="nm",
    )
    plan = _plan(target)

    result = jax.jit(lambda spectrum: resample_spectrum(spectrum, plan))(source)

    assert int(result.status.code) == SpectrumStatusCode.UNSUPPORTED_SPECTRAL_WINDOW
    assert bool(jnp.all(jnp.isnan(result.spectrum.values)))
