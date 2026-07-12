"""Tests for the canonical domain-neutral spectrum data model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstro.spectra import (
    SpectralAxis,
    SpectralCoordinate,
    SpectralSampling,
    SpectralSemantic,
    Spectrum,
    SpectrumProvenance,
    SpectrumResult,
    SpectrumStatus,
    SpectrumStatusCode,
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


def _surface_spectrum(values: object = (2.0, 3.0)) -> Spectrum:
    axis = SpectralAxis.points(
        jnp.array([1.0e-5, 2.0e-5]),
        coordinate=SpectralCoordinate.WAVELENGTH,
        unit="cm",
    )
    return Spectrum(
        axis=axis,
        values=jnp.asarray(values),
        semantic=SpectralSemantic.SURFACE_FLUX_LAMBDA,
        provenance=_provenance(),
    )


def test_point_spectrum_is_a_roundtrip_pytree() -> None:
    spectrum = _surface_spectrum()

    leaves, treedef = jax.tree_util.tree_flatten(spectrum)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert len(leaves) == 2
    np.testing.assert_array_equal(rebuilt.axis.values, spectrum.axis.values)
    np.testing.assert_array_equal(rebuilt.values, spectrum.values)
    assert rebuilt.semantic is spectrum.semantic
    assert rebuilt.provenance == spectrum.provenance
    assert rebuilt.axis.sampling is SpectralSampling.POINTS


def test_binned_axis_computes_centers_and_preserves_edges() -> None:
    axis = SpectralAxis.bins(
        jnp.array([1.0, 2.0, 4.0]),
        coordinate=SpectralCoordinate.WAVELENGTH,
        unit="cm",
        sampling=SpectralSampling.BIN_AVERAGES,
    )

    np.testing.assert_allclose(axis.values, [1.5, 3.0])
    np.testing.assert_allclose(axis.edges, [1.0, 2.0, 4.0])
    assert axis.sampling is SpectralSampling.BIN_AVERAGES


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (jnp.ones((2, 2)), "one-dimensional"),
        (jnp.array([1.0, jnp.nan]), "finite"),
        (jnp.array([0.0, 1.0]), "positive"),
        (jnp.array([2.0, 1.0]), "strictly increasing"),
        (jnp.array([1.0, 1.0]), "strictly increasing"),
    ],
)
def test_point_axis_rejects_invalid_coordinates(values: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SpectralAxis.points(
            values,
            coordinate=SpectralCoordinate.WAVELENGTH,
            unit="cm",
        )


def test_binned_axis_rejects_too_few_edges() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        SpectralAxis.bins(
            jnp.array([1.0]),
            coordinate=SpectralCoordinate.WAVELENGTH,
            unit="cm",
        )


def test_spectrum_rejects_wrong_value_shape_and_coordinate_semantic() -> None:
    wavelength = SpectralAxis.points(
        jnp.array([1.0, 2.0]),
        coordinate=SpectralCoordinate.WAVELENGTH,
        unit="cm",
    )

    with pytest.raises(ValueError, match="match the spectral axis"):
        Spectrum(
            axis=wavelength,
            values=jnp.array([1.0]),
            semantic=SpectralSemantic.SURFACE_FLUX_LAMBDA,
            provenance=_provenance(),
        )
    with pytest.raises(ValueError, match="requires a frequency axis"):
        Spectrum(
            axis=wavelength,
            values=jnp.array([1.0, 2.0]),
            semantic=SpectralSemantic.SURFACE_FLUX_NU,
            provenance=_provenance(),
        )


def test_success_requires_finite_values_but_failure_uses_nan_payload() -> None:
    invalid_spectrum = _surface_spectrum((jnp.nan, jnp.nan))

    with pytest.raises(ValueError, match="successful spectrum values must be finite"):
        SpectrumResult(
            spectrum=invalid_spectrum,
            status=SpectrumStatus(SpectrumStatusCode.OK),
        )

    result = SpectrumResult(
        spectrum=invalid_spectrum,
        status=SpectrumStatus(SpectrumStatusCode.OUTSIDE_CONVEX_HULL),
    )
    assert int(result.status.code) == SpectrumStatusCode.OUTSIDE_CONVEX_HULL
    assert bool(jnp.all(jnp.isnan(result.spectrum.values)))


def test_spectrum_and_result_survive_jit_identity() -> None:
    result = SpectrumResult(
        spectrum=_surface_spectrum(),
        status=SpectrumStatus(SpectrumStatusCode.OK),
    )

    rebuilt = jax.jit(lambda value: value)(result)

    np.testing.assert_array_equal(rebuilt.spectrum.values, [2.0, 3.0])
    assert int(rebuilt.status.code) == SpectrumStatusCode.OK
    assert rebuilt.spectrum.provenance == result.spectrum.provenance


def test_status_code_registry_is_stable() -> None:
    assert [(status.name, status.value) for status in SpectrumStatusCode] == [
        ("OK", 0),
        ("NO_DATASET", 1),
        ("NO_COVERAGE", 2),
        ("NO_COMPLETE_CELL", 3),
        ("OUTSIDE_CONVEX_HULL", 4),
        ("UNSUPPORTED_PLANE", 5),
        ("UNSUPPORTED_SPECTRAL_WINDOW", 6),
        ("BACKEND_UNAVAILABLE", 7),
        ("POLICY_NOT_VALIDATED", 8),
    ]


def test_status_supports_batched_codes_under_vmap() -> None:
    def make_status(code: object) -> SpectrumStatus:
        return SpectrumStatus(code)

    status = jax.vmap(make_status)(
        jnp.array([SpectrumStatusCode.OK, SpectrumStatusCode.OUTSIDE_CONVEX_HULL])
    )

    np.testing.assert_array_equal(
        status.code,
        [SpectrumStatusCode.OK, SpectrumStatusCode.OUTSIDE_CONVEX_HULL],
    )
    np.testing.assert_array_equal(status.ok, [True, False])
