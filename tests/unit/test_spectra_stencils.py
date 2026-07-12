"""Tests for prepared spectral parameter-space interpolation stencils."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstro.spectra import (
    FluxInterpolation,
    PreparedRectilinearStencil,
    PreparedSimplexStencil,
    SpectralAxis,
    SpectralCoordinate,
    SpectralSemantic,
    Spectrum,
    SpectrumProvenance,
    SpectrumStatusCode,
)

PROVENANCE = SpectrumProvenance(
    source_id="synthetic",
    product_id="stencil",
    native_coordinate="wavelength_nm",
    native_density="F_lambda",
    native_unit="erg s^-1 cm^-2 nm^-1",
    canonical_conversion="identity",
    citations=("synthetic:test",),
)


def _template() -> Spectrum:
    return Spectrum(
        axis=SpectralAxis.points(
            jnp.array([100.0, 200.0, 400.0]),
            coordinate=SpectralCoordinate.WAVELENGTH,
            unit="nm",
        ),
        values=jnp.ones(3),
        semantic=SpectralSemantic.SURFACE_FLUX_LAMBDA,
        provenance=PROVENANCE,
    )


def _rectilinear_values() -> jax.Array:
    x = jnp.array([1.0, 3.0])
    y = jnp.array([10.0, 20.0, 50.0])
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    base = 2.0 * xx + 0.5 * yy
    return base[..., None] * jnp.array([1.0, 2.0, 4.0])


def _rectilinear(
    interpolation: FluxInterpolation = FluxInterpolation.LINEAR,
) -> PreparedRectilinearStencil:
    return PreparedRectilinearStencil(
        parameter_axes=(jnp.array([1.0, 3.0]), jnp.array([10.0, 20.0, 50.0])),
        vertex_values=_rectilinear_values(),
        template=_template(),
        interpolation=interpolation,
    )


def test_rectilinear_recovers_vertices_and_nonuniform_midpoint() -> None:
    stencil = _rectilinear()

    vertex = stencil.evaluate(jnp.array([1.0, 20.0]))
    midpoint = stencil.evaluate(jnp.array([2.0, 35.0]))

    np.testing.assert_allclose(vertex.spectrum.values, [12.0, 24.0, 48.0])
    np.testing.assert_allclose(midpoint.spectrum.values, [21.5, 43.0, 86.0])
    assert int(vertex.status.code) == SpectrumStatusCode.OK
    assert midpoint.spectrum.provenance.operations[-1] == (
        "parameter:rectilinear:linear"
    )


def test_rectilinear_rejects_missing_or_wrong_corner_payload() -> None:
    missing = _rectilinear_values().at[1, 1, 0].set(jnp.nan)

    with pytest.raises(ValueError, match="finite complete corner grid"):
        PreparedRectilinearStencil(
            parameter_axes=(jnp.array([1.0, 3.0]), jnp.array([10.0, 20.0, 50.0])),
            vertex_values=missing,
            template=_template(),
        )
    with pytest.raises(ValueError, match="leading shape"):
        PreparedRectilinearStencil(
            parameter_axes=(jnp.array([1.0, 3.0]), jnp.array([10.0, 20.0, 50.0])),
            vertex_values=jnp.ones((2, 2, 3)),
            template=_template(),
        )


def test_rectilinear_outside_domain_returns_nan_status() -> None:
    result = _rectilinear().evaluate(jnp.array([0.5, 35.0]))

    assert int(result.status.code) == SpectrumStatusCode.OUTSIDE_CONVEX_HULL
    assert bool(jnp.all(jnp.isnan(result.spectrum.values)))


def test_positive_log_policy_interpolates_geometrically() -> None:
    values = jnp.array([[1.0, 4.0, 16.0], [9.0, 36.0, 144.0]])
    stencil = PreparedRectilinearStencil(
        parameter_axes=(jnp.array([1.0, 3.0]),),
        vertex_values=values,
        template=_template(),
        interpolation=FluxInterpolation.POSITIVE_LOG,
    )

    result = stencil.evaluate(jnp.array([2.0]))

    np.testing.assert_allclose(result.spectrum.values, [3.0, 12.0, 48.0])
    assert result.spectrum.provenance.operations[-1] == (
        "parameter:rectilinear:positive_log"
    )


def test_positive_log_rejects_nonpositive_values() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        PreparedRectilinearStencil(
            parameter_axes=(jnp.array([1.0, 3.0]),),
            vertex_values=jnp.array([[1.0, 2.0, 3.0], [0.0, 4.0, 5.0]]),
            template=_template(),
            interpolation=FluxInterpolation.POSITIVE_LOG,
        )


def _simplex(
    interpolation: FluxInterpolation = FluxInterpolation.LINEAR,
) -> PreparedSimplexStencil:
    return PreparedSimplexStencil(
        vertices=jnp.array([[0.0, 0.0], [2.0, 0.0], [0.0, 4.0]]),
        vertex_values=jnp.array([[1.0, 2.0, 4.0], [3.0, 6.0, 12.0], [5.0, 10.0, 20.0]]),
        template=_template(),
        interpolation=interpolation,
    )


def test_simplex_recovers_vertices_and_barycentric_interior() -> None:
    stencil = _simplex()

    vertex = stencil.evaluate(jnp.array([2.0, 0.0]))
    interior = stencil.evaluate(jnp.array([0.5, 1.0]))

    np.testing.assert_allclose(vertex.spectrum.values, [3.0, 6.0, 12.0])
    np.testing.assert_allclose(interior.spectrum.values, [2.5, 5.0, 10.0])
    assert int(interior.status.code) == SpectrumStatusCode.OK
    assert interior.spectrum.provenance.operations[-1] == "parameter:simplex:linear"


def test_simplex_outside_hull_returns_nan_status() -> None:
    result = _simplex().evaluate(jnp.array([2.0, 4.0]))

    assert int(result.status.code) == SpectrumStatusCode.OUTSIDE_CONVEX_HULL
    assert bool(jnp.all(jnp.isnan(result.spectrum.values)))


def test_simplex_rejects_singular_or_wrong_vertex_sets() -> None:
    with pytest.raises(ValueError, match="exactly dimension plus one"):
        PreparedSimplexStencil(
            vertices=jnp.array([[0.0, 0.0], [1.0, 0.0]]),
            vertex_values=jnp.ones((2, 3)),
            template=_template(),
        )
    with pytest.raises(ValueError, match="nonsingular"):
        PreparedSimplexStencil(
            vertices=jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            vertex_values=jnp.ones((3, 3)),
            template=_template(),
        )


def test_stencils_are_jittable_and_vmappable() -> None:
    stencil = _simplex()
    points = jnp.array([[0.5, 1.0], [1.0, 1.0]])

    jitted = jax.jit(lambda point: stencil.evaluate(point))(points[0])
    vmapped_values, vmapped_status = jax.vmap(
        lambda point: (
            stencil.evaluate(point).spectrum.values,
            stencil.evaluate(point).status.code,
        )
    )(points)

    np.testing.assert_allclose(jitted.spectrum.values, [2.5, 5.0, 10.0])
    assert vmapped_values.shape == (2, 3)
    np.testing.assert_array_equal(vmapped_status, [0, 0])
