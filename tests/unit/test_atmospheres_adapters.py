"""Tests for atmosphere product, preparation, and adapter contracts."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import pytest

from jaxstro.atmospheres import (
    ArtifactReport,
    AtmosphereAdapterRegistry,
    AtmosphereParams,
    AtmosphereQuery,
    PreparationResult,
    PreparedAtmosphere,
    ProductDescriptor,
)
from jaxstro.spectra import (
    PreparedRectilinearStencil,
    SpectralAxis,
    SpectralCoordinate,
    SpectralPlan,
    SpectralSemantic,
    Spectrum,
    SpectrumProvenance,
    SpectrumStatusCode,
)


def _plan() -> SpectralPlan:
    return SpectralPlan(
        SpectralAxis.points(
            jnp.array([100.0, 200.0]),
            coordinate=SpectralCoordinate.WAVELENGTH,
            unit="nm",
        )
    )


def _provenance() -> SpectrumProvenance:
    return SpectrumProvenance(
        source_id="synthetic",
        product_id="synthetic-product",
        native_coordinate="wavelength_nm",
        native_density="F_lambda",
        native_unit="erg s^-1 cm^-2 nm^-1",
        canonical_conversion="identity",
        citations=("synthetic:test",),
    )


def _prepared() -> PreparedAtmosphere:
    template = Spectrum(
        axis=_plan().target_axis,
        values=jnp.ones(2),
        semantic=SpectralSemantic.SURFACE_FLUX_LAMBDA,
        provenance=_provenance(),
    )
    stencil = PreparedRectilinearStencil(
        parameter_axes=(jnp.array([5000.0, 6000.0]),),
        vertex_values=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        template=template,
    )
    return PreparedAtmosphere(
        stencil=stencil,
        parameter_names=("teff",),
        spectral_plan=_plan(),
        provenance=_provenance(),
    )


def test_atmosphere_query_requires_explicit_product_and_plan() -> None:
    query = AtmosphereQuery(
        params=AtmosphereParams(teff=5500.0, logg=4.5),
        product_id="synthetic-product",
        spectral_plan=_plan(),
        family="synthetic",
    )

    assert query.product_id == "synthetic-product"
    assert query.family == "synthetic"
    assert query.params.c_o == 0.55


def test_product_descriptor_is_hashable_and_policy_explicit() -> None:
    descriptor = ProductDescriptor(
        product_id="synthetic-product",
        family="synthetic",
        parameter_names=("teff", "logg"),
        topology_policy="complete-cell-or-approved-simplex",
        flux_interpolation_policy="linear",
        provenance_id="synthetic-card",
    )

    assert hash(descriptor)
    assert descriptor.flux_interpolation_policy == "linear"


def test_artifact_report_requires_evidence_when_valid() -> None:
    report = ArtifactReport(valid=True, digest="sha256:abc", schema="synthetic-v1")

    assert report.valid
    with pytest.raises(ValueError, match="digest and schema"):
        ArtifactReport(valid=True)


def test_preparation_result_contains_exactly_one_outcome() -> None:
    prepared = _prepared()
    success = PreparationResult.success(prepared)
    failure = PreparationResult.failure(
        SpectrumStatusCode.NO_COMPLETE_CELL,
        "no complete cell",
    )

    assert success.prepared is prepared
    assert success.status is SpectrumStatusCode.OK
    assert failure.prepared is None
    assert failure.status is SpectrumStatusCode.NO_COMPLETE_CELL
    with pytest.raises(ValueError, match="exactly one"):
        PreparationResult(status=SpectrumStatusCode.OK, prepared=None)
    with pytest.raises(ValueError, match="exactly one"):
        PreparationResult(
            status=SpectrumStatusCode.NO_COMPLETE_CELL,
            prepared=prepared,
        )


def test_prepared_atmosphere_evaluates_named_parameters_without_paths() -> None:
    prepared = _prepared()

    result = prepared.evaluate(AtmosphereParams(teff=5500.0, logg=4.5))

    assert not hasattr(prepared, "path")
    assert not hasattr(prepared, "store")
    assert int(result.status.code) == SpectrumStatusCode.OK
    assert jnp.allclose(result.spectrum.values, jnp.array([2.0, 3.0]))


@dataclass(frozen=True)
class _Adapter:
    descriptor: ProductDescriptor

    def describe_product(self) -> ProductDescriptor:
        return self.descriptor

    def validate_artifact(self) -> ArtifactReport:
        return ArtifactReport(valid=True, digest="sha256:abc", schema="synthetic-v1")

    def prepare(self, query: AtmosphereQuery) -> PreparationResult:
        del query
        return PreparationResult.success(_prepared())


def _adapter(product_id: str = "synthetic-product") -> _Adapter:
    return _Adapter(
        ProductDescriptor(
            product_id=product_id,
            family="synthetic",
            parameter_names=("teff",),
            topology_policy="complete-cell",
            flux_interpolation_policy="linear",
            provenance_id="synthetic-card",
        )
    )


def test_registry_resolves_exact_product_and_rejects_duplicates() -> None:
    registry = AtmosphereAdapterRegistry((_adapter(),))

    assert registry.get("synthetic-product").describe_product().family == "synthetic"
    with pytest.raises(KeyError, match="unknown atmosphere product"):
        registry.get("other-product")
    with pytest.raises(ValueError, match="duplicate atmosphere product"):
        AtmosphereAdapterRegistry((_adapter(), _adapter()))


def test_registry_returns_policy_and_plane_failures_before_adapter_io() -> None:
    no_policy = _Adapter(
        ProductDescriptor(
            product_id="no-policy",
            family="synthetic",
            parameter_names=("teff",),
            topology_policy="complete-cell",
            flux_interpolation_policy=None,
            provenance_id="synthetic-card",
        )
    )
    registry = AtmosphereAdapterRegistry((no_policy,))
    query = AtmosphereQuery(
        params=AtmosphereParams(teff=5500.0, logg=4.5),
        product_id="no-policy",
        spectral_plan=_plan(),
        requested_parameter_names=("teff", "logg"),
    )

    result = registry.prepare(query)

    assert result.status is SpectrumStatusCode.UNSUPPORTED_PLANE

    policy_query = AtmosphereQuery(
        params=query.params,
        product_id="no-policy",
        spectral_plan=query.spectral_plan,
        requested_parameter_names=("teff",),
    )
    policy_result = registry.prepare(policy_query)
    assert policy_result.status is SpectrumStatusCode.POLICY_NOT_VALIDATED
