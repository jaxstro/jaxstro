"""Atmosphere adapter registry and transactional preparation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp

from jaxstro.spectra import (
    PreparedRectilinearStencil,
    PreparedSimplexStencil,
    SpectralPlan,
    SpectrumProvenance,
    SpectrumResult,
    SpectrumStatusCode,
)

from .params import AtmosphereParams, AtmosphereQuery
from .products import ArtifactReport, ProductDescriptor

PreparedStencil = PreparedRectilinearStencil | PreparedSimplexStencil


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PreparedAtmosphere:
    """Filesystem-free prepared atmosphere evaluation state."""

    stencil: PreparedStencil
    parameter_names: tuple[str, ...]
    spectral_plan: SpectralPlan
    provenance: SpectrumProvenance

    def __post_init__(self) -> None:
        if not self.parameter_names or any(
            not name.strip() for name in self.parameter_names
        ):
            raise ValueError("prepared atmosphere requires parameter names")
        if len(self.parameter_names) != len(set(self.parameter_names)):
            raise ValueError("prepared atmosphere parameter names must be unique")
        target = self.spectral_plan.target_axis
        template = self.stencil.template.axis
        if (
            target.coordinate is not template.coordinate
            or target.unit != template.unit
            or target.sampling is not template.sampling
            or target.values.shape != template.values.shape
        ):
            raise ValueError("prepared stencil axis must match the spectral plan")

    def evaluate(self, params: AtmosphereParams) -> SpectrumResult:
        """Evaluate named physical parameters through the prepared stencil."""
        point = jnp.stack(
            [jnp.asarray(getattr(params, name)) for name in self.parameter_names]
        )
        return self.stencil.evaluate(point)

    def tree_flatten(self):
        children = (self.stencil, self.spectral_plan)
        aux_data = (self.parameter_names, self.provenance)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        parameter_names, provenance = aux_data
        stencil, spectral_plan = children
        return cls(stencil, parameter_names, spectral_plan, provenance)


@dataclass(frozen=True)
class PreparationResult:
    """Exactly one successful prepared object or unsuccessful status."""

    status: SpectrumStatusCode
    prepared: PreparedAtmosphere | None = None
    message: str = ""

    def __post_init__(self) -> None:
        status = SpectrumStatusCode(self.status)
        success = status is SpectrumStatusCode.OK
        if success != (self.prepared is not None):
            raise ValueError("preparation result must contain exactly one outcome")
        object.__setattr__(self, "status", status)

    @classmethod
    def success(cls, prepared: PreparedAtmosphere) -> PreparationResult:
        return cls(status=SpectrumStatusCode.OK, prepared=prepared)

    @classmethod
    def failure(
        cls,
        status: SpectrumStatusCode,
        message: str,
    ) -> PreparationResult:
        if status is SpectrumStatusCode.OK:
            raise ValueError("failure status cannot be OK")
        return cls(status=status, prepared=None, message=message)


class AtmosphereAdapter(Protocol):
    """Host-side contract implemented by one exact atmosphere product."""

    def describe_product(self) -> ProductDescriptor: ...

    def validate_artifact(self) -> ArtifactReport: ...

    def prepare(self, query: AtmosphereQuery) -> PreparationResult: ...


@dataclass(frozen=True)
class AtmosphereAdapterRegistry:
    """Exact-product adapter lookup with validation before artifact I/O."""

    adapters: tuple[AtmosphereAdapter, ...]

    def __post_init__(self) -> None:
        product_ids = tuple(
            adapter.describe_product().product_id for adapter in self.adapters
        )
        if len(product_ids) != len(set(product_ids)):
            raise ValueError("duplicate atmosphere product IDs are forbidden")

    def get(self, product_id: str) -> AtmosphereAdapter:
        for adapter in self.adapters:
            if adapter.describe_product().product_id == product_id:
                return adapter
        raise KeyError(f"unknown atmosphere product: {product_id}")

    def prepare(self, query: AtmosphereQuery) -> PreparationResult:
        try:
            adapter = self.get(query.product_id)
        except KeyError:
            return PreparationResult.failure(
                SpectrumStatusCode.NO_DATASET,
                f"unknown atmosphere product: {query.product_id}",
            )
        descriptor = adapter.describe_product()
        if query.family is not None and query.family != descriptor.family:
            return PreparationResult.failure(
                SpectrumStatusCode.NO_DATASET,
                "query family does not match exact product",
            )
        requested = query.requested_parameter_names or descriptor.parameter_names
        if tuple(requested) != descriptor.parameter_names:
            return PreparationResult.failure(
                SpectrumStatusCode.UNSUPPORTED_PLANE,
                "requested parameter plane is unsupported",
            )
        if descriptor.flux_interpolation_policy is None:
            return PreparationResult.failure(
                SpectrumStatusCode.POLICY_NOT_VALIDATED,
                "product interpolation policy has not been validated",
            )
        report = adapter.validate_artifact()
        if not report.valid:
            return PreparationResult.failure(
                SpectrumStatusCode.BACKEND_UNAVAILABLE,
                report.message or "artifact validation failed",
            )
        return adapter.prepare(query)


__all__ = [
    "AtmosphereAdapter",
    "AtmosphereAdapterRegistry",
    "PreparationResult",
    "PreparedAtmosphere",
]
