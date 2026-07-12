"""Canonical, domain-neutral spectral containers and status codes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Any

import jax
import jax.numpy as jnp

from jaxstro.numerics.checks import try_concrete_bool


class SpectralCoordinate(StrEnum):
    """Coordinate used by a spectral axis."""

    WAVELENGTH = "wavelength"
    FREQUENCY = "frequency"


class SpectralSampling(StrEnum):
    """Meaning of samples associated with a spectral axis."""

    POINTS = "points"
    BIN_AVERAGES = "bin_averages"
    BIN_INTEGRALS = "bin_integrals"


class SpectralSemantic(StrEnum):
    """Physical meaning of spectral values."""

    SURFACE_FLUX_LAMBDA = "surface_flux_lambda"
    SURFACE_FLUX_NU = "surface_flux_nu"
    LUMINOSITY_LAMBDA = "luminosity_lambda"
    LUMINOSITY_NU = "luminosity_nu"
    OBSERVER_FLUX_LAMBDA = "observer_flux_lambda"
    OBSERVER_FLUX_NU = "observer_flux_nu"


class SpectrumStatusCode(IntEnum):
    """Stable status registry for expected scientific outcomes."""

    OK = 0
    NO_DATASET = 1
    NO_COVERAGE = 2
    NO_COMPLETE_CELL = 3
    OUTSIDE_CONVEX_HULL = 4
    UNSUPPORTED_PLANE = 5
    UNSUPPORTED_SPECTRAL_WINDOW = 6
    BACKEND_UNAVAILABLE = 7
    POLICY_NOT_VALIDATED = 8


@dataclass(frozen=True)
class SpectrumProvenance:
    """Hashable reconstruction record carried as static PyTree metadata."""

    source_id: str
    product_id: str
    native_coordinate: str
    native_density: str
    native_unit: str
    canonical_conversion: str
    citations: tuple[str, ...]
    artifact_digest: str | None = None
    operations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        fields = (
            self.source_id,
            self.product_id,
            self.native_coordinate,
            self.native_density,
            self.native_unit,
            self.canonical_conversion,
        )
        if any(not value.strip() for value in fields):
            raise ValueError("spectrum provenance fields must be non-empty")
        if not self.citations or any(
            not citation.strip() for citation in self.citations
        ):
            raise ValueError("spectrum provenance requires non-empty citations")


def _require_concrete(predicate: Any, message: str) -> None:
    result = try_concrete_bool(predicate)
    if result is False:
        raise ValueError(message)


def _validate_coordinates(values: Any, *, name: str, minimum_size: int) -> Any:
    array = jnp.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.shape[0] < minimum_size:
        raise ValueError(f"{name} must contain at least {minimum_size} entries")
    _require_concrete(jnp.all(jnp.isfinite(array)), f"{name} must be finite")
    _require_concrete(jnp.all(array > 0), f"{name} must be positive")
    _require_concrete(
        jnp.all(jnp.diff(array) > 0), f"{name} must be strictly increasing"
    )
    return array


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SpectralAxis:
    """Increasing wavelength or frequency coordinates with sampling semantics."""

    values: Any
    coordinate: SpectralCoordinate
    unit: str
    sampling: SpectralSampling = SpectralSampling.POINTS
    edges: Any | None = None
    resolving_power: float | None = None

    def __post_init__(self) -> None:
        coordinate = SpectralCoordinate(self.coordinate)
        sampling = SpectralSampling(self.sampling)
        values = _validate_coordinates(self.values, name="axis values", minimum_size=1)
        if not self.unit.strip():
            raise ValueError("spectral axis unit must be non-empty")
        if self.resolving_power is not None and self.resolving_power <= 0:
            raise ValueError("resolving power must be positive")

        edges = self.edges
        if sampling is SpectralSampling.POINTS:
            if edges is not None:
                raise ValueError("point-sampled axes cannot define bin edges")
        else:
            if edges is None:
                raise ValueError("binned axes require bin edges")
            edges = _validate_coordinates(edges, name="bin edges", minimum_size=2)
            if edges.shape[0] != values.shape[0] + 1:
                raise ValueError(
                    "binned axes require exactly one more edge than centers"
                )

        object.__setattr__(self, "coordinate", coordinate)
        object.__setattr__(self, "sampling", sampling)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "edges", edges)

    @classmethod
    def points(
        cls,
        values: Any,
        *,
        coordinate: SpectralCoordinate,
        unit: str,
        resolving_power: float | None = None,
    ) -> SpectralAxis:
        """Construct a point-sampled axis."""
        return cls(
            values=values,
            coordinate=coordinate,
            unit=unit,
            sampling=SpectralSampling.POINTS,
            resolving_power=resolving_power,
        )

    @classmethod
    def bins(
        cls,
        edges: Any,
        *,
        coordinate: SpectralCoordinate,
        unit: str,
        sampling: SpectralSampling = SpectralSampling.BIN_AVERAGES,
        resolving_power: float | None = None,
    ) -> SpectralAxis:
        """Construct a bin-average or bin-integral axis from increasing edges."""
        sampling = SpectralSampling(sampling)
        if sampling is SpectralSampling.POINTS:
            raise ValueError("binned axes require bin-average or bin-integral sampling")
        edge_array = _validate_coordinates(edges, name="bin edges", minimum_size=2)
        values = 0.5 * (edge_array[:-1] + edge_array[1:])
        return cls(
            values=values,
            coordinate=coordinate,
            unit=unit,
            sampling=sampling,
            edges=edge_array,
            resolving_power=resolving_power,
        )

    def tree_flatten(self):
        children = (self.values, self.edges)
        aux_data = (
            self.coordinate,
            self.unit,
            self.sampling,
            self.resolving_power,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        coordinate, unit, sampling, resolving_power = aux_data
        values, edges = children
        return cls(
            values=values,
            coordinate=coordinate,
            unit=unit,
            sampling=sampling,
            edges=edges,
            resolving_power=resolving_power,
        )


_WAVELENGTH_SEMANTICS = {
    SpectralSemantic.SURFACE_FLUX_LAMBDA,
    SpectralSemantic.LUMINOSITY_LAMBDA,
    SpectralSemantic.OBSERVER_FLUX_LAMBDA,
}
_FREQUENCY_SEMANTICS = {
    SpectralSemantic.SURFACE_FLUX_NU,
    SpectralSemantic.LUMINOSITY_NU,
    SpectralSemantic.OBSERVER_FLUX_NU,
}


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Spectrum:
    """Spectral values with explicit axis, physical semantics, and provenance."""

    axis: SpectralAxis
    values: Any
    semantic: SpectralSemantic
    provenance: SpectrumProvenance

    def __post_init__(self) -> None:
        values = jnp.asarray(self.values)
        semantic = SpectralSemantic(self.semantic)
        if values.ndim != 1 or values.shape != self.axis.values.shape:
            raise ValueError(
                "spectrum values must be one-dimensional and match the spectral axis"
            )
        if (
            semantic in _WAVELENGTH_SEMANTICS
            and self.axis.coordinate is not SpectralCoordinate.WAVELENGTH
        ):
            raise ValueError(f"{semantic.value} requires a wavelength axis")
        if (
            semantic in _FREQUENCY_SEMANTICS
            and self.axis.coordinate is not SpectralCoordinate.FREQUENCY
        ):
            raise ValueError(f"{semantic.value} requires a frequency axis")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "semantic", semantic)

    def tree_flatten(self):
        children = (self.axis, self.values)
        aux_data = (self.semantic, self.provenance)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        semantic, provenance = aux_data
        axis, values = children
        return cls(
            axis=axis,
            values=values,
            semantic=semantic,
            provenance=provenance,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SpectrumStatus:
    """Array-compatible status for one spectral result."""

    code: Any

    def __post_init__(self) -> None:
        code = jnp.asarray(self.code, dtype=jnp.int32)
        object.__setattr__(self, "code", code)

    @property
    def ok(self):
        """Return whether this status is successful."""
        return self.code == int(SpectrumStatusCode.OK)

    def tree_flatten(self):
        return (self.code,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        (code,) = children
        return cls(code=code)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SpectrumResult:
    """Fixed-shape spectrum payload plus an expected-outcome status."""

    spectrum: Spectrum
    status: SpectrumStatus

    def __post_init__(self) -> None:
        successful = try_concrete_bool(self.status.ok)
        if successful:
            finite = try_concrete_bool(jnp.all(jnp.isfinite(self.spectrum.values)))
            if finite is False:
                raise ValueError("successful spectrum values must be finite")

    def tree_flatten(self):
        return (self.spectrum, self.status), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        spectrum, status = children
        return cls(spectrum=spectrum, status=status)


__all__ = [
    "SpectralAxis",
    "SpectralCoordinate",
    "SpectralSampling",
    "SpectralSemantic",
    "Spectrum",
    "SpectrumProvenance",
    "SpectrumResult",
    "SpectrumStatus",
    "SpectrumStatusCode",
]
