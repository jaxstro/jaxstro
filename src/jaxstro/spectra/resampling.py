"""Fail-closed point interpolation and conservative spectral remapping."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax.numpy as jnp

from jaxstro.numerics import conservative_remap_1d, interpolation
from jaxstro.numerics.checks import try_concrete_bool

from .plan import CoveragePolicy, PointResamplingMethod, SpectralPlan
from .types import (
    SpectralAxis,
    SpectralSampling,
    Spectrum,
    SpectrumResult,
    SpectrumStatus,
    SpectrumStatusCode,
)


def _axis_coordinates(axis: SpectralAxis):
    if axis.sampling is SpectralSampling.POINTS:
        return axis.values
    if axis.edges is None:  # protected by SpectralAxis invariants
        raise ValueError("binned spectral axes require edges")
    return axis.edges


def _axes_identical(source: SpectralAxis, target: SpectralAxis) -> bool:
    if (
        source.coordinate is not target.coordinate
        or source.unit != target.unit
        or source.sampling is not target.sampling
        or source.resolving_power != target.resolving_power
        or source.values.shape != target.values.shape
    ):
        return False
    values_equal = try_concrete_bool(jnp.array_equal(source.values, target.values))
    if values_equal is not True:
        return False
    if source.edges is None or target.edges is None:
        return source.edges is None and target.edges is None
    if source.edges.shape != target.edges.shape:
        return False
    return try_concrete_bool(jnp.array_equal(source.edges, target.edges)) is True


def _with_operation(spectrum: Spectrum, operation: str) -> Spectrum:
    provenance = replace(
        spectrum.provenance,
        operations=(*spectrum.provenance.operations, operation),
    )
    return replace(spectrum, provenance=provenance)


def _result(
    source: Spectrum,
    target_axis: SpectralAxis,
    values: Any,
    covered: Any,
    operation: str,
) -> SpectrumResult:
    output_values = jnp.where(covered, values, jnp.nan)
    provenance = replace(
        source.provenance,
        operations=(*source.provenance.operations, operation),
    )
    spectrum = Spectrum(
        axis=target_axis,
        values=output_values,
        semantic=source.semantic,
        provenance=provenance,
    )
    status_code = jnp.where(
        covered,
        int(SpectrumStatusCode.OK),
        int(SpectrumStatusCode.UNSUPPORTED_SPECTRAL_WINDOW),
    )
    return SpectrumResult(spectrum=spectrum, status=SpectrumStatus(status_code))


def resample_spectrum(spectrum: Spectrum, plan: SpectralPlan) -> SpectrumResult:
    """Resample onto a fixed target axis without extrapolation or zero filling."""
    target = plan.target_axis
    source = spectrum.axis
    if plan.coverage_policy is not CoveragePolicy.INTERSECTION:
        raise ValueError("unsupported spectral coverage policy")
    if source.coordinate is not target.coordinate or source.unit != target.unit:
        raise ValueError("resampling requires the same coordinate and unit")
    if source.sampling is not target.sampling:
        raise ValueError("resampling requires matching sampling semantics")

    if _axes_identical(source, target):
        return SpectrumResult(
            spectrum=_with_operation(spectrum, "resample:identity"),
            status=SpectrumStatus(SpectrumStatusCode.OK),
        )

    source_coordinates = _axis_coordinates(source)
    target_coordinates = _axis_coordinates(target)
    covered = (target_coordinates[0] >= source_coordinates[0]) & (
        target_coordinates[-1] <= source_coordinates[-1]
    )

    if source.sampling is SpectralSampling.POINTS:
        if plan.point_method is PointResamplingMethod.LINEAR:
            values = interpolation.interp1d(
                source.values,
                spectrum.values,
                target.values,
                extrapolate=False,
            )
            operation = "resample:linear-points"
        else:
            values = interpolation.monotone_cubic_interp(
                source.values,
                spectrum.values,
                target.values,
                extrapolate=False,
            )
            operation = "resample:monotone-cubic-points"
        return _result(
            spectrum,
            target,
            values,
            covered,
            operation,
        )
    if source.sampling is SpectralSampling.BIN_AVERAGES:
        if source.edges is None or target.edges is None:  # protected invariants
            raise ValueError("bin-average resampling requires bin edges")
        values = conservative_remap_1d(source.edges, spectrum.values, target.edges)
        return _result(
            spectrum,
            target,
            values,
            covered,
            "resample:conservative-bin-average",
        )
    raise ValueError("non-identity bin-integral resampling is not implemented")


__all__ = ["resample_spectrum"]
