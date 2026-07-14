---
title: Spectral representations
---

# Spectral representations

## Owner import path

`jaxstro.spectra`

## Purpose

Generic spectral coordinates, sampling semantics, provenance, transformations,
resampling, and prepared interpolation stencils.

## Public records and callables

`SpectralAxis`, `SpectralCoordinate`, `SpectralSampling`, `SpectralSemantic`,
`SpectralPlan`, `Spectrum`, `SpectrumProvenance`, `SpectrumResult`,
`SpectrumStatus`, `SpectrumStatusCode`, `CoveragePolicy`, `FluxInterpolation`,
`PointResamplingMethod`, `PreparedRectilinearStencil`, `PreparedSimplexStencil`,
`resample_spectrum`, `surface_flux_to_luminosity`,
`surface_flux_to_observer_flux`, `to_flux_lambda`, `to_flux_nu`,
`to_frequency`, and `to_wavelength`.

## Shape and dtype expectations

Spectral coordinates and values are floating arrays with explicit coordinate,
density, sampling, and unit metadata. Prepared stencils have fixed topology and
shape.

## JAX transforms and AD classification

Fixed-shape evaluation and resampling compose with JAX transforms according to
the selected policy. Artifact loading, topology selection, and semantic choices
are preprocessing boundaries.

## Failure behavior

Coverage and conversion outcomes use explicit statuses. Jaxstro does not infer
filters, photometry, instruments, or model validity.

## Contract and evidence links

See [](../../30-representations/spectra-atmospheres/spectra-data-architecture.md),
[](../../30-representations/spectra-atmospheres/conservative-spectral-resampling.md),
and [](../../validation/spectra-performance.md).

## Canonical import example

```python
from jaxstro.spectra import resample_spectrum
```
