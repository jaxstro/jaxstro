---
title: Spectra data architecture
description: >-
  The data-ingestion and runtime boundary for local atmosphere spectra in the
  jaxstro foundation layer.
---

`jaxstro.atmospheres` owns the foundation boundary

```text
stellar atmosphere parameters -> spectrum
```

It does not own synthetic photometry, bolometric corrections, survey rendering,
or downstream physical interpretation. Packages above jaxstro consume spectra
and decide how to integrate, project, calibrate, or compare them.

## Why this boundary

Atmosphere grids are primitive scientific inputs reused by multiple packages.
Every package should not parse PHOENIX/NewEra text files, rediscover local cache
layout, or reinvent the host/JAX boundary. At the same time, a foundation package
should not decide what a filter, magnitude, bolometric correction, or survey
observable means. Those are domain-level choices.

The public runtime boundary is therefore exactly:

```python
AtmosphereParams -> SpectrumResult
```

`AtmosphereParams` carries the stellar-atmosphere coordinates. `SpectrumResult`
returns a `Spectrum` plus a structured `SpectrumStatus`. The spectrum is raw
spectral data; downstream packages own any transformation from spectra into
observables.

## Data ingestion

Raw PHOENIX/NewEra text files are source data, not package data. The converter in
`scripts/convert_newera_lowres.py` reads the text rows, validates float32 storage
against float64 parses, and writes local processed artifacts:

```text
data/atmospheres/newera/processed/
  manifest.parquet
  catalog.parquet
  catalog_fragments/
  validation/
  newera_lowres_v3.zarr/
```

The processed catalog is the durable host-side index. The Zarr store is the dense
spectral artifact. Both remain local and gitignored.

## Runtime layers

The spectra interface has two layers.

1. `NewEraBackend` is a host-side convenience backend. It opens the local
   processed catalog and Zarr store, checks coverage, loads only the local cell
   needed for a query, and returns a spectrum.
2. `PreparedSpectralGrid` is the JAX-side runtime object. It already contains the
   local wavelength grid and corner spectra as arrays, so interpolation can be
   used with `jit`, `vmap`, and `grad`.

The convenience path is:

```python
from jaxstro.atmospheres import AtmosphereParams, NewEraBackend

backend = NewEraBackend.open()
result = backend.spectrum(
    AtmosphereParams(teff=5772.0, logg=4.44, m_h=0.0, alpha_m=0.0)
)
```

The inference path is:

```python
import jax
from jaxstro.atmospheres import AtmosphereParams, NewEraBackend

backend = NewEraBackend.open()
prepared = backend.prepare(
    AtmosphereParams(teff=5772.0, logg=4.44, m_h=0.0, alpha_m=0.0)
)

@jax.jit
def model(teff):
    params = AtmosphereParams(teff=teff, logg=4.44, m_h=0.0, alpha_m=0.0)
    return prepared.spectrum(params).spectrum.flux_lambda
```

## Status policy

The runtime is fail-closed. In-grid spectra report `SpectrumStatus.code == OK`.
Out-of-grid requests are never silently extrapolated: interpolation weights are
clamped to the prepared grid and the status records the non-OK condition.

The v1 backend supports exact abundance-plane selection (`m_h`, `alpha_m`) and
local bilinear interpolation over `teff` and `logg`. Metallicity and
alpha-enhancement interpolation are intentionally not hidden behind the v1 API;
that policy can be added only with explicit validation.

## Dataset policy

Datasets belong in jaxstro only when they are primitive reusable inputs, not
domain interpretations. Atmosphere spectra qualify because several packages can
share `stellar parameters -> spectrum`. Filters may qualify later as generic
spectral response curves, but photometric systems, zero-point semantics,
bolometric corrections, and survey-specific rendering stay in downstream
packages until a real shared lower-level abstraction emerges.
