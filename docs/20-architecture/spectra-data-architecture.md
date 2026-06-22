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

Raw atmosphere downloads are source data, not package data. jaxstro's
converters read staged source archives, validate local processed artifacts
against parsed source values, and preserve the source archives unless an
explicit, validated deletion policy exists for a particular converter.

The processed side always has the same shape:

```text
data/atmospheres/<family>/.../processed/
  catalog.parquet
  catalog_fragments/
  validation/
  *.zarr/
```

The processed catalog is the durable host-side index. The Zarr store is the
spectral artifact. The validation JSON is the readback ledger: source hashes,
counts, units, float32 roundoff, and raw-archive preservation. All processed
artifacts remain local and gitignored.

See [](./atmosphere-capabilities.md) for the measured local dataset matrix.

## Runtime layers

The spectra interface has three layers.

1. `AtmosphereLibrary` is the catalog-first selector. It summarizes processed
   and staged local datasets, ranks coverage candidates, and reports whether a
   backend is available. It never pretends raw-only or artifact-only data can be
   used as a runtime backend.
2. Host-side backends open a processed catalog and Zarr store, check coverage,
   load only the local cell needed for a query, and return a spectrum. NewEra and
   BOSZ have implemented backends today. Sonora and TLUSTY have validated
   processed schemas and coverage records, but their runtime backends should be
   added only after their interpolation and spectral-density policies are
   explicitly tested.
3. `PreparedSpectralGrid` is the JAX-side runtime object. It already contains the
   local wavelength grid and corner spectra as arrays, so interpolation can be
   used with `jit`, `vmap`, and `grad`.

The catalog-first path is:

```python
from jaxstro.atmospheres import AtmosphereLibrary, AtmosphereParams

library = AtmosphereLibrary.from_local("data")
selection = library.select(
    AtmosphereParams(teff=5772.0, logg=4.44, m_h=0.0, alpha_m=0.0)
)

if selection.status == "ok":
    result = library.spectrum(selection.requested)
```

The backend-specific inference path is:

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
alpha-enhancement interpolation are intentionally not hidden behind the API; that
policy can be added only with explicit validation.

`AtmosphereLibrary.select(...)` has a similar fail-closed policy at the catalog
layer:

- `ok` means a processed dataset covers the request and an implemented backend is
  available.
- `backend_unavailable` means coverage exists, but jaxstro does not yet have a
  runtime backend for that processed schema.
- `no_match` means the loaded catalogs do not cover the requested coordinates or
  filters.

Those statuses are useful downstream because they distinguish "no model in the
local library" from "the model exists locally but the runtime policy is not
implemented yet."

## Dataset policy

Datasets belong in jaxstro only when they are primitive reusable inputs, not
domain interpretations. Atmosphere spectra qualify because several packages can
share `stellar parameters -> spectrum`. Filters may qualify later as generic
spectral response curves, but photometric systems, zero-point semantics,
bolometric corrections, and survey-specific rendering stay in downstream
packages until a real shared lower-level abstraction emerges.

That policy is why TLUSTY keeps native `frequency_hz` and `F_nu` in processed
artifacts, while runtime conversion to a wavelength-domain `SpectrumResult`
requires a separate backend test. It is also why Sonora keeps released
`W/m2/m` units explicit instead of normalizing into a photometric convention.
