---
title: Spectra data architecture
description: >-
  The host-to-JAX boundary for local atmosphere catalogs, prepared spectral
  grids, and raw spectra in jaxstro.
---

Atmosphere spectra begin as files but become useful inside a model as arrays.
The difficult architectural question is therefore not only *which spectrum?*
but also *which layer is allowed to touch the filesystem?*

`jaxstro.atmospheres` owns one deliberately narrow foundation boundary:

```{code-block} text
:caption: Interface notation, not Python

AtmosphereParams -> SpectrumResult
```

`AtmosphereParams` carries atmosphere-grid coordinates. `SpectrumResult`
contains raw wavelength and flux arrays plus a structured `SpectrumStatus`.
Downstream packages—not jaxstro—own filters, magnitudes, bolometric corrections,
survey rendering, and physical interpretation.

:::{figure} ./figures/spectra-runtime-boundary.webp
:name: fig-spectra-runtime-boundary
:alt: Three-stage spectra workflow from host-side catalog and artifact preparation through a JAX-ready local grid to downstream observables

The filesystem boundary is one-way. Host code discovers catalogs and loads a
small interpolation cell; the resulting `PreparedSpectralGrid` can enter JAX
transforms without reopening local artifacts. A downstream package may then turn
the raw spectrum into an observable. The measured strip uses the portable fixture
below; it does not report local dataset coverage.
:::

## Three execution layers

```{list-table} Spectra execution and ownership boundaries
:header-rows: 1
:label: tbl-spectra-execution-boundaries

* - Operation
  - Execution side
  - Owner and contract
* - Catalog discovery and candidate ranking
  - Host-side Python
  - `AtmosphereLibrary` reads local metadata, ranks coverage, and distinguishes
    usable backends from coverage-only records.
* - Artifact opening and local-cell preparation
  - Host-side Python with optional data dependencies
  - NewEra and BOSZ backends open Parquet/Zarr artifacts and prepare only the
    interpolation cell surrounding a request.
* - Prepared-grid interpolation
  - JAX-side array computation
  - `PreparedSpectralGrid.spectrum(...)` supports `jit`, `vmap`, and pathwise
    gradients within one fixed abundance plane and interpolation cell.
* - Synthetic photometry and interpretation
  - Downstream package
  - Filters, zero points, magnitudes, bolometric corrections, and survey
    observables remain outside jaxstro.
```

This separation keeps file discovery and discrete cell selection outside a
compiled model while allowing the prepared numerical kernel to remain a PyTree.
Changing the selected cell or abundance plane is a host-side event, not a smooth
scientific derivative.

## Portable JAX-side example

This complete example creates its arrays in memory. It needs no atmosphere
download, catalog, Zarr store, or optional data dependency.

```python
import jax
import jax.numpy as jnp

from jaxstro.atmospheres import (
    STATUS_MISSING_ABUNDANCE,
    STATUS_OK,
    STATUS_OUT_OF_GRID,
    AtmosphereParams,
    PreparedSpectralGrid,
)

prepared = PreparedSpectralGrid(
    teff=jnp.array([5000.0, 6000.0]),
    logg=jnp.array([4.0, 5.0]),
    wavelength=jnp.array([100.0, 101.0, 102.0]),
    flux=jnp.array(
        [
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
            [[3.0, 4.0, 5.0], [4.0, 5.0, 6.0]],
        ]
    ),
    m_h=0.0,
    alpha_m=0.0,
)

midpoint = prepared.spectrum(AtmosphereParams(teff=5500.0, logg=4.5))
outside = prepared.spectrum(AtmosphereParams(teff=4500.0, logg=4.5))
wrong_plane = prepared.spectrum(
    AtmosphereParams(teff=5500.0, logg=4.5, m_h=0.5)
)

@jax.jit
def first_flux(teff):
    params = AtmosphereParams(teff=teff, logg=4.0)
    return prepared.spectrum(params).spectrum.flux_lambda[0]

local_slope = jax.grad(first_flux)(5500.0)

assert jnp.allclose(midpoint.spectrum.flux_lambda, jnp.array([2.5, 3.5, 4.5]))
assert int(midpoint.status.code) == STATUS_OK
assert int(outside.status.code) == STATUS_OUT_OF_GRID
assert bool(outside.status.clamped)
assert int(wrong_plane.status.code) == STATUS_MISSING_ABUNDANCE
assert jnp.isclose(local_slope, 0.002)
```

The midpoint lies inside the prepared rectangle, so bilinear interpolation is a
smooth pathwise calculation there. The low-temperature request is clamped to the
nearest prepared boundary and marked non-OK rather than silently extrapolated.
The metallicity request is also non-OK because this grid represents exactly one
abundance plane.

The derivative is local to the fixed cell. It does not differentiate catalog
ranking, file loading, cell selection, or a transition between abundance planes.

## Local catalog recipe

**Execution contract — local processed artifacts required.** This recipe also
requires the `data` optional dependencies. It is intentionally not exercised by
the portable documentation test because `data/atmospheres/` is local and
gitignored.

```python
from jaxstro.atmospheres import AtmosphereLibrary, AtmosphereParams

library = AtmosphereLibrary.from_local("data")
request = AtmosphereParams(teff=5772.0, logg=4.44, m_h=0.0, alpha_m=0.0)
selection = library.select(request)

if selection.status == "ok":
    result = library.spectrum(request)
```

Selection has three fail-closed outcomes:

- `ok`: a processed dataset covers the request and an implemented backend is
  available;
- `backend_unavailable`: coverage exists, but its runtime policy is not
  implemented;
- `no_match`: no loaded catalog satisfies the requested coordinates or filters.

These states distinguish “not represented locally” from “represented, but not
yet executable.”

## Backend-specific preparation recipe

**Execution contract — local processed artifacts required.** `NewEraBackend.open()`
must resolve a compatible catalog and Zarr store before the prepared object can
enter a compiled model.

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

Opening and preparation stay outside `model`. The jitted function closes over
arrays and static unit metadata, not file handles or catalog queries.

## Data ingestion and preservation

Raw downloads are source data, not package data. Converters read staged source
archives, write processed catalogs and spectral arrays, validate readback against
parsed source values, and preserve the archives unless a separately validated
deletion policy exists.

```text
data/atmospheres/<family>/.../processed/
  catalog.parquet
  catalog_fragments/
  validation/
  *.zarr/
```

The catalog is the durable host-side index. Zarr stores the spectral arrays. The
validation ledger records source hashes, counts, units, float32 roundoff, and
archive preservation. These artifacts remain local and gitignored.

## Current capability boundary

Atmosphere support remains in progress. NewEra and BOSZ have host-side runtime
backends. Sonora and TLUSTY have validated processed schemas and coverage records,
but Sonora and TLUSTY do not yet have runtime backends. Their interpolation and
spectral-density conversion policies must be tested before they can return the
same wavelength-domain `SpectrumResult` contract.

The v1 prepared backend interpolates only over `teff` and `logg` at one exact
`m_h`, `alpha_m`, `c_m`, and microturbulence plane. Requests outside the spatial
cell are clamped and marked non-OK. Requests on another abundance plane are
marked unmodeled. Neither condition is silently presented as an in-grid model.

TLUSTY therefore retains native `frequency_hz` and `F_nu` in its processed
artifacts, while Sonora retains its released `W/m2/m` convention. Converting
either family into the runtime wavelength-domain contract is future backend work,
not an ingestion shortcut.

For measured local coverage, see [](./atmosphere-capabilities.md). For importable
types and signatures, see [](../40-api/index.md#jaxstro-atmospheres). For the
assertion-bearing evidence map, see [](../60-validation/index.md).
