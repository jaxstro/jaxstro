---
title: Spectral coordinates, densities, and prepared evaluation
description: >-
  Explicit spectral axes, density semantics, provenance, host-side preparation, and
  fixed-topology JAX evaluation.
---

Use this page when an array of spectral values must retain its coordinate, sampling,
density semantic, provenance, and interpolation boundary before it enters a compiled
calculation.

:::{important} Implemented Jaxstro capability
`jaxstro.spectra` owns generic spectral representations, transformations, resampling,
statuses, and prepared stencils. `jaxstro.atmospheres` prepares source-specific local
products for that runtime surface.
:::

## Representation contract

| Contract field | Current representation |
| --- | --- |
| Mathematical object | A one-dimensional `Spectrum` pairs values with a `SpectralAxis`, `SpectralSemantic`, mandatory `SpectrumProvenance`, and structured status. |
| Physical convention | The spectral coordinate, unit, point or bin sampling, and density semantic are explicit; canonical atmosphere preparation uses increasing wavelength in nm and named CGS density units. |
| Runtime owner | `jaxstro.spectra` owns generic types and array operations; `jaxstro.atmospheres` owns host-side source preparation. |
| Shape and unit policy | Axis values and spectrum values are one-dimensional and shape-matched; prepared topology arrays have fixed shapes, while units and semantics are static metadata. |
| Transform boundary | Prepared array evaluation supports `jit`, `vmap`, and local pathwise AD inside a fixed topology; lookup, artifact I/O, topology selection, and policy choice remain host-side. |
| Evidence | Spectra unit and validation tests cover semantic transforms, fail-closed resampling, prepared stencils, AD-vs-FD, artifacts, and policy manifests. |
| Downstream interpretation boundary | Filters, extinction, instruments, distance and radius policy, detector counts, images, likelihoods, and model validity remain downstream. |

## Why a spectrum is more than an array

A wavelength density and a frequency density represent different derivatives of the
same flux. For point-sampled values they satisfy

```{math}
:label: eq-spectra-density-jacobian

F_{\lambda}
=
F_{\nu}\left|\frac{d\nu}{d\lambda}\right|
=
F_{\nu}\frac{c}{\lambda^2}.
```

Changing only the coordinate values would violate the density relation in
[](#eq-spectra-density-jacobian). `to_flux_lambda` and `to_flux_nu` transform the
axis, values, semantic, and provenance together. These transforms currently require
point samples and canonical nm or Hz axes.

Point samples, bin averages, and bin integrals are also distinct. A point sample is a
value at one coordinate. A binned value represents an interval and therefore requires
edges. `SpectralAxis.sampling` makes that distinction explicit.

## Host preparation and runtime evaluation

`jaxstro.atmospheres` owns the host-side request:

```{code-block} text
:caption: Interface notation, not Python

AtmosphereQuery -> PreparationResult[PreparedAtmosphere]
```

`jaxstro.spectra` owns the fixed-shape array surface. Fluxax keeps ownership of
extinction, filters, instruments, counts, magnitudes, PSFs, images, and
likelihood-facing observables.

:::{figure} ./figures/spectra-runtime-boundary.webp
:name: fig-spectra-runtime-boundary
:alt: Three-stage spectra workflow from host-side catalog and artifact preparation through a JAX-ready local grid to downstream observables

The filesystem boundary is one-way. Host code selects an exact product and complete
local topology, converts vertices to one requested spectral axis, and returns a
fixed-shape PyTree. Evaluation then uses arrays only.
:::

```{list-table} Spectra execution and ownership boundaries
:header-rows: 1
:label: tbl-spectra-execution-boundaries

* - Operation
  - Execution side
  - Owner and contract
* - Product lookup and topology selection
  - Host-side Python
  - `AtmosphereLibrary` and exact-product adapters validate artifacts, choose a
    complete cell or explicitly approved simplex, and fail closed on gaps.
* - Spectral conversion and preparation
  - Host-side Python with optional data dependencies
  - Each source vertex is converted to increasing wavelength in `nm` and surface
    `F_lambda` in `erg s^-1 cm^-2 nm^-1`, then resampled onto the request plan.
* - Prepared parameter interpolation
  - JAX-side array computation
  - `PreparedRectilinearStencil` or `PreparedSimplexStencil` supports `jit`,
    `vmap`, and local pathwise derivatives inside its fixed topology.
* - Observable rendering
  - Downstream package
  - Fluxax and domain packages own distance scaling, attenuation, passbands,
    detector response, images, and likelihood semantics.
```

## Portable JAX-side example

This complete example creates a prepared cell in memory. It requires no local
atmosphere artifacts.

```python
import jax
import jax.numpy as jnp

from jaxstro.spectra import (
    FluxInterpolation,
    PreparedRectilinearStencil,
    SpectralAxis,
    SpectralCoordinate,
    SpectralSemantic,
    Spectrum,
    SpectrumProvenance,
    SpectrumStatusCode,
)

axis = SpectralAxis.points(
    jnp.array([500.0, 600.0, 700.0]),
    coordinate=SpectralCoordinate.WAVELENGTH,
    unit="nm",
)
template = Spectrum(
    axis=axis,
    values=jnp.array([1.0, 2.0, 3.0]),
    semantic=SpectralSemantic.SURFACE_FLUX_LAMBDA,
    provenance=SpectrumProvenance(
        source_id="portable-fixture",
        product_id="portable-fixture",
        native_coordinate="wavelength_nm",
        native_density="F_lambda",
        native_unit="erg s^-1 cm^-2 nm^-1",
        canonical_conversion="identity",
        citations=("fixture:documentation",),
    ),
)
prepared = PreparedRectilinearStencil(
    parameter_axes=(jnp.array([5000.0, 6000.0]), jnp.array([4.0, 5.0])),
    vertex_values=jnp.array(
        [
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
            [[3.0, 4.0, 5.0], [4.0, 5.0, 6.0]],
        ]
    ),
    template=template,
    interpolation=FluxInterpolation.LINEAR,
)

midpoint = prepared.evaluate(jnp.array([5500.0, 4.5]))
outside = prepared.evaluate(jnp.array([4500.0, 4.5]))

@jax.jit
def first_flux(teff):
    return prepared.evaluate(jnp.array([teff, 4.0])).spectrum.values[0]

local_slope = jax.grad(first_flux)(5500.0)

assert jnp.allclose(midpoint.spectrum.values, jnp.array([2.5, 3.5, 4.5]))
assert int(midpoint.status.code) == SpectrumStatusCode.OK
assert int(outside.status.code) == SpectrumStatusCode.OUTSIDE_CONVEX_HULL
assert jnp.all(jnp.isnan(outside.spectrum.values))
assert jnp.isclose(local_slope, 0.002)
```

The derivative is local to one fixed cell. Catalog ranking, artifact I/O, topology
changes, and interpolation-policy selection are discrete host operations and are not
advertised as differentiable.

## Query a local atmosphere library

**Execution contract - local processed artifacts required.** This recipe also
requires the `data` optional dependencies.

```python
import jax.numpy as jnp

from jaxstro.atmospheres import AtmosphereLibrary, AtmosphereParams, AtmosphereQuery
from jaxstro.spectra import SpectralAxis, SpectralCoordinate, SpectralPlan

library = AtmosphereLibrary.from_local("data")
query = AtmosphereQuery(
    params=AtmosphereParams(teff=5772.0, logg=4.44),
    product_id="newera-v3-lowres",
    family="newera",
    spectral_plan=SpectralPlan(
        SpectralAxis.points(
            jnp.linspace(500.0, 2500.0, 256),
            coordinate=SpectralCoordinate.WAVELENGTH,
            unit="nm",
        )
    ),
    requested_parameter_names=("teff", "logg"),
)
preparation = library.prepare(query)
if preparation.prepared is not None:
    result = preparation.prepared.evaluate(query.params)
```

Expected scientific gaps return a `SpectrumStatusCode`. Corrupt artifacts and broken
invariants raise exceptions instead of masquerading as coverage gaps.

## Current evidence boundary

Real-artifact holdouts currently accept positive-log parameter interpolation for
NewEra and linear interpolation for the BOSZ resampled product and OSTAR2002. Sonora Diamondback and both BSTAR modes remain `POLICY_NOT_VALIDATED`: their measured linear
and positive-log errors trade off under the declared selection rule. The adapters and
artifacts exist, but `AtmosphereLibrary.prepare(...)` refuses to present those paths
as supported science.

See [](./atmosphere-capabilities.md),
[](../../40-workflows/data-pipelines/query-atmosphere-spectra.md), and
[](../../60-validation/index.md) for the product, workflow, and evidence boundaries.
