---
title: Spectra data architecture
description: >-
  How jaxstro turns local atmosphere products into explicit, differentiable
  spectral arrays without taking ownership of observables.
---

A spectrum is not just an array of brightness values. Its coordinate, sampling,
physical meaning, provenance, and valid interpolation region all affect what a
downstream model is allowed to do. Jaxstro makes those pieces explicit before a
spectrum enters JAX.

`jaxstro.atmospheres` owns the host-side request:

```{code-block} text
:caption: Interface notation, not Python

AtmosphereQuery -> PreparationResult[PreparedAtmosphere]
```

`jaxstro.spectra` owns the generic array types, transformations, resampling, and
prepared interpolation stencils. Fluxax keeps ownership of extinction, filters,
instruments, counts, magnitudes, PSFs, images, and likelihood-facing observables.

:::{figure} ./figures/spectra-runtime-boundary.webp
:name: fig-spectra-runtime-boundary
:alt: Three-stage spectra workflow from host-side catalog and artifact preparation through a JAX-ready local grid to downstream observables

The filesystem boundary is one-way. Host code selects an exact product and a
complete local topology, converts its vertices to one requested spectral axis,
and returns a fixed-shape PyTree. Evaluation then uses arrays only.
:::

## Three execution layers

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

## Read the physical semantic before the numbers

Atmosphere products return **surface flux density**. An observer flux density is
a different semantic and generally requires geometry such as a radius and
distance. Jaxstro does not silently apply that geometry.

The density coordinate matters too. `F_lambda` is per wavelength interval;
`F_nu` is per frequency interval. They satisfy

```{math}
F_\lambda = F_\nu\left|\frac{d\nu}{d\lambda}\right|
          = F_\nu\frac{c}{\lambda^2}.
```

TLUSTY publishes Eddington flux `H_nu`, so its source-specific path first uses
`F_nu = 4 pi H_nu`. That factor is not universal: it belongs to the definition
of the released TLUSTY column. NewEra and Sonora publish different density
semantics and therefore use different conversions. The provenance attached to
every `Spectrum` records the native coordinate, density, unit, conversion, and
source.

Point samples and bin values are also different contracts. A point sample is a
value at one coordinate. A bin average or bin integral represents an interval
and requires edges. `SpectralAxis.sampling` records which meaning applies;
resampling never silently swaps one for another.

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

The derivative is local to one fixed cell. Catalog ranking, artifact I/O,
topology changes, and interpolation-policy selection are discrete host-side
operations and are not advertised as differentiable.

## Query a local atmosphere library

**Execution contract — local processed artifacts required.** This recipe also
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

Expected scientific gaps return a `SpectrumStatusCode`, including no exact
dataset, no complete cell, outside-convex-hull requests, unsupported spectral
windows, and policies that have not passed validation. Corrupt artifacts and
broken invariants raise exceptions instead of masquerading as coverage gaps.

## Current evidence boundary

Real-artifact holdouts currently accept positive-log parameter interpolation for
NewEra and linear interpolation for the BOSZ resampled product and OSTAR2002.
Sonora Diamondback and both BSTAR modes remain `POLICY_NOT_VALIDATED`: their
linear and positive-log errors trade off under the declared selection rule.
Those adapters and artifacts exist, but `AtmosphereLibrary.prepare(...)` refuses
to present the unratified paths as supported science.

See [](./atmosphere-capabilities.md) for the product matrix,
[](../50-howto/query-atmosphere-spectra.md) for the step-by-step recipe, and
[](../60-validation/index.md) for measured evidence.
