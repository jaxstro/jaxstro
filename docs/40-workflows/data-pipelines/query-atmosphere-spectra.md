---
title: Query atmosphere spectra
description: >-
  Build an exact, fixed-axis atmosphere request and handle supported and
  evidence-gated outcomes without hidden extrapolation.
---

Use this page when a downstream package needs a model-atmosphere **surface
spectrum**. The result is not yet an observer flux, magnitude, count rate, or
image.

## 1. Choose an exact product

Start from [](../../30-representations/spectra-atmospheres/atmosphere-capabilities.md).
Product identity
includes scientifically meaningful choices such as BOSZ resolution, Sonora
cloud/metallicity/C/O, or TLUSTY composition and C/N state.

```python
product_id = "newera-v3-lowres"
```

Do not interpolate product IDs or use them as approximate labels.

## 2. Declare the output axis

Atmosphere adapters return increasing wavelength in `nm` and surface `F_lambda`
in `erg s^-1 cm^-2 nm^-1`. Point samples are explicit here; a binned request
would need bin edges and bin semantics.

```python
import jax.numpy as jnp

from jaxstro.spectra import SpectralAxis, SpectralCoordinate, SpectralPlan

plan = SpectralPlan(
    SpectralAxis.points(
        jnp.linspace(500.0, 2500.0, 256),
        coordinate=SpectralCoordinate.WAVELENGTH,
        unit="nm",
    )
)
```

Requests outside the common coverage of every selected topology vertex fail;
they are not padded with zeros or extrapolated.

## 3. Prepare on the host

**Execution contract - local processed artifacts required.** Install the `data`
extra and point the library at the local data root.

```python
from jaxstro.atmospheres import AtmosphereLibrary, AtmosphereParams, AtmosphereQuery

library = AtmosphereLibrary.from_local("data")
query = AtmosphereQuery(
    params=AtmosphereParams(teff=5772.0, logg=4.44),
    product_id=product_id,
    family="newera",
    spectral_plan=plan,
    requested_parameter_names=("teff", "logg"),
)
preparation = library.prepare(query)
```

Preparation performs filesystem I/O, artifact validation, exact-product lookup,
topology selection, unit conversion, and spectral resampling. Keep it outside a
jitted model.

## 4. Handle the structured outcome

```python
from jaxstro.spectra import SpectrumStatusCode

if preparation.status is SpectrumStatusCode.OK:
    prepared = preparation.prepared
    assert prepared is not None
    result = prepared.evaluate(query.params)
elif preparation.status is SpectrumStatusCode.POLICY_NOT_VALIDATED:
    raise RuntimeError("the product exists, but its interpolation policy is unratified")
else:
    raise RuntimeError(f"atmosphere request unavailable: {preparation.status.name}")
```

Common expected outcomes include `NO_DATASET`, `NO_COMPLETE_CELL`,
`OUTSIDE_CONVEX_HULL`, `UNSUPPORTED_SPECTRAL_WINDOW`, and
`POLICY_NOT_VALIDATED`. Broken artifacts and invariant violations raise directly.

## 5. Evaluate inside the prepared topology

```python
import jax

@jax.jit
def surface_flux(teff, logg):
    params = AtmosphereParams(teff=teff, logg=logg)
    return prepared.evaluate(params).spectrum.values
```

The prepared object contains arrays, not an active file reader. `jit`, `vmap`,
and local AD are supported inside its fixed region. If inference crosses a cell
or product boundary, prepare the new region on the host.

## 6. Hand the surface spectrum downstream

Inspect `result.spectrum.semantic` and `result.spectrum.provenance` before any
rendering. Radius/distance scaling, extinction, filters, detector response,
photometry, PSFs, and likelihood observables belong in Fluxax or another domain
package. Keeping that handoff explicit prevents a surface `F_lambda` from being
mistaken for an observer flux density.
