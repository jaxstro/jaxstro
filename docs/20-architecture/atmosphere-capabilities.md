---
title: Atmosphere capabilities
description: >-
  Exact local atmosphere products, canonical spectral semantics, and the
  evidence boundary between implemented and supported interpolation paths.
---

`jaxstro.atmospheres` turns an exact local model-atmosphere product into a
provenance-carrying surface spectrum. It does not compute extinction,
photometry, detector counts, images, or likelihoods; Fluxax and other domain
packages own those observables.

## Mental model

```text
source archive -> catalog + spectral store -> exact product adapter
               -> prepared fixed topology -> canonical SpectrumResult
```

Source archives remain preserved. Host-side adapters own artifact validation,
product identity, topology selection, source-unit conversion, and resampling.
The prepared cell or simplex is a fixed-shape PyTree for JAX evaluation.

## Capability matrix

```{list-table} Local atmosphere libraries
:header-rows: 1
:label: tbl-atmosphere-capabilities

* - Product family
  - Local rows
  - Released spectral semantic
  - Canonical conversion
  - Runtime policy
* - PHOENIX/NewEra low-resolution v3
  - 38,352
  - wavelength in nm; `F_lambda` in `W m^-2 nm^-1`
  - multiply by `1e3`
  - positive-log accepted
* - BOSZ 2025 recomputed, resampled products
  - 3,303 in the local bridge artifact
  - wavelength in angstrom; `F_lambda` per angstrom
  - angstrom to nm; multiply density by `10`
  - linear accepted for the measured resampled product
* - Sonora Diamondback 2024
  - 1,440 valid spectra; 4 resource-fork entries skipped
  - wavelength in micron; wavelength-density flux in `W m^-2 m^-1`
  - micron to nm; multiply density by `1e-6`
  - `POLICY_NOT_VALIDATED`
* - TLUSTY OSTAR2002
  - 690 across 10 exact composition products
  - frequency in Hz; Eddington flux `H_nu`
  - `F_nu = 4 pi H_nu`, then `F_nu` to `F_lambda`
  - linear accepted
* - TLUSTY BSTAR2006, `vturb=2`
  - 981 across 6 exact composition products
  - frequency in Hz; Eddington flux `H_nu`
  - same explicit TLUSTY conversion
  - `POLICY_NOT_VALIDATED`
* - TLUSTY BSTAR2006, `vturb=10`
  - 551 across 11 standard/CN composition products
  - frequency in Hz; Eddington flux `H_nu`
  - same explicit TLUSTY conversion
  - `POLICY_NOT_VALIDATED`
```

“Adapter implemented” and “runtime policy accepted” are intentionally separate.
Sonora and BSTAR artifacts can be opened and converted, but the registry returns
`POLICY_NOT_VALIDATED` because their measured interpolation metrics trade off.

## Exact product identity

Atmosphere interpolation never crosses a model family, resolution product,
composition plane, cloud prescription, or C/N variant. Product IDs encode those
choices. Examples include:

```text
newera-v3-lowres
bosz-2025-recomputed:ap:r10000:resam
sonora-diamondback-2024:f1:m-0.5:co1
tlusty-ostar2002:z1
tlusty-bstar2006:vturb2:z1
tlusty-bstar2006:vturb10:z1:standard
```

TLUSTY has 27 composition-scoped products rather than three dataset aliases.
This prevents duplicate `(Teff, logg)` nodes from different abundances from
entering one topology.

## Topology and coverage rules

An adapter chooses a complete rectilinear cell first. A sparse region is usable
only when its simplex is explicitly approved; there is no nearest-neighbor or
arbitrary triangulation fallback. Each selected source spectrum must cover the
requested spectral plan. Expected gaps return structured status codes rather
than clamped science.

Prepared evaluation is differentiable only inside that fixed topology. Product
selection, cell changes, and boundary crossings remain discrete host events.

## Dataset-specific details

Sonora filenames encode gravity in SI acceleration, while `AtmosphereParams.logg`
uses cgs. The catalog stores both the released value and
`log10(g_m_s2 * 100)`.

TLUSTY spectra are ragged: the processed store uses deterministic `gridNNN`
subgroups with subgroup-local frequency axes. Some published axes contain
duplicate printed frequencies. At the canonical boundary Jaxstro mean-coalesces
samples at identical published coordinates before enforcing a strictly
increasing wavelength axis; that operation is recorded in spectrum provenance.

## Evidence

The policy manifest is
[`docs/validation/atmosphere-interpolation.json`](../60-validation/index.md).
Real-artifact acceptance, fixed-shape JAX transforms, and AD-vs-FD checks live in
`tests/validation/test_atmospheres_spectra.py`. See
[](../50-howto/query-atmosphere-spectra.md) for a request recipe and
[](./spectra-data-architecture.md) for ownership and semantics.
