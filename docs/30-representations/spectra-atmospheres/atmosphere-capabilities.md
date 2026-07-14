---
title: Atmosphere libraries and coverage
description: >-
  Exact local atmosphere products, canonical surface-flux semantics, topology rules,
  and evidence-gated interpolation policies.
---

Use this page when choosing whether a local atmosphere product can support a
particular parameter point and spectral request without crossing an unvalidated
interpolation boundary.

:::{important} Implemented Jaxstro capability
`jaxstro.atmospheres` discovers, validates, selects, and prepares exact local
NewEra, BOSZ, Sonora, and TLUSTY products. Some product policies intentionally fail
closed because acceptance evidence is incomplete.
:::

## Representation contract

| Contract field | Current representation |
| --- | --- |
| Mathematical object | A product-scoped atmosphere parameter grid whose vertices provide source spectra and whose selected local topology becomes a prepared spectral evaluator. |
| Physical convention | Released coordinates and density semantics are product-specific; adapters convert to canonical increasing wavelength in nm and surface flux `F_lambda` in CGS per nm. |
| Runtime owner | `jaxstro.atmospheres` owns catalogs, product identity, topology selection, adapters, artifact reports, and preparation status. |
| Shape and unit policy | Host catalogs may be ragged or sparse; prepared cells and simplices are fixed-shape PyTrees, and canonical spectra are one-dimensional. |
| Transform boundary | Host selection is discrete; only evaluation inside a fixed accepted topology has a local JAX transform contract. |
| Evidence | Real-artifact holdouts, policy manifests, transform tests, and AD-vs-FD checks distinguish adapter availability from accepted runtime policy. |
| Downstream interpretation boundary | Atmosphere model adequacy, stellar radius, distance, extinction, photometry, instruments, and likelihood semantics remain outside Jaxstro. |

## Surface spectra and geometry

Atmosphere products represent a released surface flux or source-specific precursor,
not an observer flux. Under the separate spherical-isotropic geometry assumption,

```{math}
:label: eq-atmosphere-surface-to-observer

F_{\lambda,\mathrm{obs}}
=
\left(\frac{R}{d}\right)^2
F_{\lambda,\mathrm{surface}}.
```

Jaxstro can evaluate the algebraic transform when radius and distance are supplied,
but the atmosphere library does not silently choose them. The downstream model owns
whether [](#eq-atmosphere-surface-to-observer) is appropriate.

## Capability matrix

| Product family | Released semantic | Canonical conversion | Runtime policy |
| --- | --- | --- | --- |
| PHOENIX/NewEra low-resolution v3 | wavelength in nm; `F_lambda` in `W m^-2 nm^-1` | multiply density by `1e3` | positive-log accepted |
| BOSZ 2025 recomputed resampled products | wavelength in angstrom; `F_lambda` per angstrom | angstrom to nm; multiply density by `10` | linear accepted for measured resampled product |
| Sonora Diamondback 2024 | wavelength in micron; wavelength density in `W m^-2 m^-1` | micron to nm; multiply density by `1e-6` | `POLICY_NOT_VALIDATED` |
| TLUSTY OSTAR2002 | frequency in Hz; Eddington flux `H_nu` | `F_nu = 4 pi H_nu`, then density conversion | linear accepted |
| TLUSTY BSTAR2006, `vturb=2` and `vturb=10` | frequency in Hz; Eddington flux `H_nu` | same explicit TLUSTY conversion | `POLICY_NOT_VALIDATED` |

"Adapter implemented" and "runtime policy accepted" are separate claims. Sonora and
BSTAR artifacts can be opened and converted, but the registry returns
`POLICY_NOT_VALIDATED` because measured interpolation metrics trade off under the
declared selection rule.

## Exact product identity

Atmosphere interpolation never crosses model family, resolution product, composition
plane, cloud prescription, or C/N variant. Product IDs encode those choices. TLUSTY has 27 composition-scoped products rather than three broad dataset aliases, which
prevents duplicate `(teff, logg)` nodes from distinct abundance planes from entering
one topology.

## Topology and coverage

An adapter chooses a complete rectilinear cell first. A sparse region is usable only
when its simplex is explicitly approved; there is no nearest-neighbor or arbitrary triangulation fallback. Every selected vertex must cover the requested spectral plan.
Expected gaps return structured statuses instead of clamped or extrapolated science.

Prepared evaluation is differentiable only inside the selected fixed topology.
Product changes, cell changes, and boundary crossings are discrete host events.

:::{warning} Coverage is not physical validity
A point inside a released grid can still be scientifically inappropriate for a
particular object. Coverage and interpolation evidence support data handling, not the
physical assumptions of the source atmosphere family.
:::

## Dataset-specific rationale

Sonora filenames encode gravity as SI acceleration while `AtmosphereParams.logg`
uses cgs. The catalog stores the released value and the explicit conversion
`log10(g_m_s2 * 100)`.

TLUSTY spectra are ragged and use subgroup-local frequency axes. Some published axes
repeat printed coordinates. At the canonical boundary Jaxstro mean-coalesces samples
at identical published coordinates before enforcing a strictly increasing wavelength
axis, and records that operation in provenance.

The policy manifest is
[`docs/validation/atmosphere-interpolation.json`](../../60-validation/index.md).
See [](./source-artifacts-and-adapters.md) for artifact mechanics and
[](./spectra-data-architecture.md) for the runtime ownership chain.
