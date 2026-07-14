---
title: Source artifacts and adapters
description: >-
  Host-side source discovery, exact product identity, canonical conversion, topology
  selection, provenance, and fail-closed preparation.
---

Use this page when a local atmosphere archive must become a reproducible,
source-specific, fixed-shape spectral input without hiding file or conversion choices
inside JAX.

:::{important} Implemented Jaxstro capability
`jaxstro.atmospheres` provides catalog discovery, artifact reports, product adapters,
topology selection, overlap diagnostics, and prepared-atmosphere results for current
local libraries.
:::

## Representation contract

| Contract field | Current representation |
| --- | --- |
| Mathematical object | A host-side mapping from released files and catalog rows to an exact product descriptor, validated local topology, canonical spectrum vertices, and a prepared evaluator. |
| Physical convention | Every adapter records native coordinate, density, unit, conversion, citations, and exact product identity before producing canonical surface spectra. |
| Runtime owner | `jaxstro.atmospheres` owns adapters, backends, catalog selection, artifact validity, topology, and preparation results. |
| Shape and unit policy | Released files may be ragged; adapters validate them on the host and produce one-dimensional canonical spectra stacked into a fixed cell or simplex. |
| Transform boundary | Filesystem access, parsing, ranking, and topology are not traced; only the prepared array object crosses into JAX evaluation. |
| Evidence | Parser, catalog, artifact, adapter, overlap, real-file, and preparation tests check exact failures and provenance; policy acceptance uses separate validation evidence. |
| Downstream interpretation boundary | Jaxstro does not acquire every archive automatically, choose a physical atmosphere family, or own observables built from the prepared surface spectrum. |

## One-way data boundary

The adapter boundary can be summarized as

```{math}
:label: eq-atmosphere-adapter-map

\mathcal{A}:
(\mathrm{artifact},\mathrm{product},\boldsymbol{\theta},\mathrm{plan})
\longmapsto
(\mathrm{prepared\ arrays},\mathrm{provenance},\mathrm{status}).
```

In [](#eq-atmosphere-adapter-map), the left side is host-owned and may involve paths,
metadata, optional data dependencies, and discrete decisions. The right side contains
fixed-shape arrays plus static metadata suitable for evaluation without filesystem
access.

## Exact product routing

`AtmosphereAdapterRegistry` routes an exact `product_id` to a `NewEraBackend`,
`BoszBackend`, `SonoraBackend`, or `TlustyBackend`. `ProductDescriptor` records the
parameter plane, topology, and evidence-selected interpolation policy.
`ArtifactReport` records whether the catalog and spectral store are present and match
the expected identity.

Broad family names are useful for discovery but insufficient for interpolation.
Resolution, composition, clouds, microturbulence, and abundance variants remain part
of exact product identity.

## Native-to-canonical conversion

Each adapter validates the released axes and columns before conversion. It records
native semantics in `SpectrumProvenance`, performs the source-specific density and
coordinate transform, orders the canonical axis strictly increasingly, and resamples
to the request plan. Source-specific operations such as TLUSTY's `F_nu = 4 pi H_nu`
or duplicate-coordinate coalescing remain visible in provenance.

:::{warning} An adapter is not an interpolation approval
Artifact validity and canonical conversion can pass even when parameter interpolation
has not met its declared error policy. In that case preparation returns
`POLICY_NOT_VALIDATED` rather than exposing the adapter as supported science.
:::

## Failure classes

Expected scientific gaps use structured statuses: no exact dataset, missing complete
topology, outside convex hull, unsupported spectral window, or unvalidated policy.
Corrupt artifacts, malformed axes, and invariant violations raise exceptions because
they are broken inputs, not normal coverage gaps.

The evidence supports deterministic source handling and fixed preparation rules. It
does not prove that a released atmosphere family is appropriate for a particular
object or that a downstream observable model is correct.
