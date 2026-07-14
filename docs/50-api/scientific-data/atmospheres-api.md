---
title: Atmosphere libraries
---

# Atmosphere libraries

## Owner import path

`jaxstro.atmospheres`

## Purpose

Catalog discovery, exact-product adapters, acquisition planning, topology
preparation, and evidence-gated atmosphere-spectrum evaluation.

## Public records and callables

The owner exports `AtmosphereLibrary`, `AtmosphereAdapter`,
`AtmosphereAdapterRegistry`, query, selection, preparation, topology, coverage,
and artifact records; NewEra, BOSZ, Sonora, and TLUSTY backends and metadata;
and the discovery, parsing, indexing, selection, acquisition, overlap, and data
directory helpers listed by `jaxstro.atmospheres.__all__`.

## Shape and dtype expectations

Catalog and artifact metadata are host-side records. Prepared spectra and
stencils use fixed floating array shapes and explicit coordinate/flux semantics.

## JAX transforms and AD classification

Only prepared evaluation paths whose interpolation policy has passed its
evidence gate claim JAX-side transformation. I/O, product selection, and
topology changes are host-side.

## Failure behavior

Selection and preparation return structured outcomes for missing data,
unsupported topology, coverage, and policy gaps. Importability is not evidence
of scientific validity.

## Contract and evidence links

See [](../../30-representations/spectra-atmospheres/atmosphere-capabilities.md),
the [source-backed cards](../research-infrastructure/source-provenance/atmospheres.md),
and [](../research-infrastructure/contracts.md).

## Canonical import example

```python
from jaxstro.atmospheres import AtmosphereLibrary
```
