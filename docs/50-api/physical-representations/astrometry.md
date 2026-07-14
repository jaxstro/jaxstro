---
title: Astrometric constants
---

# Astrometric constants

## Owner import path

`jaxstro.astrometry`

## Purpose

Named conversion constants used by astrometric transformations.

## Public records and callables

`KM_PER_PC`, `MAS_PER_RAD`, `ARCSEC_PER_RAD`, `DEG_PER_RAD`, `YR_PER_MYR`, and
`K_PROPER_MOTION`.

## Shape and dtype expectations

These are frozen scalar conversion values; their names identify the unit
relationship.

## JAX transforms and AD classification

Constants enter array expressions as static values and have no parameter AD
claim.

## Failure behavior

There is no runtime failure state. Stored precision and convention are part of
the source-provenance contract.

## Contract and evidence links

See [](../../30-representations/geometry-coordinates/astrometry.md) and the
[source-backed cards](../research-infrastructure/source-provenance/transforms.md).

## Canonical import example

```python
from jaxstro.astrometry import K_PROPER_MOTION
```
