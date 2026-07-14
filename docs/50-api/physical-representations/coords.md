---
title: Coordinate transformations
---

# Coordinate transformations

## Owner import path

`jaxstro.coords`

## Purpose

Sky tangent, Galactic/equatorial, Cartesian/spherical, parallax, proper-motion,
and observing-geometry transformations with explicit frames.

## Public records and callables

`sky_tangent`, `cluster_to_galactic_cartesian`, `galactic_to_equatorial`,
`equatorial_to_galactic`, `cartesian_to_spherical`,
`spherical_to_cartesian`, `compute_parallax`, `compute_proper_motions`, and
`zenith_parallactic`.

## Shape and dtype expectations

Cartesian arrays have trailing length-three axes. Public angles, positions,
velocities, parallax, and proper-motion units follow each signature's explicit
contract.

## JAX transforms and AD classification

Array transforms compose with JIT, VMAP, and smooth-pathwise AD on regular
domains. Poles, origins, coincident geometries, and angular branches are named
singular boundaries.

## Failure behavior

Physically undefined geometry remains non-finite or raises according to the
public contract; no surrogate finite gradient is invented.

## Contract and evidence links

See [](../../30-representations/geometry-coordinates/coordinate-transformations.md),
the [source-backed cards](../research-infrastructure/source-provenance/transforms.md),
and [](../../60-validation/validation.md).

## Canonical import example

```python
from jaxstro.coords import galactic_to_equatorial
```
