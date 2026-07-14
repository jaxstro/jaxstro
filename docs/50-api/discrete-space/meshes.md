---
title: Structured meshes
---

# Structured meshes

## Owner import path

`jaxstro.numerics.meshes`

## Purpose

Structured one-dimensional mesh geometry, neighbor records, finite-volume
stencils, and conservative remapping.

## Public records and callables

`Mesh1D`, `FaceGeometry1D`, `CellNeighbors1D`, `structured_edges_1d`,
`face_geometry_1d`, `cell_neighbors_1d`, `divergence_1d`,
`cell_to_face_average`, and `conservative_remap_1d`.

## Shape and dtype expectations

Edges and cell values are one-dimensional floating arrays. Neighbor records use
fixed integer index arrays and masks.

## JAX transforms and AD classification

Fixed-topology array stencils compose with JIT and smooth-pathwise AD in values.
Topology and index construction are discrete preprocessing.

## Failure behavior

Invalid edges or incompatible cell/face shapes raise. Conservative remapping
applies only on overlapping domains.

## Contract and evidence links

See [](../../20-methods/discrete-space/meshes.md) and
[](../../60-validation/index.md).

## Canonical import example

```python
from jaxstro.numerics.meshes import conservative_remap_1d
```
