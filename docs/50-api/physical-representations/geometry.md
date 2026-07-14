---
title: Geometry
---

# Geometry

## Owner import path

`jaxstro.geometry`

## Purpose

Vector normalization, angular distance, rotations, quaternions, and rigid
transforms.

## Public records and callables

`normalize`, `angular_distance`, `rotation_matrix`,
`quaternion_from_axis_angle`, `quaternion_conjugate`, `quaternion_multiply`,
`quaternion_rotate`, `rigid_transform`, `invert_rigid`, and `compose_rigid`.

## Shape and dtype expectations

Vectors have trailing length-three axes; quaternions have trailing length-four
axes. Rotation matrices are trailing three-by-three floating arrays.

## JAX transforms and AD classification

Kernels compose with JIT, VMAP, and smooth-pathwise AD away from zero vectors,
coincident directions, and branch boundaries.

## Failure behavior

Singular normalization and undefined angular geometry remain explicit. Rigid
composition order is not inferred.

## Contract and evidence links

See [](../../30-representations/geometry-coordinates/geometry.md) and
[](../../60-validation/index.md).

## Canonical import example

```python
from jaxstro.geometry import quaternion_rotate
```
