---
title: Geometry helpers
description: >-
  Vector normalization, angular distances, axis-angle rotations, quaternions,
  and rigid transforms with explicit composition conventions.
---

`jaxstro.geometry` is the small science-general layer that sits next to the
astronomy-specific coordinate transforms. It provides generic vector geometry and
rigid-transform helpers without owning any sky-coordinate convention or domain
physics.

## Vectors and angles

`normalize(vectors, axis=..., return_norm=...)` returns unit vectors and,
optionally, the original norms. `angular_distance(a, b)` normalizes both inputs
and returns the angle in radians.

Degenerate zero-vector behavior is explicit through the `eps` argument to
`normalize`; callers that need a nonzero floor should pass it deliberately.

## Rotations and quaternions

`rotation_matrix(axis, angle)` uses the right-handed axis-angle convention.
Quaternions are stored as `[w, x, y, z]`. The module exposes
`quaternion_from_axis_angle`, `quaternion_multiply`,
`quaternion_conjugate`, and `quaternion_rotate`.

The tests compare quaternion rotation against the corresponding rotation matrix
and check inverse round trips.

## Rigid transforms

`rigid_transform(points, rotation, translation)` applies

```{math}
p' = R p + t
```

to a single point or a leading batch of points. `invert_rigid` returns the
inverse transform. `compose_rigid(outer_R, outer_t, inner_R, inner_t)` returns a
transform equivalent to `outer(inner(point))`.

## Validation

Unit tests cover norm preservation, rotation identities, quaternion/matrix
parity, inverse rigid transforms, explicit composition order, and JAX transforms.
Validation tests compare FD-vs-AD derivatives for smooth rotation and angular
distance paths.
