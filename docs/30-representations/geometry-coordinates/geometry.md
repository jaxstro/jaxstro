---
title: Vector geometry and rigid transforms
description: >-
  Vector normalization, angular distances, right-handed rotations, quaternions, and
  rigid transforms with explicit composition conventions.
---

Use this page when a computation needs domain-neutral vector geometry or a rigid
three-dimensional transform without selecting an astronomical coordinate frame.

:::{important} Implemented Jaxstro capability
`jaxstro.geometry` provides generic vector, rotation, quaternion, and rigid-transform
helpers implemented with JAX arrays.
:::

## Representation contract

| Contract field | Current representation |
| --- | --- |
| Mathematical object | Euclidean vectors, angular separation, rotation matrices, unit quaternions, and rigid transforms. |
| Physical convention | Rotations are right-handed; quaternions are stored as scalar-first `[w, x, y, z]`; rigid composition is explicitly `outer(inner(point))`. |
| Runtime owner | `jaxstro.geometry` owns the domain-neutral geometric maps. |
| Shape and unit policy | Vectors have a final component axis, rotations are `(3, 3)`, quaternions are `(4,)`, and angles are radians; coordinate units remain caller-owned and must be consistent. |
| Transform boundary | Regular finite vectors and fixed composition paths support `jit`, `vmap`, and local AD; zero-vector normalization, collinear angular endpoints, and singular axes are excluded boundaries. |
| Evidence | Unit tests check norm preservation, matrix-quaternion parity, inverse transforms, and composition; validation compares smooth-domain AD with finite differences. |
| Downstream interpretation boundary | Jaxstro does not assign a sky frame, body frame, handedness conversion, uncertainty model, or domain geometry policy. |

## Vectors and angles

`normalize(vectors, axis=..., return_norm=...)` returns unit vectors and optionally
their original norms. `angular_distance(a, b)` normalizes both inputs and evaluates

```{math}
:label: eq-geometry-angular-distance

\alpha
=
\arccos\left(
\frac{\mathbf{a}\cdot\mathbf{b}}
     {\lVert\mathbf{a}\rVert\lVert\mathbf{b}\rVert}
\right).
```

Inputs may have leading batch dimensions. `axis` identifies the component dimension.
Angles are returned in radians and do not carry a runtime unit object.

Degenerate zero-vector behavior is explicit through the `eps` argument to
`normalize`. With `eps=0`, a zero vector produces an undefined division; with a
positive `eps`, the caller has deliberately chosen a numerical floor. Neither choice
creates a physical direction for a zero-vector input.

## Rotations and quaternions

`rotation_matrix(axis, angle)` uses the right-handed axis-angle convention.
`quaternion_from_axis_angle`, `quaternion_multiply`, `quaternion_conjugate`, and
`quaternion_rotate` use quaternions stored as `[w, x, y, z]`. Quaternion rotation
normalizes the quaternion before applying the Hamilton product.

:::{warning} A convention mismatch can preserve norms and still be wrong
Scalar-last quaternions, left-handed axes, passive rotations, and reversed
multiplication order can all produce plausible arrays. Record the convention at the
boundary instead of inferring it from shape.
:::

## Rigid transforms and order

`rigid_transform(points, rotation, translation)` applies

```{math}
:label: eq-geometry-rigid-transform

\mathbf{p}' = \mathbf{R}\mathbf{p}+\mathbf{t}.
```

`invert_rigid` returns the inverse. `compose_rigid(outer_R, outer_t, inner_R,
inner_t)` returns a transform equivalent to `outer(inner(point))`; reversing the
arguments changes the result because rigid-transform composition is not commutative.

The tests establish the algebra and the smooth executed-map derivatives represented
by [](#eq-geometry-angular-distance) and [](#eq-geometry-rigid-transform). They do not
identify which frame a vector belongs to or whether a downstream rotation convention
matches these choices.
