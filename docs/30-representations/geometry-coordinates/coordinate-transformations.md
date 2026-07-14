---
title: Coordinate transformations
description: >-
  Explicit sky, Galactic, spherical, tangent-frame, parallax, and proper-motion
  coordinate conventions.
---

Use this page when an array must be interpreted in a named spatial or sky coordinate
frame before a model compares positions, angles, parallaxes, or proper motions.

:::{important} Implemented Jaxstro capability
`jaxstro.coords` provides importable JAX coordinate transformations with documented
frame orientation, units, output shapes, and singular domains.
:::

## Representation contract

| Contract field | Current representation |
| --- | --- |
| Mathematical object | Maps among local Cartesian vectors, spherical coordinates, ICRS directions, IAU Galactic directions, and local sky-tangent observables. |
| Physical convention | ICRS right ascension and declination, the IAU Galactic rotation, right-handed local tangent frames, RA-star proper motion, and named angular units. |
| Runtime owner | `jaxstro.coords` owns coordinate evaluation; `jaxstro.astrometry` owns reusable conversion constants. |
| Shape and unit policy | Cartesian batches normally use shape `(N, 3)`; sky angles are degrees or radians as named, positions are parsecs, velocities are km/s, parallax is mas, and proper motion is mas/yr. |
| Transform boundary | Regular fixed-frame paths support `jit`, `vmap`, and local derivatives; poles, origins, horizons, coincident observer-star positions, and wrapped angles are explicit singular or nonsmooth boundaries. |
| Evidence | Coordinate unit tests check round trips and reference landmarks; validation tests compare smooth-domain AD against finite differences and test singular behavior. |
| Downstream interpretation boundary | Jaxstro does not choose a survey frame, epoch, reference-star model, perspective model, or likelihood convention for a domain package. |

## Frames are part of the data

A Cartesian vector does not identify its origin, orientation, scale, or frame. For
the spherical convention used by `cartesian_to_spherical`, a point is represented by

```{math}
:label: eq-coordinates-spherical-map

x=r\sin\theta\cos\phi,
\qquad
y=r\sin\theta\sin\phi,
\qquad
z=r\cos\theta,
```

where `theta` is the polar angle from positive z and `phi` is the azimuth from
positive x toward positive y. The inverse returns radius in parsecs and angles in
radians. At the origin both angles are undefined; on the z axis `phi` is undefined.

`galactic_to_equatorial` and `equatorial_to_galactic` use the fixed IAU Galactic
rotation expressed in ICRS and return degrees. Longitude and right ascension are
wrapped to `[0, 360)`. Coordinate wrapping preserves a direction but introduces a
numeric discontinuity at the branch cut.

## Local sky geometry

`sky_tangent` embeds local `(x, y, z)` positions around a system center into an ICRS
tangent frame defined by center right ascension, center declination, distance, and an
optional roll. `compute_proper_motions` uses the same frame to project velocities onto
the local RA-star and declination bases. `zenith_parallactic` represents local
observing geometry through zenith distance and parallactic angle.

:::{warning} Coordinate singularities are physical representation boundaries
Right ascension is undefined at a celestial pole, azimuth is undefined on an axis,
and parallactic angle is undefined at zenith or nadir. A finite convention value at
one of these locations must not be interpreted as a unique tangent direction or a
scientifically meaningful gradient.
:::

## Shapes and transformations

Most Cartesian functions accept a leading batch of three-vectors or an explicit
`(N, 3)` array. Scalar frame parameters broadcast according to their function
contract. The public functions use JAX arrays and fixed formulas, so batching and
compilation are natural on regular domains. Local derivatives describe the executed
coordinate map; they do not make frame selection, angle wrapping, or singular-basis
choices differentiable.

The round-trip and reference tests support the mathematical mapping claim in
[](#eq-coordinates-spherical-map). They do not establish that a downstream dataset
uses the same epoch, frame realization, or observational convention.
