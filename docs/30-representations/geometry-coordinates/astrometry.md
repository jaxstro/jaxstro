---
title: Astrometric conversion conventions
description: >-
  Angular, distance, time, and proper-motion conversion constants used by current
  coordinate calculations.
---

Use this page when converting between radians, arcseconds, parsecs, years, transverse
velocity, and proper motion, or when auditing the convention behind a stored factor.

:::{important} Implemented Jaxstro capability
`jaxstro.astrometry` provides importable scalar astrometric conversion constants.
Coordinate-dependent parallax and proper-motion maps remain in `jaxstro.coords`.
:::

## Representation contract

| Contract field | Current representation |
| --- | --- |
| Mathematical object | Scalar conversion constants connecting angular, distance, time, and transverse-velocity representations. |
| Physical convention | IAU exact astronomical-unit and parsec definitions, radians as the angular base, Julian-year-derived velocity conversion, and the RA-star proper-motion convention. |
| Runtime owner | `jaxstro.astrometry` owns `KM_PER_PC`, `MAS_PER_RAD`, `ARCSEC_PER_RAD`, `DEG_PER_RAD`, `YR_PER_MYR`, and `K_PROPER_MOTION`. |
| Shape and unit policy | Values are scalar Python floats with units named in symbols and docstrings; array broadcasting is caller-owned. |
| Transform boundary | Constants can participate in transformed JAX expressions but are static coefficients; frame-dependent astrometric maps are owned by `jaxstro.coords`. |
| Evidence | Unit tests check stored conversion values and coordinate tests check parallax and proper-motion use on regular geometries. |
| Downstream interpretation boundary | These factors do not define epochs, catalogs, covariance models, solar-system corrections, or survey calibration. |

## Proper-motion scale

The conventional relation used by Jaxstro is

```{math}
:label: eq-astrometry-proper-motion

v_{\perp}[\mathrm{km\,s^{-1}}]
=
K_{\mu}\,
\mu[\mathrm{mas\,yr^{-1}}]
d[\mathrm{kpc}],
\qquad
K_{\mu}=4.74047.
```

`K_PROPER_MOTION` retains the long-standing rounded compatibility literal. The exact
conventional value derived from one astronomical unit per Julian year has more
digits. Code requiring a different precision or epoch convention must make that
choice explicit.

For angular conversions, `MAS_PER_RAD`, `ARCSEC_PER_RAD`, and `DEG_PER_RAD` all
represent the same angle scale in different units. `KM_PER_PC` is derived from the
Jaxstro parsec and kilometer definitions, while `YR_PER_MYR` is exactly one million.

## From constants to coordinate maps

`compute_parallax` and `compute_proper_motions` live in `jaxstro.coords` because their
outputs depend on spatial and tangent-frame geometry, not only scalar conversion.
The conversion in [](#eq-astrometry-proper-motion) is meaningful only after the
transverse basis and observer-star distance are defined.

:::{warning} Units do not remove frame singularities
At a celestial pole the RA basis is undefined, and at zero observer-star separation
parallax and proper motion have no finite physical meaning. Applying a conversion
constant cannot repair an undefined coordinate representation.
:::

The current evidence verifies constants and regular-domain coordinate calculations.
It does not provide a complete astrometric model with epoch propagation, aberration,
light-time, covariance, or survey systematics.
