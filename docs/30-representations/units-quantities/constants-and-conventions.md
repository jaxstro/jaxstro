---
title: Constants and physical conventions
description: >-
  Source-backed CGS constants and the conventions needed to interpret nominal and
  derived astrophysical scales.
---

Use this page when a calculation depends on a physical constant, an astronomical
conversion scale, or the distinction between a nominal convention and a measured
property.

:::{important} Implemented Jaxstro capability
`jaxstro.constants` provides source-backed module-level constants. Values are frozen
at package build time; the module does not query external authorities at runtime.
:::

## Representation contract

| Contract field | Current representation |
| --- | --- |
| Mathematical object | Scalar physical constants and named conversion scales used as coefficients in scientific relations. |
| Physical convention | CGS unless a symbol explicitly names another unit; CODATA 2018, revised-SI exact values, IAU nominal conversions, and cited photometric conventions are distinguished. |
| Runtime owner | `jaxstro.constants` owns raw numeric values and their source comments. |
| Shape and unit policy | Public values are scalar Python numbers; units are encoded by symbol names and documentation rather than a runtime `Quantity` wrapper. |
| Transform boundary | Constants may appear inside `jit`, `vmap`, and `grad`, but they are static coefficients rather than differentiated inputs. |
| Evidence | Source-backed cards in [](../../50-api/research-infrastructure/source-provenance/constants.md) and constant tests check values, derivations, and stored-precision identities. |
| Downstream interpretation boundary | A constant does not select a domain unit system, define code units, or establish that a model using it is scientifically valid. |

## Coefficients need conventions

A number becomes interpretable only after its unit and authority are known. For
example, Newtonian gravity in CGS uses

```{math}
:label: eq-constants-newtonian-gravity

F = G\frac{m_1m_2}{r^2},
\qquad
[G] = \mathrm{cm^3\,g^{-1}\,s^{-2}}.
```

`G_CGS` is therefore compatible with masses in grams, lengths in centimeters, and
times in seconds. Substituting a value expressed in another unit system without a
conversion produces a numerically valid array and a physically invalid calculation.

Selected conventions include:

- `G_CGS`, `C_CGS`, `H_CGS`, `K_B`, and related microphysical constants use the
  recorded CODATA 2018 values or exact revised-SI definitions converted to CGS.
- `RSUN_CM`, `LSUN_ERG_S`, and `TEFF_SUN` are IAU nominal conversion constants. They
  are not measurements of a time-varying Sun.
- `MSUN_G` is a rounded compatibility conversion derived from the exact IAU nominal
  solar mass parameter and Jaxstro's frozen `G_CGS`; it is not an exact nominal
  solar mass defined by the IAU.
- `AB_ZEROPOINT_JY` records the conventional 3631 Jy AB reference value and cites
  its photometric source.

See [](#eq-constants-newtonian-gravity) when checking that a kernel's base units
match the constant it consumes.

## Raw constants and quantity constants

`jaxstro.constants` remains the backwards-compatible raw-CGS surface. The separate
`jaxstro.quantity.constants` module wraps selected values in explicit units and
attaches inspectable metadata:

```python
from jaxstro import constants as C
import jaxstro.quantity as q

raw_g = C.G_CGS
typed_g = q.constants.G
source = q.constants.metadata("G")
```

The two surfaces answer different boundary needs. Raw constants keep inner kernels
small; quantity constants make dimensional checks explicit at public boundaries.

:::{warning} A source is not a model validation
A cited coefficient can be correct while the surrounding approximation, geometry,
or domain model is wrong. Constant provenance supports the coefficient claim only.
:::

## Shape, transforms, and evidence

The values are scalar and broadcast according to ordinary JAX rules when combined
with arrays. Gradients flow through variables around a constant, but there is no
runtime uncertainty model for the coefficient itself. If a research question treats
a constant as uncertain, the caller must represent that uncertain parameter
explicitly rather than differentiate with respect to a module-level literal.

The generated source cards identify exact sources, stored values, derivations, code
locations, and assertion-bearing tests. They do not prove the scientific adequacy of
every downstream equation that imports the constants.
