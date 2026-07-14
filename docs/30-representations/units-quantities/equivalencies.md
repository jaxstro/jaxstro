---
title: Equivalencies and representation changes
description: >-
  Explicit conversions between physically related but dimensionally distinct
  quantities.
---

Use this page when two values describe the same physical state through different
dimensions, such as wavelength and frequency, and ordinary unit conversion correctly
refuses the change.

:::{important} Implemented Jaxstro capability
`jaxstro.quantity` implements explicit spectral, temperature-energy, and mass-energy
equivalencies. No equivalency is enabled globally or selected implicitly.
:::

## Representation contract

| Contract field | Current representation |
| --- | --- |
| Mathematical object | A named bidirectional physical relation that maps a `Quantity` to a target `Unit` with different dimensions. |
| Physical convention | `c`, `h`, and `k_B` use Jaxstro's frozen CGS constants; wavelength-frequency-energy, temperature-energy, and mass-energy relations are explicit equivalencies. |
| Runtime owner | `jaxstro.quantity`, through `jaxstro.quantity.equivalencies`, owns construction and dispatch. |
| Shape and unit policy | Scalar or array-valued quantities retain their value shape; the target unit is explicit and output values share that one unit. |
| Transform boundary | Fixed equivalency arithmetic is JAX-array compatible; unit choice and equivalency selection remain static Python metadata. |
| Evidence | Quantity equivalency unit tests cover round trips, incompatible targets, and JAX transformations on fixed conversion paths. |
| Downstream interpretation boundary | An equivalency does not choose a spectral convention, thermodynamic model, rest-frame policy, or relativistic approximation for the caller. |

## Why direct conversion must fail

Unit conversion changes scale while preserving dimensions. An equivalency invokes a
physical relation. Wavelength and frequency are related by

```{math}
:label: eq-equivalency-spectral

\nu = \frac{c}{\lambda},
\qquad
E = h\nu.
```

Because length, inverse time, and energy are different dimensions, converting between
them without naming [](#eq-equivalency-spectral) would hide a modeling decision.

```python
import jaxstro.quantity as q

wavelength = 500 * q.nm
frequency = wavelength.to(q.Hz, equivalencies=q.equivalencies.spectral())
energy = frequency.to(q.erg, equivalencies=q.equivalencies.spectral())
```

The other implemented relations are

```{math}
:label: eq-equivalency-energy-relations

E = k_B T,
\qquad
E = mc^2.
```

They are requested with `temperature_energy()` and `mass_energy()`, respectively.

## Static choice, dynamic values

The equivalency objects and target units are static Python choices. Array values flow
through JAX operations, so fixed-path `jit`, `vmap`, and derivatives with respect to
the numeric values are meaningful on finite positive domains. Changing which
equivalency is active is not a differentiable operation.

:::{warning} Equivalent coordinates do not imply equivalent densities
Changing a spectral coordinate from wavelength to frequency also changes a density
by its Jacobian. Use `jaxstro.spectra` for `F_lambda` and `F_nu` transformations;
converting only the axis quantity is insufficient.
:::

## Where the claim stops

Round-trip and transformation tests show that the implemented relations evaluate
consistently. They do not establish that a downstream workflow chose the correct
rest frame, temperature interpretation, or relativistic regime. That scientific
choice remains visible at the caller boundary.
