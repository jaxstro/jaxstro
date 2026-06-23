---
title: Quantities, units, and dimensional boundaries
description: >-
  The conceptual contract for jaxstro.quantity: exact dimensions, static units,
  JAX-traceable values, explicit equivalencies, and raw-array kernels inside the
  foundation.
---

`jaxstro.quantity` is the unit-aware boundary layer for code that needs physical
inputs without smuggling unit mistakes into a differentiable kernel. A
`Quantity` is one JAX value plus one static `Unit`; arrays are allowed, but every
element in the array shares the same unit.

```python
import jax.numpy as jnp
import jaxstro.quantity as q

radius = jnp.array([1.0, 2.0]) * q.Rsun
mass = q.Quantity(1.0, q.Msun)
density = mass / radius**3
```

The value is the dynamic PyTree child. The unit is immutable auxiliary metadata,
so `jax.jit`, `jax.vmap`, and `jax.grad` trace through values while dimensions and
scale factors stay static and auditable.

## Exact dimensions

Dimensions are fixed exponent vectors with exact rational powers. That is why
`q.cm ** Fraction(1, 2)` is valid and `q.cm ** 0.5` is rejected at the algebra
layer: decimals must be rationalized by the parser or caller before they become
dimension metadata.

Addition and subtraction require compatible dimensions; the right operand is
converted to the left operand's unit and the result keeps the left unit.
Multiplication, division, and powers combine units algebraically. Raw scalars may
scale any quantity, but they may only add to dimensionless quantities.

## Parser and serialization

The parser accepts common expressions without executing Python code:

```text
cm
km/s
Msun/yr
erg / s / cm^2 / Hz
g cm^2 s^-2
(km/s)^2
cm^(1/2)
cm^0.5
sqrt(cm)
```

Decimal powers are accepted only when they cleanly rationalize under the parser
rule. Canonical formatting writes rational powers, for example `cm^(1/2)` instead
of `cm^0.5`.

Scalar quantities serialize compactly when the unit can be parsed:

```json
{"value": 1.0, "unit": "Msun/yr"}
```

Custom units fall back to a structured payload with symbol, CGS scale, and exact
dimension powers. Array-valued quantity serialization is deliberately deferred to
explicit array helpers.

## Bases and roles

Named bases are presentation profiles. They do not replace direct conversion:

```python
mass.to(q.Msun)
radius.to_basis(q.bases.STELLAR, role="stellar_radius")
```

Roles disambiguate domain conventions. In the `PLANETARY` basis,
`stellar_mass` maps to `Msun`, `planet_mass` maps to `g`, and `orbit` maps to
`AU`. `DYNAMICAL` aliases `STAR_CLUSTER`; `EXOPLANET` aliases `PLANETARY`.

## Constants and equivalencies

Quantity constants mirror `jaxstro.constants` for backwards-compatible CGS
values, while carrying inspectable provenance metadata:

```python
q.constants.G
q.constants.metadata("G")
q.constants.raw_value_cgs("G")
```

Exact dimensional conversion is the default. Physically related but
dimensionally different conversions require explicit equivalencies:

```python
(500 * q.nm).to(q.Hz, equivalencies=q.equivalencies.spectral())
(1 * q.K).to(q.erg, equivalencies=q.equivalencies.temperature_energy())
(1 * q.g).to(q.erg, equivalencies=q.equivalencies.mass_energy())
```

## Boundary pattern

Library kernels should validate quantities at the public boundary, convert once,
and run raw JAX arrays internally:

```python
def public_api(radius):
    radius_cm = radius.to_value(q.cm)
    return _raw_kernel(radius_cm)
```

This keeps units visible where humans call the API and keeps the differentiable
kernel small, fast, and ordinary.

## Exclusions

Version 1 does not provide broad NumPy dispatch, offset temperature units,
logarithmic magnitude units, mixed-unit table containers, or automatic domain
equivalencies. Those features need separate semantics and tests before they
belong in the foundation.
