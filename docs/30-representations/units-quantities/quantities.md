---
title: Quantities, units, and dimensional boundaries
description: >-
  Exact dimensions, static units, JAX-traceable values, parsing, serialization, and
  raw-array kernel boundaries.
---

Use this page when a public API must reject dimensional mistakes before values enter
a differentiable raw-array kernel.

:::{important} Implemented Jaxstro capability
`jaxstro.quantity` provides immutable units, exact dimensions, a PyTree `Quantity`,
explicit conversion, strict parsing, scalar serialization, bases, and math wrappers.
:::

## Representation contract

| Contract field | Current representation |
| --- | --- |
| Mathematical object | A numeric scalar or array paired with one immutable unit whose dimension exponents are exact rational exponents. |
| Physical convention | Every unit has a scale to CGS; exact rational exponents define dimensional compatibility and canonical algebra. |
| Runtime owner | `jaxstro.quantity` owns `Quantity`, `Unit`, dimensions, registries, parsing, bases, serialization, and math wrappers. |
| Shape and unit policy | The numeric value may have any JAX array shape, but the full value shares one unit; mixed-unit arrays are outside the primitive. |
| Transform boundary | Values support fixed-unit `jit`, `vmap`, and `grad`; unit metadata, parsing, registry mutation, and Python exception paths remain static. |
| Evidence | Quantity unit and integration tests check algebra, conversions, parser failures, PyTree behavior, JAX transforms, and serialization round trips. |
| Downstream interpretation boundary | A valid dimension does not select a physical model, approve ecosystem migration, or define domain-specific table and catalog semantics. |

## Exact dimensions

If a quantity has dimensions

```{math}
:label: eq-quantity-dimension-vector

[x]
=
\mathsf{M}^{a}
\mathsf{Len}^{b}
\mathsf{t}^{c}
\mathsf{Temp}^{d}
\mathsf{I}^{e}
\mathsf{N}^{f}
\mathsf{Lum}^{g},
\qquad
a,\ldots,g \in \mathbb{Q},
```

the factors map to the runtime dimension tuple in this exact order:

| Runtime dimension | Equation factor | Exponent |
| --- | --- | --- |
| `mass` | `\mathsf{M}` | `a` |
| `length` | `\mathsf{Len}` | `b` |
| `time` | `\mathsf{t}` | `c` |
| `temperature` | `\mathsf{Temp}` | `d` |
| `current` | `\mathsf{I}` | `e` |
| `amount` | `\mathsf{N}` | `f` |
| `luminosity` | `\mathsf{Lum}` | `g` |

The distinct `\mathsf{Len}` and `\mathsf{Lum}` factors avoid using one symbol for
both length and luminosity. Dimensional equality means equality of all seven entries
in this complete exponent vector. The exact rational representation makes unit
equality, hashing, powers, and serialization stable. At the algebra layer,
`q.cm ** Fraction(1, 2)` is accepted while a raw floating `0.5` is rejected; the
parser may rationalize a documented decimal input before constructing metadata.

```python
import jax.numpy as jnp
import jaxstro.quantity as q

radius = jnp.array([1.0, 2.0]) * q.Rsun
mass = q.Quantity(1.0, q.Msun)
density = mass / radius**3
```

Addition and subtraction require compatible dimensions. The right operand is
converted to the left unit, and the result keeps the left unit. Multiplication,
division, and exact powers combine unit algebra. Raw scalars may scale a quantity but
may add only to a dimensionless quantity.

## PyTree and transform behavior

The value is the dynamic PyTree child. The unit is immutable auxiliary metadata, so
`jax.jit`, `jax.vmap`, and `jax.grad` trace numeric values while dimensions and scale
factors remain static. A transform is meaningful only along a fixed unit path. Unit
selection and dimensional failures are Python boundary behavior, not differentiable
branches.

:::{warning} Dimensionally valid does not mean physically smooth
Clipping, thresholds, branch changes, and singular model relations can still make a
quantity-valued program nonsmooth. Quantity checks establish dimensions, not the
scientific meaning of a derivative.
:::

## Parser and serialization

The parser accepts expressions such as `km/s`, `Msun/yr`, `cm^(1/2)`, `cm^0.5`,
and `sqrt(cm)` without executing Python. Decimal powers are accepted only when they
rationalize under the parser rule; canonical formatting writes rational powers.

Known scalar quantities serialize compactly:

```json
{"value": 1.0, "unit": "Msun/yr"}
```

Custom units use a structured payload containing the symbol, CGS scale, and exact
dimensions. Array-valued quantity serialization is deliberately deferred to explicit
array helpers so shape, dtype, and storage policy cannot be hidden in a JSON scalar
contract.

## Bases, constants, and equivalencies

Named bases are role-aware presentation profiles, not replacements for direct
conversion. Quantity constants mirror the backwards-compatible values in
`jaxstro.constants` while adding unit and source metadata. Dimensionally different
relations require the explicit equivalencies described in [](./equivalencies.md).

## Boundary pattern

Library APIs should validate a quantity, convert once, and pass an ordinary array to
the numerical core:

```python
def public_api(radius):
    radius_cm = radius.to_value(q.cm)
    return _raw_kernel(radius_cm)
```

This pattern keeps the human-facing boundary explicit and the compiled kernel small.
It also makes the contract in [](#eq-quantity-dimension-vector) inspectable without
making unit metadata a dynamic value in every internal operation.

Version 1 does not provide broad NumPy dispatch, offset temperatures, logarithmic
magnitudes, mixed-unit table containers, or automatic domain equivalencies. Those
features require separate scientific semantics and evidence.
