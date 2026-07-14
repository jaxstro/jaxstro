---
title: Unit systems and quantity adoption
description: >-
  The current jaxstro.units contract, the implemented jaxstro.quantity layer, and
  their explicit migration boundary.
---

Use this page when choosing between a named mass-length-time unit system and an
explicit unit-aware value at a public scientific boundary.

:::{important} Implemented Jaxstro capability
`jaxstro.units` is the canonical ecosystem unit-system contract, and
`jaxstro.quantity` is an implemented evaluation surface. Ecosystem adoption and any
replacement cutover remain deferred pending downstream evidence.
:::

## Representation contract

| Contract field | Current representation |
| --- | --- |
| Mathematical object | A `UnitSystem` is a named mass-length-time basis; a `Quantity` is a JAX value paired with one immutable `Unit`. |
| Physical convention | CGS is the canonical ecosystem base; named canonical ecosystem unit systems declare mass, length, and time scales, while quantity units store exact dimensions and a scale to CGS. |
| Runtime owner | `jaxstro.units` and `jaxstro.quantity` own the two additive, importable surfaces. |
| Shape and unit policy | `UnitSystem` methods convert scalar or array values by explicit scale factors; one `Quantity` array has one unit and one shared dimension vector. |
| Transform boundary | Fixed scale conversion and quantity arithmetic support JAX array transforms; registry lookup, parsing, basis-role choice, and migration policy remain static host decisions. |
| Evidence | Unit conversion tests, quantity algebra/parser/serialization tests, and compatibility tests verify behavior without claiming downstream adoption. |
| Downstream interpretation boundary | Domain packages choose their default systems, code units, roles, and migration schedule; Jaxstro does not hide a domain default in core kernels. |

## Two current representations

`jaxstro.units.UnitSystem` represents a basis through the CGS value of one mass,
length, and time unit. If a physical quantity has dimensions
$M^aL^bT^c$, conversion between two systems follows

```{math}
:label: eq-unit-system-scale

x_{\mathrm{target}}
=
x_{\mathrm{source}}
\frac{M_{\mathrm{source}}^aL_{\mathrm{source}}^bT_{\mathrm{source}}^c}
     {M_{\mathrm{target}}^aL_{\mathrm{target}}^bT_{\mathrm{target}}^c}.
```

Named systems include `CGS`, `ASTRO_STELLAR`, `ASTRO_DYNAMICAL`, and
`ASTRO_PLANETARY`, with documented aliases. `DEFAULT_UNITS` is `CGS` because Jaxstro
is the domain-agnostic foundation. Downstream packages may select another default for
their own domain, but core Jaxstro APIs require explicit units or an explicit
physical constant such as `G`.

`jaxstro.quantity` represents the unit on the value itself:

```python
import jax.numpy as jnp
import jaxstro.quantity as q

mass = q.Quantity(1.0, q.Msun)
radius = 2.0 * q.Rsun
speed = jnp.array([10.0, 20.0]) * q.km / q.s
density = (mass / radius**3).to(q.g / q.cm**3)
```

The system is additive. Existing `jaxstro.units` behavior remains stable while
downstream packages evaluate quantity migration at their own boundaries. The in-house
design rationale is recorded in [](../../30-decisions/0006-build-own-quantity-not-unxt.md).

## Quantity data model

A `Quantity` is a registered JAX PyTree whose dynamic child is the value. Its unit is
static auxiliary metadata. A `Unit` is immutable and records a symbol, CGS scale,
exact fixed-vector dimensions, and small semantic metadata such as the angle tag.

This split allows numeric values to trace while units remain inspectable. One
`Quantity` array has one unit. Mixed-unit table columns and row-wise unit tags require
a different container and are not part of the primitive.

The implementation is layered:

```text
jaxstro.quantity
  dimensions.py      exact canonical dimension vectors
  unit.py            immutable Unit and algebra
  quantity.py        Quantity PyTree and arithmetic
  units.py            core CGS and SI units
  astro.py            astronomical units
  registry.py         layered exact-symbol registries
  parser.py           expression parser and canonical formatter
  bases.py            role-aware presentation bases
  constants.py        quantity constants and metadata
  equivalencies.py    explicit cross-dimension relations
  math.py             dimension-aware JAX wrappers
  serialization.py    compact and structured scalar payloads
  errors.py           structured quantity exceptions
```

## Arithmetic and static metadata

Addition and subtraction require compatible dimensions, convert the right operand to
the left operand's unit, and preserve the left unit. Multiplication and division
combine dimensions. Powers require integer or exact rational exponents. Raw scalars
can scale any quantity but can only add to dimensionless quantities.

Angle units are dimensionless for scale algebra but carry an angle semantic tag.
`q.math.sin` and `q.math.cos` require that tag and convert to radians before
evaluation. Logarithms and exponentials require dimensionless input.

:::{warning} Static metadata creates a compilation boundary
A JIT-compiled function specializes to unit metadata. Passing the same numeric shape
with a different unit can require another trace. Parsing or choosing a registry entry
inside a traced kernel is not a supported dynamic operation.
:::

## Registries, parsing, and bases

Core and astronomy registries use exact symbols with a small documented alias set.
The parser accepts products, divisions, parentheses, `sqrt(...)`, rational powers,
and decimal powers that rationalize under the documented bound. Canonical formatting
writes exact rational exponents.

Registries are layered so downstream packages can build scoped extensions. Global registration is reserved for interactive convenience, not reproducible package code.
Unknown symbols fail with structured errors rather than loose normalization.

Named quantity bases such as `CGS`, `SI`, `STELLAR`, `PLANETARY`, and
`STAR_CLUSTER` are role-aware presentation profiles. They do not replace direct unit
conversion. A planetary profile can map a stellar mass, a planet mass, and an orbit
to different units because those roles are explicit.

## Migration boundary

Each legacy `UnitSystem` exposes representative `quantity_units` and
`quantity_scales` bridges. They support boundary conversion and migration notebooks;
they do not change aliases, emit deprecations, or approve a cutover.

The current evidence verifies implementation behavior, compatibility, serialization,
and JAX transformations in Jaxstro. Ecosystem adoption requires separate downstream
parity, performance, serialization, ergonomics, and migration-cost evidence. See
[](#eq-unit-system-scale) for the scale relation both surfaces must preserve.
