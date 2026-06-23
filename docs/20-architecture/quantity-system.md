---
title: Quantity system architecture
description: >-
  The jaxstro.quantity design: concrete units, dimensional analysis,
  JAX-aware quantities, astrophysical bases, exact parsing, and migration from
  legacy unit systems.
---

`jaxstro.quantity` is the unit-aware value layer for jaxstro. It is the
computational-astrophysics counterpart to observation-led unit ecosystems:
concrete enough for day-to-day science, strict enough for library internals, and
light enough to stay inside the foundation package.

The system is additive. Existing `jaxstro.units` code remains stable while
downstream packages migrate at their own pace.
[](../30-decisions/0006-build-own-quantity-not-unxt.md) records why jaxstro
builds this layer in-house instead of adopting a heavy dependency.

## Public shape

The user-facing namespace is intentionally compact:

```python
import jaxstro.quantity as q

mass = q.Quantity(1.0, q.Msun)
radius = 2.0 * q.Rsun
speed = jnp.array([10.0, 20.0]) * q.km / q.s

density = mass / radius**3
density.to(q.g / q.cm**3)
mass.to_basis(q.bases.STELLAR, role="stellar_mass")
```

The implementation is a layered package rather than one crowded module:

```text
jaxstro.quantity
  dimensions.py        # fixed-vector dimensions + symbolic public names
  unit.py              # Unit object, unit algebra, scale/dimension metadata
  quantity.py          # Quantity PyTree, arithmetic, conversion
  units.py             # core SI/CGS units and documented prefixed units
  astro.py             # astronomy units: Msun, Rsun, AU, pc, yr, micron...
  registry.py          # layered registries and scoped extension registries
  parser.py            # unit expression parser and canonical formatter
  bases.py             # CGS, SI, STELLAR, PLANETARY, STAR_CLUSTER...
  constants.py         # versioned constants as quantities + raw helpers
  equivalencies.py     # explicit spectral/temp-energy/mass-energy conversions
  math.py              # dimension-aware math wrappers
  serialization.py     # compact + structured quantity/unit serialization
  errors.py            # rich structured exceptions
```

`jaxstro.quantity.__init__` re-exports common units and types so examples stay
short: `q.Quantity`, `q.Unit`, `q.cm`, `q.erg`, `q.Msun`, `q.micron`,
`q.bases.STELLAR`, `q.constants.G`, and `q.math.sqrt`.

## Data model

A `Quantity` is a JAX value plus one static unit:

```python
Quantity(value=jnp.asarray(...), unit=q.cm)
```

A `Unit` is immutable scale and dimensional metadata:

```python
Unit(
    symbol="cm",
    scale_to_cgs=1.0,
    dimensions=dimensions.length,
    metadata=UnitMetadata(...),
)
```

Dimensions use a fixed canonical exponent vector internally, with exact rational
exponents and readable symbolic names such as `dimensions.energy`,
`dimensions.velocity`, and `dimensions.dimensionless`. This keeps equality,
hashing, serialization, and unit algebra stable while preserving a friendly API.

One `Quantity` array has one unit. Mixed-unit tables belong in a later table
layer, not in the primitive scalar/array quantity object.

## Arithmetic

Ordinary arithmetic follows scientific-Python expectations:

- Addition and subtraction require compatible dimensions. The right operand is
  converted to the left operand's unit, and the result keeps the left unit.
- Multiplication and division combine units algebraically.
- Powers accept integer and exact rational exponents.
- Comparisons require compatible dimensions and use the left operand's unit.
- Raw Python and JAX scalars are dimensionless. They may scale any quantity, but
  they may only add to dimensionless quantities.

`Quantity` is a PyTree: values trace through JAX, while units and dimensions are
static metadata. Public jaxstro APIs may accept quantities, but computational
kernels should validate and convert at the boundary, then run on plain JAX
arrays. That keeps `jit`, `vmap`, and `grad` behavior simple and auditable.

Dimension-aware math wrappers live in `jaxstro.quantity.math`. Version 1 covers
core arithmetic and high-value wrappers such as square roots, logarithms,
exponentials, trigonometric functions, reductions, and `where`-style helpers
where dimensional semantics are clear. A broader NumPy/JAX interop layer is a
later roadmap item.

Angles use a hybrid rule: radians are dimensionless for scale algebra, but angle
units carry an angle semantic tag. Trigonometric wrappers accept tagged angles,
and validation-sensitive APIs can require explicit angle semantics.

Errors are structured and readable. Exceptions such as `DimensionError`,
`UnitConversionError`, and `UnitParseError` carry fields like `operation`,
`expected`, `actual`, `left_unit`, and `right_unit` so tests can assert behavior
without matching fragile strings.

## Parser and serialization

The parser accepts common astrophysical unit expressions without executing
Python code:

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
(erg/cm^3)^0.5
```

Input is versatile, but the internal representation is exact. Decimal powers are
accepted only when they rationalize cleanly under a documented rule; canonical
serialization prints rational exponents, for example `cm^(1/2)` instead of
`cm^0.5`. Ambiguous repeating decimals should raise a helpful error that asks
the user to write the rational exponent explicitly.

Aliases are deliberately strict. A small documented set such as `Msun`/`msun`
is easier to teach and serialize than a large alias garden. Unknown symbols get
friendly suggestions rather than loose normalization.

Known units serialize compactly:

```json
{"value": 1.0, "unit": "Msun/yr"}
```

Custom or archival units have a structured fallback:

```json
{
  "value": 1.0,
  "unit": {
    "symbol": "code_mass",
    "scale_cgs": 1.989e33,
    "dimensions": {"mass": 1}
  }
}
```

## Registries and bases

Registries are layered. Core SI/CGS units live in the core registry; astronomy
units live in the astro registry; downstream packages create scoped extension
registries for package-specific units. Global registration is reserved for
interactive convenience, not reproducible package code.

SI prefixes use a hybrid model. Common documented prefixed objects such as
`km`, `nm`, `micron`, `um`, `ms`, and `MHz` exist directly. The parser may
generate standard SI prefixes for unambiguous core units, but it must not invent
ambiguous astrophysical prefixed symbols unless they are explicitly registered.

Named bases are role-aware presentation profiles, not replacements for direct
unit conversion:

```python
q.bases.CGS
q.bases.SI
q.bases.STELLAR
q.bases.PLANETARY
q.bases.STAR_CLUSTER
q.bases.CLOSE_BINARY
q.bases.WIDE_BINARY
q.bases.COMPACT_BINARY
```

Direct conversion remains the primitive:

```python
mass.to(q.Msun)
radius.to(q.Rsun)
```

Role-aware bases make domain workflows natural:

```python
planet_mass.to_basis(q.bases.PLANETARY, role="planet_mass")
star_mass.to_basis(q.bases.PLANETARY, role="stellar_mass")
semi_major_axis.to_basis(q.bases.PLANETARY, role="orbit")
```

## Constants and equivalencies

Constants are quantities plus raw-value helpers:

```python
q.constants.G
q.constants.c
q.constants.sigma_sb
q.constants.k_B
```

The registry is versioned and inspectable. Physical constants should identify
their CODATA release; solar and astronomical constants should identify their IAU
nominal-value source. Runtime remains self-contained, but constants carry enough
metadata for docs, validation, and downstream provenance.

Exact dimensional conversion is the default. Physically related but
non-identical conversions require explicit equivalencies:

```python
wavelength.to(q.Hz, equivalencies=q.equivalencies.spectral())
temperature.to(q.erg, equivalencies=q.equivalencies.temperature_energy())
mass.to(q.erg, equivalencies=q.equivalencies.mass_energy())
```

Version 1 supports Kelvin but not offset temperature units such as Celsius or
Fahrenheit. It also defers full logarithmic and magnitude units; `mag` and `dex`
need careful zero-point and context semantics before they should enter the
foundation.

## Migration posture

The first release of `jaxstro.quantity` is additive. `jaxstro.units` keeps its
existing `UnitSystem` behavior, aliases, and `DEFAULT_UNITS` policy. Quantity is
the future refactor target for fluxax, gravax, progenax, and related packages,
but deprecations wait until those downstream migrations have a tested path.

For compatibility, each legacy `UnitSystem` may expose representative quantity
units through `quantity_units` and `quantity_scales`. These helpers are bridges
for boundary conversion and migration notebooks; they do not replace
`UnitSystem`, change existing aliases, or emit deprecation warnings.

The docs should teach both worlds: legacy `UnitSystem` patterns for existing
code and quantity-first boundary validation for new code. Package internals
should converge on quantity-aware public APIs and raw-array internal kernels.
