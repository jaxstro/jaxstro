# jaxstro Quantity Design

## Goal

Build `jaxstro.quantity` as a full-featured, JAX-aware quantity system for
theorists and computational astrophysicists: concrete units, dimension-safe
arithmetic, exact parsing, astrophysical bases, versioned constants, explicit
equivalencies, and a soft migration path from existing `jaxstro.units`.

## Context

The current `jaxstro.units` module provides `UnitSystem` objects and named code
unit systems. That layer remains useful and stable, but downstream packages need
a richer quantity model for user-facing APIs, configuration, provenance, paper
tables, and boundary validation. Fluxax, gravax, progenax, and related packages
should eventually be able to accept physical quantities directly, validate their
dimensions, convert once at the boundary, and run raw JAX kernels internally.

ADR 0006 already chooses an in-house quantity layer over unxt/quax. This design
expands that decision from a minimal class into a SoTA foundation package.

## Architecture

Use a layered package:

```text
jaxstro.quantity
  __init__.py
  dimensions.py
  unit.py
  quantity.py
  units.py
  astro.py
  registry.py
  parser.py
  bases.py
  constants.py
  equivalencies.py
  math.py
  serialization.py
  errors.py
```

`jaxstro.quantity.__init__` re-exports the ergonomic public surface:

```python
import jaxstro.quantity as q

q.Quantity
q.Unit
q.cm
q.km
q.s
q.erg
q.Msun
q.Rsun
q.micron
q.bases.STELLAR
q.constants.G
q.math.sqrt
```

This keeps the implementation modular while letting users write compact
scientific code.

## Core model

`Quantity = value + Unit`.

`Unit = symbol/name + scale_to_cgs + dimensions + metadata`.

Dimensions are fixed canonical exponent vectors internally, with exact rational
exponents and symbolic public names such as `dimensions.energy`,
`dimensions.velocity`, and `dimensions.dimensionless`.

Quantities may hold scalars or arrays, but one `Quantity` has one unit. Per-row
or mixed-unit tables are deferred to a later table layer.

## Arithmetic

Version 1 supports core arithmetic and dimension-aware math wrappers.

- Addition/subtraction require compatible dimensions. The right operand converts
  to the left operand's unit; the result keeps the left operand's unit.
- Multiplication/division combine units algebraically.
- Powers support integer and exact rational exponents.
- Comparisons require compatible dimensions and convert right to left.
- Raw Python/JAX scalars are dimensionless. They can scale any quantity; they
  can only add to dimensionless quantities.
- Dimensionless scalar quantities are represented by ordinary `Quantity` values
  with the dimensionless unit, plus a convenience constructor such as
  `q.scalar(...)`.

Angles use a hybrid model: dimensionless for scale algebra, tagged as angles for
math wrappers and validation.

`Quantity` is a JAX PyTree with dynamic traced values and static unit metadata.
The coding pattern for jaxstro internals is quantity-aware public APIs and raw
array kernels:

```python
def public_api(radius, mass):
    radius_value = q.require(radius, q.dimensions.length).to_value(q.cm)
    mass_value = q.require(mass, q.dimensions.mass).to_value(q.g)
    return _raw_kernel(radius_value, mass_value)
```

Broad NumPy/JAX dispatch is intentionally deferred until the core semantics are
stable.

## Errors

Use rich structured exceptions:

- `DimensionError`
- `UnitConversionError`
- `UnitParseError`
- `UnitRegistryError`
- `EquivalencyError`

Messages should explain what happened, what was expected, what was provided,
and the obvious fix when one exists. Exceptions should also carry structured
fields such as `operation`, `expected`, `actual`, `left_unit`, and `right_unit`
for downstream tooling and tests.

## Registries

Use layered registries:

- core registry: SI/CGS/base units and common prefixed units;
- astro registry: `Msun`, `Rsun`, `Lsun`, `AU`, `pc`, `yr`, `day`, `micron`,
  `um`, `nm`, `Angstrom`, and related astronomy units;
- scoped downstream registries for package-specific units;
- global registration only for interactive convenience.

Aliases are strict and deliberately small. For solar mass, support `Msun` and
`msun`, not a long list of variants. The parser should suggest close matches
instead of accepting loose spellings silently.

SI prefixes use a hybrid approach: documented prefixed units exist as concrete
objects, and the parser can generate standard SI-prefixed units where the symbol
is unambiguous.

## Parser

Target grammar supports:

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

Rules:

- no Python `eval`;
- registered symbols and documented aliases only;
- explicit multiplication through `*`, whitespace, or adjacent parsed factors;
- `/` for division;
- powers with `^` and optionally `**`;
- parentheses;
- `sqrt(...)` syntax sugar;
- integer, rational, and cleanly rationalizable decimal powers;
- exact rational internal exponents;
- canonical serialization as rational powers, not decimals.

Ambiguous repeating decimals should raise a helpful error that asks the user to
write the rational exponent explicitly, for example `cm^(1/3)`.

## Serialization

Known units use compact serialization:

```json
{"value": 1.0, "unit": "Msun/yr"}
```

Custom units use a structured fallback:

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

Round-tripping should be deterministic. Registry-backed compact strings are the
default for configs, provenance, and paper-table metadata.

## Bases

Named bases are role-aware presentation/conversion profiles:

- `CGS`
- `SI`
- `STELLAR`
- `PLANETARY`
- `STAR_CLUSTER`
- `CLOSE_BINARY`
- `WIDE_BINARY`
- `COMPACT_BINARY`

Direct conversion remains primitive:

```python
mass.to(q.Msun)
radius.to(q.Rsun)
```

Role-aware conversion handles field conventions:

```python
planet_mass.to_basis(q.bases.PLANETARY, role="planet_mass")
star_mass.to_basis(q.bases.PLANETARY, role="stellar_mass")
semi_major_axis.to_basis(q.bases.PLANETARY, role="orbit")
```

Field vocabulary aliases are allowed, for example `DYNAMICAL` as an alias for
`STAR_CLUSTER` and `EXOPLANET` as an alias for `PLANETARY`.

## Constants

Constants are quantities plus raw helpers. Runtime values are self-contained,
but each constant carries versioned provenance:

- physical constants from a pinned CODATA release;
- solar and astronomical nominal constants from pinned IAU sources.

Examples:

```python
q.constants.G
q.constants.c
q.constants.sigma_sb
q.constants.k_B
q.constants.default_set
```

Implementation must verify and cite current CODATA/IAU source values before
landing constants.

## Equivalencies

Exact dimensional conversion is the default. Physically related conversions use
explicit equivalencies:

- `spectral()`: wavelength, frequency, and photon energy;
- `temperature_energy()`: temperature and energy through `k_B`;
- `mass_energy()`: mass and energy through `c^2`.

No implicit spectral conversion in v1.

## Deferred features

- Broad NumPy/JAX interop dispatch.
- Celsius/Fahrenheit and other affine offset units.
- Full logarithmic and magnitude units such as `mag`, `dex`, and `dB`.
- Mixed-unit table containers.
- Automatic domain-profile equivalencies.

The docs should explain these exclusions as deliberate scope boundaries, not
missing pieces.

## Documentation requirements

The first Quantity release needs tutorial-grade docs:

- conceptual guide to quantities, units, dimensions, and bases;
- API reference;
- parser grammar and canonical serialization;
- dimension-aware math wrappers;
- constants and provenance;
- equivalencies;
- migration from `jaxstro.units`;
- downstream refactor examples for fluxax/gravax/progenax-style APIs;
- validation/provenance page.

## Acceptance criteria

- Existing `jaxstro.units` behavior remains backwards compatible.
- `jaxstro.quantity` supports ergonomic construction with both
  `q.Quantity(value, unit)` and `value * q.cm`.
- Arithmetic, conversion, parsing, serialization, bases, constants, and
  equivalencies have focused tests.
- JAX transform tests cover representative `jit`, `vmap`, and `grad` paths.
- Docs build cleanly and clearly mark implemented v1 behavior versus roadmap
  features.
