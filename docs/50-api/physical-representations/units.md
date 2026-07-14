---
title: Unit systems
---

# Unit systems

## Owner import path

`jaxstro.units`

## Purpose

The canonical ecosystem `UnitSystem` contract and explicit scale conversions.

## Public records and callables

`UnitSystem`, `PhotometricUnits`, `CGS`, `ASTRO_STELLAR`,
`ASTRO_DYNAMICAL`, `ASTRO_PLANETARY`, `DEFAULT`, `STELLAR`, `STAR`,
`BINARY`, `SOLAR`, `PLANETARY`, `SOLAR_PHOTOMETRIC`, `CGS_PHOTOMETRIC`,
`UNIT_SYSTEMS`, and `get_units`.

## Shape and dtype expectations

Unit systems are static metadata containing scalar CGS scales. Conversion
methods accept broadcast-compatible numeric values.

## JAX transforms and AD classification

Scale multiplication and division compose with JAX transforms. Unit choice is
static and not a traced global context.

## Failure behavior

Unknown named systems raise. Core APIs require explicit units or physical
constants; convenience wrappers may resolve `units=None` to CGS.

## Contract and evidence links

See [](../../30-representations/units-quantities/quantities.md),
[](../../30-decisions/0007-cgs-as-default-units.md), and the
[source-backed cards](../research-infrastructure/source-provenance/constants.md).

## Canonical import example

```python
from jaxstro.units import ASTRO_DYNAMICAL
```
