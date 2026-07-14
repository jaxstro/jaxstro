---
title: Physical constants
---

# Physical constants

## Owner import path

`jaxstro.constants`

## Purpose

Versioned CGS physical constants, exact nominal conversions, compatibility
scales, and photometric zero points.

## Public records and callables

The module exports sourced scalar constants including `G_CGS`, `C_CGS`,
`H_CGS`, `K_B`, `SIGMA_SB`, `A_RAD`, `MSUN_G`, `RSUN_CM`, `LSUN_ERG_S`,
`PC_CM`, `AU_CM`, `JY_CGS`, and the declared conversion factors in
`jaxstro.constants.__all__`.

## Shape and dtype expectations

Constants are frozen Python floating scalars in the units named by each symbol;
there is no runtime source lookup.

## JAX transforms and AD classification

Constants may enter JAX expressions as static values. They are not
differentiated source parameters.

## Failure behavior

No runtime failure state is hidden. Version and rounding distinctions are part
of the documented value contract.

## Contract and evidence links

See [](../../30-representations/units-quantities/constants-and-conventions.md),
the [source-backed cards](../research-infrastructure/source-provenance/constants.md),
and [](../research-infrastructure/contracts.md).

## Canonical import example

```python
from jaxstro.constants import G_CGS
```
