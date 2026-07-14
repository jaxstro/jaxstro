---
title: Dimensional quantities
---

# Dimensional quantities

## Owner import path

`jaxstro.quantity`

## Purpose

Implemented dimensional quantity evaluation, unit parsing and registries,
conversion, serialization, constants, and equivalencies.

## Public records and callables

`Quantity`, `Unit`, `Dimension`, `UnitRegistry`, their typed errors, concrete
units and dimensions, `get_unit`, `format_unit`, `parse_unit`, serialization
helpers, and the public `constants`, `equivalencies`, `astro`, `bases`, and
`units` modules.

## Shape and dtype expectations

Quantity values are JAX-compatible arrays. Unit and dimension objects are
static metadata at traced boundaries.

## JAX transforms and AD classification

Value conversion composes with JAX transforms when units are fixed. Parsing,
registry mutation, and dimensional validation are host-side.

## Failure behavior

Dimension, conversion, equivalency, parsing, and registry failures use explicit
typed exceptions. Ecosystem adoption remains deferred; this page is not a
cutover announcement.

## Contract and evidence links

See [](../../30-representations/units-quantities/quantity-system.md),
[](../../30-representations/units-quantities/quantities.md), and
[](../../70-project/decisions/0006-build-own-quantity-not-unxt.md).

## Canonical import example

```python
from jaxstro.quantity import Quantity
```
