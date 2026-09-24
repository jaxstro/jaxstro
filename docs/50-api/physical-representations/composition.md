---
title: Chemical composition
---

# Chemical composition

## Owner import path

`jaxstro.composition`

## Purpose

One record of the chemical composition, passed to every consumer that needs it:
equation-of-state tables, opacities and protostellar models. It exists so that a
composition is chosen once, explicitly, and read by all of them, rather than each
consumer carrying its own default.

## Public records and callables

`Composition(X, Y, Z, deuterium_to_hydrogen)`: hydrogen, helium and metal mass
fractions and the deuterium-to-hydrogen number ratio. Every field is required.
`SUM_TOLERANCE` is the allowed `|X + Y + Z - 1|`, float64 roundoff on decimal inputs.

## Shape and dtype expectations

Each field is a scalar: a Python float or a zero-dimensional array. The record is an
Equinox module and a JAX pytree.

## JAX transforms and AD classification

The fields are pytree leaves and can be traced and differentiated, for example a
metallicity under `jax.grad`. Validation runs only on concrete values; a traced
composition is guarded by its consumer's validity check.

## Failure behavior

Construction raises `ValueError` when a mass fraction lies outside `[0, 1]`, when the
fractions do not sum to one within `SUM_TOLERANCE`, or when D/H is negative. Missing
fields raise `TypeError`: there are no defaults.

## Contract and evidence links

Tests: `tests/unit/test_composition.py`. Introduced for hydrax audit finding F13
(seam S1, 2026-09-23).

## Canonical import example

```python
from jaxstro.composition import Composition

solar = Composition(X=0.70, Y=0.28, Z=0.02, deuterium_to_hydrogen=2.5e-5)
```
