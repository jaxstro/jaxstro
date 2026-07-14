---
title: Universal-variable Kepler propagation
---

# Universal-variable Kepler propagation

## Owner import path

`jaxstro.numerics.kepler`

## Purpose

Propagate one relative Cartesian two-body state with one nondimensional
universal-variable solve across elliptic, parabolic-limit, and hyperbolic conics.

## Public records and callables

`universal_kepler_step(...)` returns `UniversalKeplerResult`. The exhaustive
`KEPLER_STATUS_CONVERGED`, `KEPLER_STATUS_INVALID_INPUT`,
`KEPLER_STATUS_NONFINITE_ITERATION`, `KEPLER_STATUS_SINGULAR_RADIUS`, and
`KEPLER_STATUS_MAX_STEPS` identifiers.

## Shape and dtype expectations

Position and velocity are length-three floating arrays. `mu` and `dt` are
scalars; their units must be consistent with the Cartesian state.

## JAX transforms and AD classification

`jit` and `vmap` are supported. JVP and VJP evidence is smooth-pathwise only
while shape, iteration budget, status path, Stumpff branch, and conic regime stay
fixed. This is not an implicit-root derivative.

## Failure behavior

Invalid inputs, non-finite iteration, singular radius, and exhausted steps have
typed statuses. A failed result retains the original Cartesian state.

## Contract and evidence links

See the generated [](../research-infrastructure/contracts.md) entry and the
owner tests in `tests/unit/test_kepler.py` and
`tests/validation/test_kepler_gradients.py`.

## Canonical import example

```python
from jaxstro.numerics.kepler import universal_kepler_step
```
