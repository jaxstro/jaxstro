---
title: Fixed-step differential equations
---

# Fixed-step differential equations

## Owner import path

`jaxstro.numerics.ode`

## Purpose

Fixed-step first-order integrators and velocity Verlet for separable
second-order systems.

## Public records and callables

`ODEResult`, `VerletResult`, `euler_step`, `midpoint_step`, `rk4_step`, `euler`,
`midpoint`, `rk4`, `solve_fixed_step`, and `velocity_verlet`.

## Shape and dtype expectations

State arrays keep a fixed shape across every step. Time and step values must be
floating and compatible with the callback's state dtype.

## JAX transforms and AD classification

Fixed-length scans compose with `jit`, `vmap`, and AD along a fixed executed
route. This module does not provide adaptive-step or event-time derivatives.

## Failure behavior

Callback, shape, and dtype failures propagate. There is no hidden adaptive
retry, stiffness detection, or scientific acceptance policy.

## Contract and evidence links

See [](../../20-methods/change-constraints-evolution/ode.md) and
[](../../60-validation/validation.md).

## Canonical import example

```python
from jaxstro.numerics.ode import solve_fixed_step
```
