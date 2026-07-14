---
title: One-dimensional interpolation
---

# One-dimensional interpolation

## Owner import path

`jaxstro.numerics.interpolation`

## Purpose

Linear, Hermite, monotone cubic, and natural cubic interpolation with explicit
table and boundary behavior.

## Public records and callables

`interp1d(...)` is the clamped linear baseline. `cubic_hermite_interp(...)`
evaluates supplied node derivatives; `pchip_slopes(...)` constructs
shape-preserving slopes; and `monotone_cubic_interp(...)` combines those slopes
with the Hermite evaluator. `natural_cubic_spline_coeffs`, `eval_cubic_spline`,
`TabulatedFunction1D`, `MonotoneTabulatedFunction1D`, and
`NaturalCubicSpline1D` provide prepared table surfaces.

## Shape and dtype expectations

Coordinates are one-dimensional, strictly increasing floating arrays. Values
share the tabulated axis length and may carry payload dimensions.

## JAX transforms and AD classification

Evaluation is smooth-pathwise inside branch-stable intervals. Knots, plateaus,
sign changes, and clamped boundaries are nonsmooth derivative boundaries.

## Failure behavior

Concrete invalid tables raise. Out-of-domain linear queries clamp to endpoints;
method-specific traced validation limits remain explicit.

## Contract and evidence links

See [](../../20-methods/approximation-integration/interpolation.md), the
generated [](../research-infrastructure/contracts.md), and
[](../../40-workflows/investigations/interpolation-boundary-policies.md).

## Canonical import example

```python
from jaxstro.numerics.interpolation import monotone_cubic_interp
```
