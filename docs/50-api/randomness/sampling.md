---
title: Sampling primitives
---

# Sampling primitives

## Owner import path

`jaxstro.numerics.sampling`

## Purpose

Differentiable inverse-CDF draws from tabulated weights and stratified uniform
deviates with caller-owned random keys.

## Public records and callables

`inverse_cdf_draw(weight, grid, unif, reg=1e-30)` integrates an unnormalized
tabulated weight and inverts its cumulative distribution at a supplied uniform
deviate. `stratified_uniform(key, n, *, minval=0.0, maxval=1.0)` draws one
uniform deviate from each of `n` equal-width strata.

## Shape and dtype expectations

For `inverse_cdf_draw`, `weight` and `grid` are one-dimensional floating arrays
with matching shape `(n,)`, `grid` is uniformly spaced, and `unif` is a scalar
floating deviate in `[0, 1]`. The result is a floating scalar. The integration
spacing is `grid[1] - grid[0]`, so the input needs at least two grid points.

For `stratified_uniform`, `key` is an explicit JAX key array, `n` is a positive
static integer, and the result has shape `(n,)`. `minval` and `maxval` are scalar
bounds.

## JAX transforms and AD classification

Both functions use fixed output shapes and compose with JIT and VMAP.
`inverse_cdf_draw` is differentiable with respect to `weight` and `unif` away
from interpolation knots; it consumes no random key. `stratified_uniform`
consumes a discrete PRNG key, so the caller owns the key and no derivative is
defined through it. Its `n` argument is static under JIT.

## Failure behavior

`inverse_cdf_draw` builds a cumulative trapezoid and normalizes it by
`cdf[-1] + reg`. At zero-total weight, the guard produces an all-zero CDF and
interpolation clamps the supported draw to `grid[-1]`; callers that consider a
zero total invalid must enforce that precondition. Shape mismatches, a grid
shorter than two points, nonuniform spacing, and out-of-contract values are not
eagerly validated.

`stratified_uniform` raises `ValueError` when static `n < 1`. Invalid key or
bound inputs propagate from JAX. Neither function assesses Monte Carlo
convergence.

## Contract and evidence links

See [](../../20-methods/probability-sampling/sampling.md) and
[](../../40-workflows/reproducible-research/random-state-ownership.md).

## Canonical import example

```python
from jaxstro.numerics.sampling import inverse_cdf_draw
```
