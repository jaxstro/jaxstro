---
title: Stable statistics helpers
---

# Stable statistics helpers

## Owner import path

`jaxstro.numerics.stats`

## Purpose

Numerically stable elementary transforms, Gaussian summaries, relative errors,
and convergence checks.

## Public records and callables

`safe_log`, `safe_exp`, `safe_div`, `logsumexp`, `gaussian_logpdf`,
`gaussian_loglikelihood`, `stable_log1p`, `stable_expm1`, `relative_error`, and
`check_convergence`.

## Shape and dtype expectations

Inputs are broadcast-compatible floating arrays. Reductions use explicit axes
where applicable.

## JAX transforms and AD classification

Array kernels compose with JIT, VMAP, and AD. Floors, ceilings, and denominator
guards create explicit piecewise derivative boundaries.

## Failure behavior

Safe helpers return their documented floored, capped, or guarded values; they do
not turn those numerical policies into scientific validity claims.

## Contract and evidence links

See [](../../20-methods/probability-sampling/distributions.md) and
[](../../60-validation/validation.md).

## Canonical import example

```python
from jaxstro.numerics.stats import logsumexp
```
