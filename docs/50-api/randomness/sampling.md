---
title: Sampling primitives
---

# Sampling primitives

## Owner import path

`jaxstro.numerics.sampling`

## Purpose

Inverse-CDF draws and stratified uniforms with caller-owned random keys.

## Public records and callables

`inverse_cdf_draw` and `stratified_uniform`.

## Shape and dtype expectations

Keys are explicit JAX key arrays. Sample shapes and stratum counts are static;
PPF callbacks must accept the generated floating quantiles.

## JAX transforms and AD classification

Fixed-shape generation composes with JIT. Derivatives may flow through a smooth
PPF's parameters, not through the discrete key.

## Failure behavior

Callback and shape failures propagate. This module does not normalize a target
density or assess Monte Carlo convergence.

## Contract and evidence links

See [](../../20-methods/probability-sampling/sampling.md) and
[](../../40-workflows/reproducible-research/random-state-ownership.md).

## Canonical import example

```python
from jaxstro.numerics.sampling import inverse_cdf_draw
```
