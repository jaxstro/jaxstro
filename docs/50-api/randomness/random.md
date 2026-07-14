---
title: Random streams and resampling
---

# Random streams and resampling

## Owner import path

`jaxstro.numerics.random`

## Purpose

Explicit key streams, reproducibility manifests, and discrete particle
resampling.

## Public records and callables

`KeyArray`, `key_stream`, `fold_in_stream`, and `seed_manifest` own PRNG-stream
bookkeeping. `systematic_resample`, `stratified_resample`, and
`residual_resample` own discrete resampling.

## Shape and dtype expectations

Keys are JAX key arrays. Resampling weights are a finite one-dimensional
floating array; returned ancestor indices are integer arrays of fixed size.

## JAX transforms and AD classification

Stream and resampling mechanics compose with JIT where sizes are static.
Resampling choices are discrete and make no AD claim. Value-dependent finite
and nonnegative checks are skipped under tracing, so transformed callers own
those preconditions.

## Failure behavior

Concrete eager inputs reject non-one-dimensional or empty weights, nonpositive
sample counts, non-finite values, and negative values. All-zero weights are
valid: normalization deliberately falls back to a uniform distribution. No
hidden key or seed is created.

## Contract and evidence links

See [](../../20-methods/probability-sampling/random.md),
[](../../20-methods/probability-sampling/sampling.md), and
[](../../40-workflows/reproducible-research/random-state-ownership.md).

## Canonical import example

```python
from jaxstro.numerics.random import systematic_resample
```
