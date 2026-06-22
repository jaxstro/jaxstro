---
title: Random streams and resampling
description: >-
  Explicit PRNG key streams, deterministic seed manifests, and shape-stable
  resampling helpers for scientific workflows.
---

`jaxstro.numerics.random` complements the older `rng` helpers with a slightly
higher-level first slice for simulation and particle-filter workflows. The
boundary is still explicit: callers pass and receive keys rather than relying on
hidden global random state.

## Key streams

`key_stream(key, num)` returns a next key plus `num` subkeys. This makes key
ownership visible at call sites:

```python
next_key, subkeys = key_stream(key, 4)
```

`fold_in_stream(key, num, start=...)` derives a deterministic stream by folding
in consecutive integer indices.

`seed_manifest(seed, stream=...)` returns a tiny deterministic dictionary for
logs and provenance records. It is metadata, not a random generator.

## Resampling

The module provides three shape-stable resampling helpers:

- `systematic_resample(key, weights, num_samples=...)`
- `stratified_resample(key, weights, num_samples=...)`
- `residual_resample(key, weights, num_samples=...)`

Weights are normalized internally. If the total weight is zero, helpers fall back
to a uniform distribution rather than emitting `NaN`.

`num_samples` is static under JIT. Returned indices always have shape
`(num_samples,)` when supplied, otherwise `(len(weights),)`.

## Validation

Unit tests check deterministic key behavior, seed manifests, degenerate
resampling distributions, exact residual integer counts, and JIT compatibility
with static sample counts.
