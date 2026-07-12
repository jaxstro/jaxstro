---
title: Random streams and resampling
description: >-
  Explicit PRNG key streams, deterministic seed manifests, and shape-stable
  resampling helpers for scientific workflows.
---

Randomness is part of a numerical method's state, even when it is not stored in
a mutable object. JAX makes that state explicit through splittable keys: a draw
uses the key value it receives, but it does not mutate that value or advance
a hidden global generator. This functional splitting model supports reproducible,
parallelizable execution {cite:t}`JAXJEP263`.

`jaxstro.numerics.random` packages that ownership discipline with three
shape-stable resampling schemes. The schemes are standard particle-filter
families, but they do not have interchangeable statistical guarantees:
{cite:t}`DoucCappeMoulines2005` compare residual, stratified, and systematic
resampling and give a counterexample to a blanket variance-improvement claim for
systematic resampling.

## Key streams

`key_stream(key, num)` returns one key for the caller's future work and `num`
keys for the current operation. In the example below, the caller owns `next_key`;
each resampler receives a distinct subkey.

```python
import jax.numpy as jnp
import jax.random as jrandom

from jaxstro.numerics.random import (
    fold_in_stream,
    key_stream,
    residual_resample,
    seed_manifest,
    stratified_resample,
    systematic_resample,
)

seed = 17
key = jrandom.PRNGKey(seed)
next_key, subkeys = key_stream(key, 3)
folded = fold_in_stream(key, 3, start=100)
manifest = seed_manifest(seed, stream="particle-filter")

# Five draws make these normalized weights exact integer residual counts.
weights = jnp.array([0.4, 0.4, 0.2])
num_samples = 5
systematic = systematic_resample(subkeys[0], weights, num_samples=num_samples)
stratified = stratified_resample(subkeys[1], weights, num_samples=num_samples)
residual = residual_resample(subkeys[2], weights, num_samples=num_samples)

# Recreating and splitting the same seed reproduces the same systematic draw.
_, replay_subkeys = key_stream(jrandom.PRNGKey(seed), 3)
replay = systematic_resample(
    replay_subkeys[0], weights, num_samples=num_samples
)

assert jnp.array_equal(replay, systematic)
assert jnp.array_equal(jnp.bincount(residual, length=3), jnp.array([2, 2, 1]))
```

`fold_in_stream(key, num, start=...)` derives a deterministic stream by folding
in consecutive integer indices. It is useful when an integer identity such as a
replica, object, or step number owns the stream. It is not an automatic key-reuse
detector.

`seed_manifest(seed, stream=...)` returns a tiny deterministic dictionary for
logs and provenance records. It is metadata, not a random generator, and it does
not capture the full software or hardware environment.

## Resampling

The module provides three shape-stable resampling helpers:

- `systematic_resample(key, weights, num_samples=...)`
- `stratified_resample(key, weights, num_samples=...)`
- `residual_resample(key, weights, num_samples=...)`

Weights are normalized internally. A nonempty all-zero vector uses a documented
zero-total fallback to a uniform distribution rather than producing `NaN`. This
fallback does not apply to negative or non-finite values: concrete eager inputs
must be one-dimensional, nonempty, finite, and nonnegative.

`num_samples` is static under JIT. Returned indices always have shape
`(num_samples,)` when supplied, otherwise `(len(weights),)`. They are integer
indices into the input weight vector, not differentiable particle values.

## Execution and differentiation boundaries

```{list-table} Randomness and resampling contracts
:header-rows: 1
:label: tbl-random-contracts

* - Surface
  - Contract
  - JAX boundary
  - Gradient class
* - `key_stream`
  - Returns a caller-owned `next_key` and the requested number of subkeys.
  - `num` is static; keys remain explicit array values.
  - `validation_only`: key splitting is discrete state construction.
* - `fold_in_stream`
  - Folds consecutive integer identities into one parent key.
  - `num` and `start` are static.
  - `validation_only`: integer identities and keys have no pathwise derivative.
* - Systematic and stratified resampling
  - Return shape-stable integer indices from normalized weights.
  - `num_samples` is static; valid inputs work under `jax.jit`.
  - `validation_only`: inverse-CDF search and sampled indices are discrete.
* - Residual resampling
  - Emits deterministic floor counts followed by a systematic residual tail.
  - Exact integer counts can leave an empty random tail without changing shape.
  - `validation_only`: floor, search, and integer selection are discrete.
* - Input validation
  - Eager calls reject empty, non-1D, negative, or non-finite weights and
    nonpositive sample counts; nonempty all-zero weights use the uniform
    zero-total fallback.
  - Value-dependent eager validation is skipped while weights are traced; the
    caller must supply finite, nonnegative weights under `jax.jit`.
  - `validation_only`: this is a domain guard, not a differentiable operation.
```

The classification matters: a downstream loss must not treat the returned
indices as a smooth pathwise sample. If a model needs gradients through a
resampling decision, it needs an explicitly chosen estimator or relaxation
outside this API; jaxstro does not invent one.

## Validation

Unit tests check deterministic key behavior, seed manifests, eager rejection of
invalid weights, the all-zero fallback, exact residual integer counts, and JIT
compatibility with static sample counts. The example above is executed verbatim
by the documentation test.

For signatures, see [](../40-api/index.md#jaxstro-numerics-random). For the
assertion-bearing evidence map, see [](../60-validation/index.md). The package's
five differentiation labels, including `validation_only`, are defined in
[](./index.md#gradient-contracts).
