---
title: Random computation
description: >-
  Explicit PRNG key ownership, deterministic stream construction, and seed
  metadata for reproducible JAX computations.
---

Randomness is part of a numerical method's state, even when it is not stored in
a mutable object. JAX makes that state explicit through splittable keys: a draw
uses the key value it receives, but it does not mutate that value or advance a
hidden global generator. This functional splitting model supports reproducible,
parallelizable execution {cite:t}`JAXJEP263`.

## Scope: random computation

This page owns explicit PRNG key management and deterministic random-stream
construction. It does not own the statistical meaning of a sampling algorithm
or the discrete choice made by a resampler. Those decisions belong on
[](./sampling.md).

## Key streams

`key_stream(key, num)` returns one key for the caller's future work and `num`
keys for the current operation. In the example below, the caller owns
`next_key`; each current operation receives a distinct subkey.

```python
import jax.random as jrandom

from jaxstro.numerics.random import fold_in_stream, key_stream, seed_manifest

seed = 17
key = jrandom.PRNGKey(seed)
next_key, subkeys = key_stream(key, 3)
folded = fold_in_stream(key, 3, start=100)
manifest = seed_manifest(seed, stream="particle-filter")

# Recreating and splitting the same seed reproduces the same keys.
replay_next_key, replay_subkeys = key_stream(jrandom.PRNGKey(seed), 3)

assert (replay_next_key == next_key).all()
assert (replay_subkeys == subkeys).all()
```

`fold_in_stream(key, num, start=...)` derives a deterministic stream by folding
in consecutive integer indices. It is useful when an integer identity such as a
replica, object, or step number owns the stream. It is not an automatic key-reuse
detector.

`seed_manifest(seed, stream=...)` returns a small deterministic dictionary for
logs and provenance records. It is metadata, not a random generator, and it does
not capture the full software or hardware environment.

## Execution and differentiation boundaries

```{list-table} Random-computation contracts
:header-rows: 1
:label: tbl-random-computation-contracts

* - Surface
  - Contract
  - JAX boundary
  - Gradient class
* - `key_stream`
  - The caller owns `next_key`; current operations receive distinct subkeys.
  - `num` is static, and keys remain explicit array values.
  - `validation_only`: key splitting is discrete state construction.
* - `fold_in_stream`
  - Consecutive integer identities are folded into one parent key.
  - `num` and `start` are static.
  - `validation_only`: integer identities and keys have no pathwise derivative.
* - `seed_manifest`
  - The integer seed and stream label are recorded deterministically.
  - The result is host metadata rather than traced random state.
  - `validation_only`: provenance metadata is not a differentiable quantity.
```

Explicit keys make ownership and replay testable. They do not prove that a
sampling scheme is unbiased, that Monte Carlo error is small, or that a sampled
integer decision has a pathwise derivative.

## Validation

Unit tests check deterministic stream construction, fold-in identities, seed
manifests, fixed output shapes, and JIT compatibility where the static-count
contract applies. The example above is executed by the documentation tests.

For signatures, see [](../../50-api/randomness/random.md). For the
assertion-bearing evidence map, see [](../../60-validation/validation.md). The
package's differentiation labels, including `validation_only`, are defined in
[](../methods.md#gradient-contracts).
