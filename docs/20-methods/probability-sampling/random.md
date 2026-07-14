---
title: Random computation
description: >-
  Explicit PRNG key ownership, deterministic stream construction, and bounded
  reproducibility claims.
---

## The question this method answers

How can a researcher make random state explicit, replayable, and safe to divide
among concurrent computations? JAX uses pseudo-random number generator (PRNG)
keys as immutable array values rather than a hidden mutable global generator
{cite:t}`JAXJEP263`.

:::{important}
A key is numerical state. Using a key does not mutate it, so the caller must
return, store, or derive the key for every later operation.
:::

## Before computation: what should be true?

Choose one owner for each parent key and a deterministic rule for assigning
subkeys to replicas, objects, steps, or named operations. Record the integer
seed and stream meaning. Never pass the same key to two draws merely because the
calls occur in different functions.

:::{warning}
Deterministic replay establishes that the same inputs reproduce the same key
stream. It does not establish statistical independence, unbiasedness, or a
sufficient effective sample size.
:::

## Define the mathematical objects

A PRNG maps a finite key and counter-like state to deterministic bits that are
designed to behave like random draws under specified tests. A JAX key is an
array token for this state. `split` derives multiple child keys; `fold_in`
combines a key with an integer identity. Neither operation consumes or mutates
the original key value.

A seed manifest is host metadata describing how a root key was initialized. It
is not the key itself and does not capture the full software, device, or
floating-point environment.

## Derive the method

Jaxstro's stream helper reserves one child for the caller's future work and
returns the remaining children for the current operation:

```{math}
:label: eq-key-split
(K_{\mathrm{next}},K_1,\ldots,K_m)=\operatorname{split}(K,m+1).
```

When an integer identity owns a stream, folding that identity into a shared
parent gives

```{math}
:label: eq-key-fold-in
K_i=\operatorname{fold\_in}(K,s+i),
\qquad i=0,\ldots,m-1,
```

where $s$ is the requested starting index. The mapping is deterministic, so an
identity-to-key rule can be reconstructed without depending on loop order.

## What the algorithm actually does

`key_stream(key, num)` calls `jax.random.split(key, num + 1)` and returns a key
of shape `(2,)` plus subkeys of shape `(num, 2)` for legacy `PRNGKey` inputs.
`fold_in_stream(key, num, start=0)` vmaps `jax.random.fold_in` over consecutive
integer indices. `num` and `start` are static, so changing them changes output
shape or traced program structure and can recompile.

`seed_manifest(seed, stream="default", algorithm="jax.random")` returns a host
dictionary. It is not JIT-oriented random state and performs no key-reuse audit.

## What JAX differentiates

Keys and folded integer identities are discrete values. Splitting, folding, and
seed metadata are `validation_only` operations with no pathwise derivative.
Downstream transformations can differentiate a smooth function of a random draw
with respect to floating parameters when an appropriate pathwise construction
exists, but that is a property of the sampling algorithm, not of key splitting.

```{list-table} Random-computation contracts
:header-rows: 1
:label: tbl-random-computation-contracts

* - Surface
  - Execution contract
  - Gradient class
* - `key_stream`
  - The caller owns `next_key`; `num` is static.
  - `validation_only`
* - `fold_in_stream`
  - Integer identities and static `num` and `start` determine the stream.
  - `validation_only`
* - `seed_manifest`
  - Returns host metadata rather than traced random state.
  - `validation_only`
```

## Using it in Jaxstro

```python
import jax.random as jrandom

from jaxstro.numerics.random import fold_in_stream, key_stream, seed_manifest

seed = 17
key = jrandom.PRNGKey(seed)
next_key, subkeys = key_stream(key, 3)
folded = fold_in_stream(key, 3, start=100)
manifest = seed_manifest(seed, stream="particle-filter")

replay_next_key, replay_subkeys = key_stream(jrandom.PRNGKey(seed), 3)

assert (replay_next_key == next_key).all()
assert (replay_subkeys == subkeys).all()
assert not (subkeys[0] == subkeys[1]).all()
assert manifest["seed"] == seed
```

## How to audit the result

1. Draw a key-ownership tree before running concurrent or nested random work.
2. Recreate the root key and compare every derived key exactly.
3. Check output shapes and verify sibling subkeys differ on the fixture.
4. Run split and fold-in helpers under JIT with the documented static counts.
5. Record seed, stream names, identity ranges, package version, and environment.
6. Audit statistical properties separately with a method-appropriate test.

:::{tip}
Fold stable object or replica identifiers into a parent key when execution order
may change. Split sequentially when the algorithm itself owns an ordered stream.
Document which rule you chose.
:::

## Where the claim stops

Explicit key ownership prevents hidden generator mutation and enables exact
replay of key construction. It does not detect every accidental key reuse,
guarantee independence, validate a resampler, or quantify Monte Carlo error.
Seed metadata alone is not a complete provenance record.

## Connected ideas

:::{seealso}
Review probability state in
[](../../10-foundations/mathematical-objects/probability-and-distributions.md),
connect keys to explicit scientific state in
[](../../30-representations/parameters-state/pytrees-as-scientific-state.md), and
follow [](../../40-workflows/reproducible-research/random-state-ownership.md).
Signatures are in [](../../50-api/randomness/random.md), and evidence routes
through [](../../60-validation/validation.md). The gradient taxonomy is in
[](../methods.md#gradient-contracts).
:::
