---
title: Sampling and resampling
description: >-
  Shape-stable sampling and resampling decisions with explicit statistical,
  execution, and differentiation boundaries.
---

Sampling turns a probability law into draws. Resampling turns a weighted
discrete population into integer ancestor indices. Both operations depend on
random state, but neither is merely key management: the algorithm determines
the statistical contract and the kind of discrete decision returned.

## Scope: sampling and resampling

This page owns the statistical and computational meaning of discrete sampling
and resampling helpers. Explicit PRNG stream construction is documented
separately in [](./random.md), so the key-ownership narrative is not
duplicated here.

The current `jaxstro.numerics.random` surface provides systematic, stratified,
and residual resampling. These standard particle-filter families do not have
interchangeable statistical guarantees. {cite:t}`DoucCappeMoulines2005`
compare them and give a counterexample to a blanket variance-improvement claim
for systematic resampling.

## Resampling a weighted population

```python
import jax.numpy as jnp
import jax.random as jrandom

from jaxstro.numerics.random import (
    residual_resample,
    stratified_resample,
    systematic_resample,
)

seed = 17
weights = jnp.array([0.4, 0.4, 0.2])
num_samples = 5
systematic = systematic_resample(
    jrandom.PRNGKey(seed), weights, num_samples=num_samples
)
stratified = stratified_resample(
    jrandom.PRNGKey(seed + 1), weights, num_samples=num_samples
)
residual = residual_resample(
    jrandom.PRNGKey(seed + 2), weights, num_samples=num_samples
)
replay = systematic_resample(
    jrandom.PRNGKey(seed), weights, num_samples=num_samples
)

assert jnp.array_equal(replay, systematic)
assert jnp.array_equal(jnp.bincount(residual, length=3), jnp.array([2, 2, 1]))
```

Weights are normalized internally. A nonempty all-zero vector uses a documented
zero-total fallback to a uniform distribution rather than producing `NaN`.
This fallback does not apply to negative or non-finite values: concrete eager
inputs must be one-dimensional, nonempty, finite, and nonnegative.

`num_samples` is static under JIT. Returned indices always have shape
`(num_samples,)` when supplied, otherwise `(len(weights),)`. They are integer
indices into the input weight vector, not differentiable particle values.

## Execution and differentiation boundaries

```{list-table} Sampling and resampling contracts
:header-rows: 1
:label: tbl-sampling-resampling-contracts

* - Surface
  - Contract
  - JAX boundary
  - Gradient class
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

The classification matters: a downstream loss must not treat returned integer
indices as a smooth pathwise sample. A model that needs gradients through a
sampling or resampling decision requires an explicitly chosen estimator or
relaxation outside this API. Jaxstro does not invent one.

## Validation

Unit tests check deterministic replay, eager rejection of invalid weights, the
all-zero fallback, exact residual integer counts, output bounds, and JIT
compatibility with static sample counts. The example above is executed by the
documentation test.

For signatures, see [](../../40-api/index.md#jaxstro-numerics-random). For the
assertion-bearing evidence map, see [](../../60-validation/index.md). The
package's differentiation labels, including `validation_only`, are defined in
[](../methods.md#gradient-contracts).
