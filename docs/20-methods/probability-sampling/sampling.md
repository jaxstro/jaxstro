---
title: Sampling and resampling
description: >-
  Inverse-CDF draws, stratified uniforms, and discrete resampling with explicit
  key, fallback, and differentiation boundaries.
---

## The question this method answers

How can a researcher turn a distribution into a draw, or turn weighted particles
into integer ancestor indices? Sampling maps a random uniform value through a
probability law. Resampling makes a discrete allocation decision from normalized
weights.

:::{important}
A continuous inverse-CDF draw and a discrete ancestor index have different
derivative contracts. Do not treat every random output as pathwise
differentiable merely because the function runs under `jax.jit`.
:::

## Before computation: what should be true?

For tabulated inverse-CDF sampling, provide one-dimensional weights and a
one-dimensional uniformly spaced grid of the same length, with at least two
points and a meaningful positive integral. For resampling, provide a nonempty
one-dimensional finite nonnegative weight vector. Choose and own a distinct key
for each random operation.

:::{warning}
Concrete resampling rejects invalid weights, but eager validation is skipped while weights are traced. The caller must supply finite, nonnegative weights under `jax.jit`. `inverse_cdf_draw` does not validate grid spacing, monotonicity,
weight sign, total weight, or $u\in[0,1]$.
:::

## Define the mathematical objects

An inverse CDF $F^{-1}(u)$ is the quantile at cumulative probability $u$. A
stratum divides $[0,1]$ into one of $n$ equal intervals. A resampler receives
weights $w_i$, normalizes them to probabilities, and returns integer indices
whose multiplicities define the new population.

Systematic resampling uses one random offset shared by evenly spaced positions.
Stratified resampling uses one uniform value in each stratum. Residual resampling
first assigns deterministic floor counts, then samples the remaining slots.

## Derive the method

If $U\sim\operatorname{Uniform}(0,1)$ and $F$ is a continuous CDF, then

```{math}
:label: eq-inverse-cdf-sampling
X=F^{-1}(U),
\qquad
\Pr(X\le x)=\Pr(U\le F(x))=F(x).
```

One uniform $V_i\sim\operatorname{Uniform}(0,1)$ per stratum gives

```{math}
:label: eq-stratified-uniform
U_i=\frac{i+V_i}{n},
\qquad i=0,\ldots,n-1.
```

For normalized weights $\bar{w}_i$ and $N$ requested samples, residual
resampling assigns

```{math}
:label: eq-residual-counts
N_i=\lfloor N\bar{w}_i\rfloor,
\qquad
R=N-\sum_iN_i,
\qquad
q_i=\frac{N\bar{w}_i-N_i}{R}
```

when $R>0$, then draws the remaining $R$ indices from $q$.

## What the algorithm actually does

`inverse_cdf_draw(weight, grid, unif, reg=1e-30)` integrates with the repository's
uniform-grid cumulative trapezoid using `grid[1] - grid[0]`, divides by
`cdf[-1] + reg`, and calls `jnp.interp`. A zero total produces an all-zero CDF
and a finite endpoint fallback rather than `NaN`; that fallback is not a draw
from a normalized law.

`stratified_uniform(key, n, minval=0, maxval=1)` returns shape `(n,)` in stratum
order. `n` is static. It does not permute the result.

Systematic, stratified, and residual resamplers normalize weights internally. A
nonempty all-zero vector uses a documented zero-total fallback to uniform
probabilities. `num_samples` is static and controls output shape; omitting it
returns `len(weights)` integer indices. Each function deterministically uses the
key value it receives but does not return a replacement key. The caller must not
reuse that key for another draw.

The residual core evaluates one random offset even when deterministic floor
counts fill every slot, preserving one fixed traced program and output shape.

## What JAX differentiates

For positive tabulated mass and a fixed active interpolation interval,
`inverse_cdf_draw` has a piecewise pathwise derivative with respect to weights
and `unif`. Cumulative construction is smooth, but CDF knot coincidences,
interval changes in `jnp.interp`, a zero total, and the regularization-dominated
regime are derivative boundaries.

`stratified_uniform` is pathwise differentiable with respect to floating
`minval` and `maxval` for a fixed key and static $n$; keys and $n$ are discrete.
Resampling returns integer indices after cumulative search, floor, and discrete
selection. Those indices are `validation_only`, not pathwise samples. A
downstream differentiable resampler needs an explicit estimator or relaxation
outside this API.

```{list-table} Sampling and resampling contracts
:header-rows: 1
:label: tbl-sampling-resampling-contracts

* - Surface
  - Execution contract
  - Gradient class
* - Tabulated inverse CDF
  - Uniform grid and positive mass are caller-owned.
  - Piecewise pathwise away from knot and fallback boundaries.
* - Stratified uniforms
  - `n` is static and the caller owns the key.
  - Pathwise only for floating interval bounds with fixed key and `n`.
* - Resampling
  - `num_samples` is static; outputs are integer indices.
  - `validation_only`
* - Input validation
  - Concrete invalid weights raise; the zero-total fallback is uniform.
  - `validation_only`
```

## Using it in Jaxstro

```python
import jax.numpy as jnp
import jax.random as jrandom

from jaxstro.numerics.random import (
    residual_resample,
    stratified_resample,
    systematic_resample,
)
from jaxstro.numerics.sampling import inverse_cdf_draw, stratified_uniform

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

grid = jnp.linspace(0.0, 1.0, 5)
draw = inverse_cdf_draw(jnp.ones(5), grid, jnp.array(0.4))
uniforms = stratified_uniform(jrandom.PRNGKey(seed + 3), 4)

assert jnp.array_equal(replay, systematic)
assert jnp.array_equal(jnp.bincount(residual, length=3), jnp.array([2, 2, 1]))
assert jnp.isfinite(draw)
assert uniforms.shape == (4,)
```

## How to audit the result

1. Verify tabulated support, grid uniformity, weight sign, and integrated mass.
2. Check inverse-CDF draws against analytic quantiles on a known distribution.
3. Check each stratified uniform falls in its own interval.
4. Replay each resampler with the same key and verify shape and index bounds.
5. Exercise the zero-total fallback and exact residual-count fixture explicitly.
6. Compare resampling frequencies and variance on repeated independent-key trials;
   do not infer those properties from replay alone {cite:t}`DoucCappeMoulines2005`.

:::{tip}
Audit the key tree, distribution law, and discrete resampling policy as three
separate objects. Passing one audit cannot substitute for the others.
:::

## Where the claim stops

The inverse-CDF helper assumes a uniform grid and does not certify a normalized
density. The zero-total fallback is finite behavior, not statistical validity.
Systematic, stratified, and residual resampling have different variance
properties; reproducibility does not rank them. Integer indices carry no
automatic gradient estimator.

## Connected ideas

:::{seealso}
Review CDFs in
[](../../10-foundations/mathematical-objects/probability-and-distributions.md),
connect samples to uncertainty representation in
[](../../30-representations/uncertainty/ensemble-propagation.md), and establish
key ownership with
[](../../40-workflows/reproducible-research/random-state-ownership.md).
Continuous helper signatures are in [](../../50-api/randomness/sampling.md),
resampler signatures are in [](../../50-api/randomness/random.md), and evidence
belongs in [](../../60-validation/validation.md). The gradient taxonomy is in
[](../methods.md#gradient-contracts).
:::
