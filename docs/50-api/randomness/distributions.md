---
title: Probability distributions
---

# Probability distributions

## Owner import path

`jaxstro.numerics.distributions`

## Purpose

Log-density, CDF, and inverse-CDF kernels for normal, lognormal, finite
power-law, and truncated-normal families.

## Public records and callables

`normal_logpdf`, `normal_cdf`, `normal_ppf`, `lognormal_logpdf`,
`lognormal_cdf`, `lognormal_ppf`, `powerlaw_logpdf`, `powerlaw_cdf`,
`powerlaw_ppf`, `truncated_normal_logpdf`, `truncated_normal_cdf`, and
`truncated_normal_ppf`.

## Shape and dtype expectations

Parameters and evaluation points are broadcast-compatible floating arrays.
Inverse CDF inputs lie on the documented unit-interval domain.

## JAX transforms and AD classification

Kernels compose with JIT, VMAP, and smooth-pathwise AD on regular domains. The
finite power-law family uses a smooth removable-singularity formulation through
`alpha=-1`.

## Failure behavior

Support policy is explicit: log densities may return negative infinity, CDFs
may saturate, and invalid quantile inputs are outside the contract.

## Contract and evidence links

See [](../../20-methods/probability-sampling/distributions.md), the generated
[](../research-infrastructure/contracts.md), and
[](../../40-workflows/investigations/powerlaw-removable-limit.md).

## Canonical import example

```python
from jaxstro.numerics.distributions import powerlaw_ppf
```
