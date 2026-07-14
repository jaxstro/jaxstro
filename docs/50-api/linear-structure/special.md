---
title: Special functions and bases
---

# Special functions and bases

## Owner import path

`jaxstro.numerics.special`

## Purpose

Stable Planck-law kernels, log-weight normalization, and polynomial bases.

## Public records and callables

`planck_lambda_cgs`, `log_planck_lambda_cgs`, `planck_nu_cgs`,
`log_planck_nu_cgs`, `log_normalize`, `normalize_log_weights`,
`legendre_basis`, `chebyshev_t_basis`, and `laguerre_basis`.

## Shape and dtype expectations

Planck inputs are floating arrays in the units named by each function. Basis
order is a concrete integer; evaluation arrays may be batched.

## JAX transforms and AD classification

Array kernels compose with JIT, VMAP, and smooth-pathwise AD on their positive,
finite domains. Basis order is static.

## Failure behavior

Concrete nonpositive physical inputs raise where required. The functions do not
invent finite values outside their mathematical domains.

## Contract and evidence links

See [](../../20-methods/linear-structure/special-functions.md) and
[](../../60-validation/validation.md).

## Canonical import example

```python
from jaxstro.numerics.special import log_planck_lambda_cgs
```
