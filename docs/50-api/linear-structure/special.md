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
`legendre_basis`, `chebyshev_t_basis`, `laguerre_basis`,
`riccati_bessel_basis`, `riccati_bessel_at_order`, `riccati_seed_order`,
and `riccati_wronskian_residual`.

## Shape and dtype expectations

Planck inputs are floating arrays in the units named by each function. Basis
order is a concrete integer; evaluation arrays may be batched.

`riccati_bessel_basis` returns a `(S, C)` pair, each with a leading axis over
`l = 0 .. degree`. Its argument must be positive: `C_l` diverges at the origin.
Its `seed_order` must clear both `degree` and the largest argument in use; the
default clears `degree` only and is valid only for `x < degree`.
`riccati_seed_order(degree, x_max)` computes a sufficient value.

`riccati_bessel_at_order` returns the same pair at a single order without
materializing the lower ones, for callers sweeping one order over many
arguments.

## JAX transforms and AD classification

Array kernels compose with JIT, VMAP, and smooth-pathwise AD on their positive,
finite domains. Basis order is static.

## Failure behavior

Concrete nonpositive physical inputs raise where required. The functions do not
invent finite values outside their mathematical domains.

A `seed_order` too low for the argument does **not** raise. It returns finite,
smooth, wrong values; `riccati_wronskian_residual` is the gate that detects it.

## Contract and evidence links

See [](../../20-methods/linear-structure/special-functions.md) and
[](../../60-validation/validation.md).

## Canonical import example

```python
from jaxstro.numerics.special import log_planck_lambda_cgs
```
