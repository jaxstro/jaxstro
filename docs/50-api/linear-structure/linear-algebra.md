---
title: Linear algebra
---

# Linear algebra

## Owner import path

`jaxstro.numerics.linear_algebra`

## Purpose

Small dense vector, solve, covariance, correlation, and conditioning helpers.

## Public records and callables

`norm2`, `project_onto`, `condition_number`, `weighted_lstsq`, `qr_solve`,
`svd_solve`, `covariance_matrix`, `correlation_from_covariance`,
`correlation_matrix`, `is_positive_definite`, `add_diagonal_jitter`, and
`positive_definite_jitter`.

## Shape and dtype expectations

Vectors and matrices are floating arrays with explicit trailing vector or
matrix axes. Solves require compatible row, column, weight, and right-hand-side
dimensions.

## JAX transforms and AD classification

Dense algebra composes with JIT and smooth-pathwise AD where rank and selected
singular subspaces remain stable. Diagnostics are nonsmooth at rank changes and
repeated singular values.

## Failure behavior

Concrete invalid weights, covariance shapes, non-finite values, and negative
variances raise. Exact rank deficiency makes `condition_number` return positive
infinity, never NaN.

## Contract and evidence links

See [](../../20-methods/linear-structure/linear-algebra.md) and
[](../../60-validation/validation.md).

## Canonical import example

```python
from jaxstro.numerics.linear_algebra import condition_number
```
