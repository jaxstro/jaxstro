---
title: Compensated arithmetic
---

# Compensated arithmetic

## Owner import path

`jaxstro.numerics.compensated`

## Purpose

Neumaier-style compensated scalar, array, vector, and dot-product reductions.

## Public records and callables

`neumaier_add`, `compensated_sum`, `compensated_sum_array`,
`compensated_vector_sum`, and `compensated_dot`.

## Shape and dtype expectations

Inputs are floating arrays. Array and vector helpers use a leading reduction
axis; dot products require matching one-dimensional shapes.

## JAX transforms and AD classification

Fixed-order reductions compose with JIT and AD. Compensation improves rounding
behavior but does not make floating arithmetic exact or order-independent.

## Failure behavior

Shape and dtype errors propagate. Non-finite inputs remain non-finite; no values
are silently replaced.

## Contract and evidence links

See [](../../20-methods/linear-structure/linear-algebra.md) and the numerical
tests linked from [](../research-infrastructure/testing.md).

## Canonical import example

```python
from jaxstro.numerics.compensated import compensated_sum_array
```
