---
title: Fixed-node quadrature
---

# Fixed-node quadrature

## Owner import path

`jaxstro.numerics.quadrature`

## Purpose

Deterministic fixed-node factories and Hermite expansion helpers.

## Public records and callables

`gauss_legendre_nodes`, `gauss_hermite_nodes`, `gauss_laguerre_nodes`,
`clenshaw_curtis_nodes`, `hermite_e_basis`, and `hermite_coefficients`.

## Shape and dtype expectations

Node count and expansion order are concrete integers. Returned nodes and
weights are one-dimensional floating arrays.

## JAX transforms and AD classification

Nodes and weights are generated host-side as setup constants. Gradients flow
through integrand values, not through rule construction or node count.

## Failure behavior

Invalid rule orders raise. The factory does not adapt order or supply an error
estimate.

## Contract and evidence links

See [](../../20-methods/approximation-integration/quadrature.md) and
[](../../60-validation/validation.md).

## Canonical import example

```python
from jaxstro.numerics.quadrature import gauss_hermite_nodes
```
