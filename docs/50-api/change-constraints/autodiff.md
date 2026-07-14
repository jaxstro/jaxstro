---
title: Automatic differentiation products
---

# Automatic differentiation products

## Owner import path

`jaxstro.numerics.autodiff`

## Purpose

Named wrappers expose JVP, VJP, Hessian-vector, Gauss-Newton, and empirical
Fisher products without owning a scientific derivative-validity policy.

## Public records and callables

`jvp`, `vjp`, `jacobian_vector_product`, `vector_jacobian_product`, `hvp`,
`gauss_newton_product`, and `empirical_fisher_product`.

## Shape and dtype expectations

Inputs and tangents must follow the wrapped function's PyTree shapes. Products
preserve JAX array dtypes; mixed or integer differentiation follows JAX rules.

## JAX transforms and AD classification

These helpers compose JAX transformations. Their derivatives are only as
scientifically meaningful as the selected function and branch.

## Failure behavior

JAX tracing, shape, or dtype errors propagate. The module does not replace
non-finite products or certify derivative meaning.

## Contract and evidence links

See [](../../20-methods/change-constraints-evolution/autodiff.md) and
[](../../60-validation/validation.md).

## Canonical import example

```python
from jaxstro.numerics.autodiff import hvp
```
