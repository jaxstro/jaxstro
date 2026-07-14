---
title: Linear operators
---

# Linear operators

## Owner import path

`jaxstro.numerics.operators`

## Purpose

Small PyTree linear operators and explicit algebraic composition.

## Public records and callables

`LinearOperator`, `DenseOperator`, `DiagonalOperator`, `ScaledOperator`,
`SumOperator`, `ProductOperator`, `TransposeOperator`, `BlockDiagonalOperator`,
`scale`, `add`, `compose`, `transpose`, and `block_diag`.

## Shape and dtype expectations

Operators declare matrix shapes; `matvec` and `rmatvec` require compatible
trailing vector axes. Component dtypes determine result dtype through JAX.

## JAX transforms and AD classification

Operator PyTrees compose with JIT, VMAP, and AD when their leaves and shapes are
fixed. Composition order is explicit.

## Failure behavior

Incompatible shapes raise during construction or evaluation. The module does
not infer sparsity, conditioning, or solver policy.

## Contract and evidence links

See [](../../20-methods/linear-structure/operators.md) and
[](../../60-validation/validation.md).

## Canonical import example

```python
from jaxstro.numerics.operators import DenseOperator
```
