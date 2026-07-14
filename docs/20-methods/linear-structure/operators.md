---
title: Linear operators
description: >-
  Small PyTree linear operators for matrix-free algebra without committing
  jaxstro to a sparse or iterative solver stack.
---

Scientific code often wants to pass around "something matrix-like" without
materializing every algebraic composition as a dense array. `jaxstro` provides a
small LinearOperator protocol for that use case while deliberately avoiding a
large sparse framework or iterative solver stack.

Every operator exposes:

- `shape`
- `matvec(x)` for forward multiplication
- `rmatvec(x)` for transpose multiplication
- `to_dense()` for inspection and validation

The dense conversion is part of the first-slice contract because these operators
are intended for small-to-moderate scientific helper problems where parity tests
against explicit matrices are the clearest evidence.

## Primitive operators

`DenseOperator(matrix)` wraps an explicit matrix. `DiagonalOperator(diagonal)`
stores only the diagonal and applies it by elementwise multiplication.

Both are JAX PyTrees, so the array values remain differentiable leaves under
`jit`, `vmap`, and `grad`.

## Composition

The composition helpers preserve the mathematical operation in the call graph:

- `scale(op, scalar)` represents a scalar multiple.
- `add(left, right)` represents an operator sum with identical shape.
- `compose(left, right)` represents `left @ right`.
- `transpose(op)` swaps `matvec` and `rmatvec`.
- `block_diag(*blocks)` applies each block to its matching vector slice.

Shape checks happen eagerly in the composition helpers when the necessary shapes
are available from the operator objects.

## Validation

Unit tests compare every operation against explicit dense matrix algebra,
including reverse multiplication and block-diagonal assembly. Validation tests
compare FD-vs-AD gradients through dense and diagonal operator leaves.

This module does not implement sparse formats, iterative solves, preconditioners,
or implicit differentiation. Those remain out of scope for the foundation layer.
