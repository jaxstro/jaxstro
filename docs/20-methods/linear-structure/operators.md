---
title: Linear operators
description: >-
  Small PyTree linear maps, composition, transpose action, and dense parity
  without committing Jaxstro to a solver stack.
---

## The question this method answers

How can code carry a matrix-like transformation without materializing every sum,
product, transpose, or block as one dense array? A linear operator records how a
map acts on a vector and how its transpose acts on a cotangent.

:::{important}
"Matrix-free" describes representation, not automatic speed. Jaxstro's small
operators provide composable algebra and a dense audit surface; they do not own
an iterative solver or a sparse storage format.
:::

## Before computation: what should be true?

Name the domain and codomain dimensions. An operator with shape $(m,n)$ accepts
a vector of shape $(n,)$ in `matvec` and a vector of shape $(m,)$ in `rmatvec`.
Composition requires matching inner dimensions, addition requires identical
shapes, and a block diagonal operator requires at least one block.

:::{warning}
Composition helpers check operator metadata, but individual `matvec` calls rely
on JAX array algebra for input-shape failure. Shape checks happen eagerly when
`add`, `compose`, or `block_diag` constructs the Python operator structure.
:::

## Define the mathematical objects

A map $A$ is linear when scalar coefficients distribute over inputs:

```{math}
A(\alpha x+\beta z)=\alpha A(x)+\beta A(z).
```

The transpose action $A^\mathsf{T}y$ maps a cotangent in the output space back
to the input space. A PyTree is a nested JAX structure with array leaves and
static structure. Jaxstro operators are Equinox modules: matrix, diagonal, and
scalar values are differentiable PyTree leaves, while the chosen composition
tree and its shapes are static program structure.

## Derive the method

The defining linearity relation is

```{math}
:label: eq-linear-operator
A(\alpha x+\beta z)=\alpha A(x)+\beta A(z).
```

For compatible maps $B:\mathbb{R}^n\rightarrow\mathbb{R}^k$ and
$A:\mathbb{R}^k\rightarrow\mathbb{R}^m$, composition applies the right map
first:

```{math}
:label: eq-operator-composition
(A\circ B)x=A(Bx)=(AB)x.
```

Reverse multiplication follows by reversing the factors. The defining adjoint
identity is

```{math}
:label: eq-operator-adjoint
\langle y,Ax\rangle=\langle A^\mathsf{T}y,x\rangle,
\qquad
(AB)^\mathsf{T}y=B^\mathsf{T}(A^\mathsf{T}y).
```

These relations give independent audits that do not depend on the operator's
internal representation.

## What the algorithm actually does

`DenseOperator` stores an $(m,n)$ array. `DiagonalOperator` stores an $(n,)$
diagonal. `scale`, `add`, `compose`, and `transpose` wrap existing operators;
they do not materialize a matrix during `matvec` or `rmatvec`.
`block_diag(*blocks)` slices an input according to block widths, applies each
block, and concatenates the outputs. `to_dense()` is deliberately public so
small and moderate fixtures can be compared against explicit matrix algebra.

The block implementation uses Python loops over the static block tuple. Python
structure is static during tracing; the array work inside each operator remains
JAX-traceable. Changing the number or types of blocks creates a different
program structure and can trigger recompilation.

Thus shape checks happen eagerly for composition metadata; vector shape errors
remain ordinary JAX array-algebra errors at application time.

## What JAX differentiates

JAX differentiates `matvec`, `rmatvec`, and `to_dense` with respect to floating
array leaves and floating input vectors. For a scalar loss built from $Ax$, AD
computes derivatives of the executed composition, including sensitivities to a
dense matrix, a diagonal, or a scale leaf. It does not differentiate the
Protocol, tuple length, shapes, slicing offsets, or choice of operator class.

An operator transpose is an algebraic transpose, not a custom derivative rule.
No solve occurs, so these classes make no implicit-solution derivative claim.

## Using it in Jaxstro

```python
import jax
import jax.numpy as jnp

from jaxstro.numerics.operators import DenseOperator, DiagonalOperator, compose

left_matrix = jnp.array([[1.0, 2.0], [0.0, 1.0]])
right_matrix = jnp.array([[2.0, 0.0], [1.0, 3.0]])
left = DenseOperator(left_matrix)
right = DiagonalOperator(jnp.array([2.0, 3.0]))
operator = compose(left, right)

x = jnp.array([0.5, -1.0])
y = jnp.array([1.5, 0.25])
forward = operator.matvec(x)
reverse = operator.rmatvec(y)
dense = operator.to_dense()
gradient = jax.grad(lambda vector: jnp.sum(operator.matvec(vector) ** 2))(x)

assert operator.shape == (2, 2)
assert jnp.allclose(forward, dense @ x)
assert jnp.allclose(reverse, dense.T @ y)
assert jnp.allclose(jnp.vdot(y, forward), jnp.vdot(reverse, x))
assert gradient.shape == x.shape
```

## How to audit the result

1. Compare `matvec(x)` with `to_dense() @ x` on several nontrivial vectors.
2. Compare `rmatvec(y)` with `to_dense().T @ y`.
3. Check the adjoint inner-product identity to float tolerance.
4. Compare sums, products, transposes, and block assembly with explicit matrices.
5. For claimed leaf gradients, compare AD with central finite differences while
   the composition tree and shapes remain fixed.

:::{tip}
Use `to_dense()` as an audit oracle only when the fixture is small enough. The
scientific claim should concern algebraic parity, not the dense conversion's
scalability.
:::

## Where the claim stops

The module does not provide sparse formats, iterative solves, preconditioners,
shape-polymorphic block structure, or custom implicit differentiation. Dense
parity demonstrates that a represented operator matches its explicit matrix on
the tested fixtures; it does not establish conditioning or solver convergence.

## Connected ideas

:::{seealso}
Start with linear maps in
[](../../10-foundations/mathematical-objects/linear-algebra-language-of-change.md),
then connect operator trees to
[](../../30-representations/parameters-state/pytrees-as-scientific-state.md) and
the differentiable-program workflow in
[](../../40-workflows/differentiable-research/what-jax-differentiates.md).
The owner reference is [](../../50-api/linear-structure/operators.md), and
executable evidence routes through [](../../60-validation/validation.md).
:::
