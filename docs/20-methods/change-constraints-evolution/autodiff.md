---
title: Autodiff products
description: >-
  JVPs, VJPs, curvature products, and the scientific meaning of derivatives
  taken through executed JAX programs.
---

## The question this method answers

How does a scientific program's output change under a small, specified change
to its input, and how can that local change be computed without forming a dense
Jacobian? Autodiff products answer this question for the program that JAX
actually executes. Begin with [](../../10-foundations/mathematical-objects/what-is-a-derivative.md)
if derivatives as local linear maps are new.

:::{tip}
Use a Jacobian-vector product (JVP) when the input direction is known. Use a
vector-Jacobian product (VJP) when a scalar loss or output direction supplies a
cotangent. Form a dense Jacobian only when its entries are themselves the object
of study.
:::

## Before computation: what should be true?

The function must accept and return floating JAX arrays with shapes compatible
with the proposed tangent or cotangent. More importantly, the executed branch
must represent the scientific perturbation being claimed. A derivative through
a clip, discrete index, or branch transition can be finite yet answer the wrong
question.

:::{important}
Write down the input quantity, output quantity, perturbation direction, units,
and smooth domain before calling AD. JAX transformability is an execution fact,
not a certificate that the resulting sensitivity is scientifically meaningful.
:::

## Define the mathematical objects

Let $f:\mathbb{R}^n\rightarrow\mathbb{R}^m$ be differentiable at $x$. Its
derivative $D f(x)$ is the linear map that gives the first-order response

```{math}
f(x+\epsilon v)=f(x)+\epsilon D f(x)[v]+o(\epsilon),
```

where $v\in\mathbb{R}^n$ is a tangent direction and $o(\epsilon)/\epsilon\to0$.
The Jacobian $J\in\mathbb{R}^{m\times n}$ is a coordinate representation of
that map. A cotangent $w\in\mathbb{R}^m$ weights output directions. For a
scalar $f:\mathbb{R}^n\to\mathbb{R}$, the Hessian
$H=\nabla^2 f(x)\in\mathbb{R}^{n\times n}$ describes local curvature.

The data representation matters: parameter arrays and scientific PyTrees are
discussed in [](../../30-representations/parameters-state/pytrees-as-scientific-state.md).

## Derive the method

The JVP pushes the input direction $v$ through the derivative:

```{math}
:label: eq-autodiff-jvp
\operatorname{JVP}(f,x,v)=D f(x)[v]=Jv.
```

The VJP pulls the output cotangent $w$ back to input space:

```{math}
:label: eq-autodiff-vjp
\operatorname{VJP}(f,x,w)=D f(x)^{\mathsf T}[w]=J^{\mathsf T}w.
```

These are adjoint operations. Their defining scalar identity is

```{math}
:label: eq-autodiff-adjoint
w^\mathsf{T} D f(x)[v]
=w^\mathsf{T}Jv
=v^\mathsf{T}J^\mathsf{T}w.
```

For scalar $f$, applying a JVP to the gradient avoids materializing $H$:

```{math}
H v = D(\nabla f)(x)[v].
```

For residuals $r(x)\in\mathbb{R}^m$, the least-squares objective
$F(x)=\tfrac12 r(x)^\mathsf{T}r(x)$ has the Gauss-Newton curvature
approximation $J_r^\mathsf{T}J_r$. Its product is computed as one JVP followed
by one VJP. For per-example score vectors $s_i$, the empirical Fisher-style
product is $N^{-1}\sum_i s_i(s_i^\mathsf{T}v)$.

## What the algorithm actually does

`jvp` delegates to `jax.jvp` and returns `(f(x), Jv)`. `vjp` constructs JAX's
pullback and returns `(f(x), J.T @ w)`. The product-only aliases discard the
primal value. `hvp` applies `jax.jvp` to `jax.grad(f)`. `gauss_newton_product`
chains the module's JVP and VJP helpers. `empirical_fisher_product` vmaps a
two-argument score function over the leading data axis, stacks the scores, and
applies the mean outer-product matrix without constructing that matrix.

No helper sanitizes non-finite values, checks scientific units, or changes JAX's
dtype rules. Shape, tracing, and dtype errors propagate.

## What JAX differentiates

JAX differentiates the finite program represented by `f` along the supplied
direction. JVPs use forward-mode linearization; VJPs use a reverse-mode
pullback. Curvature products differentiate the executed gradient or residual
program, including its smooth branches and any local saturation.

:::{warning}
A finite JVP, VJP, or HVP does not validate the model or the derivative target.
At clips, limiter switches, discrete choices, or singular points, AD may return
a one-sided convention, zero, or a branch-local value. Empirical Fisher-style
products are generic score outer products; Jaxstro does not claim that a given
`score_fn` is a likelihood score or that the result is a calibrated Fisher
information matrix.
:::

## Using it in Jaxstro

Use the owner-qualified module so the runtime boundary is explicit:

```python
import jax.numpy as jnp

from jaxstro.numerics.autodiff import jvp, vjp


def model(x):
    return jnp.array([x[0] ** 2 + x[1], jnp.sin(x[1])])


x = jnp.array([2.0, 0.5])
v = jnp.array([0.1, -0.2])
w = jnp.array([1.0, 3.0])
value, pushed = jvp(model, x, v)
_, pulled = vjp(model, x, w)
```

Here `x` and `v` have shape `(2,)`, `value` and `w` have shape `(2,)`, `pushed`
has the output shape, and `pulled` has the input shape. `hvp` requires a scalar
output. The current empirical Fisher helper assumes vector parameters and
per-example vector scores compatible with ordinary matrix products.

## How to audit the result

Choose a point away from known nonsmooth boundaries. Compare $Jv$ with the
central directional finite difference

```{math}
\frac{f(x+hv)-f(x-hv)}{2h},
```

then repeat over a decreasing sequence of $h$ values to separate truncation
error from roundoff. Check the adjoint identity in [](#eq-autodiff-adjoint) with
independent $v$ and $w$. For an HVP, finite-difference the gradient, not the
original scalar function. Record dtypes, units, step sizes, absolute and
relative disagreements, and whether the executed branch stayed fixed.

The package-wide audit vocabulary and executable evidence are in
[](../../60-validation/methods/validation-methods.md).

## Where the claim stops

These helpers reduce the cost and clarify the spelling of derivative products.
They do not prove differentiability, condition a model, choose meaningful
directions, certify a Hessian, or supply inference semantics. Dense Jacobian
parity on a toy problem is implementation evidence, not scientific validation
of a downstream model.

## Connected ideas

:::{seealso}
Connect local linear maps to
[](../../10-foundations/models-and-computation/sensitivity-conditioning-identifiability.md),
structured parameters to
[](../../30-representations/parameters-state/parameters-and-transforms.md),
runtime ownership to [](../../50-api/change-constraints/autodiff.md), and audit
claims to [](../../60-validation/validation.md). Optimization curvature products
continue in [](./optimization.md).
:::
