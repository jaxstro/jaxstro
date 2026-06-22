---
title: Autodiff products
description: >-
  Thin JAX-native helpers for JVPs, VJPs, HVPs, Gauss-Newton products, and
  empirical Fisher-style products.
---

`jaxstro.numerics.autodiff` exposes small derivative-product helpers that
downstream scientific packages often reimplement in tests, optimizers, and
diagnostics. These helpers do not change JAX's differentiation semantics; they
name common products and keep call sites readable.

## First-order products

`jvp(f, x, tangent)` returns `(f(x), J tangent)`.
`vjp(f, x, cotangent)` returns `(f(x), J^T cotangent)`.

`jacobian_vector_product(...)` and `vector_jacobian_product(...)` are aliases
that return only the product.

## Curvature products

`hvp(f, x, tangent)` computes the Hessian-vector product for scalar-output
functions by differentiating `jax.grad(f)` with a JVP.

`gauss_newton_product(residual_fn, x, tangent)` computes

```{math}
J^\mathsf{T}Jv
```

for a residual function. This is useful for least-squares diagnostics without
forming a dense Jacobian in production code.

`empirical_fisher_product(score_fn, params, data, tangent)` computes the mean
outer-product score product from per-example score vectors. It is intentionally
generic: probability-model semantics remain outside `jaxstro`.

## Validation

Unit tests compare these helpers against explicit dense Jacobians, Hessians, and
score matrices on small functions. Validation tests compare HVPs against a
finite-difference derivative of the gradient.
