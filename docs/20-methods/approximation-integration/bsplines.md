---
title: B-splines
description: >-
  Fixed-knot local basis construction, calculus, fitting, and regularization
  primitives with explicit derivative boundaries.
---

## The question this method answers

How can a smooth function be represented by local basis functions so that one
coefficient changes only a limited region? A degree-$p$ B-spline writes
$S(x)=\sum_i c_iB_{i,p}(x)$ using knots, local basis functions, and coefficients.

:::{tip}
Use B-splines when local support and explicit smoothness are more useful than a
single global polynomial. Keep knot selection and the regularization weight as
visible modeling decisions.
:::

## Before computation: what should be true?

The knot vector must be one-dimensional, nondecreasing, long enough for the
nonnegative integer degree, and define a positive-width active domain. The
coefficient axis must have length
$n_{\mathrm{basis}}=n_{\mathrm{knots}}-p-1$. Decide whether knots are fixed
representation choices or data-dependent preprocessing.

:::{important}
Basis construction, coefficient fitting, and smoothing policy are separate
problems. Jaxstro supplies fixed-knot mechanics and a roughness primitive; it
does not choose knots, smoothing strength, or scientific model complexity.
:::

Basis vectors are a linear representation; see
[](../../10-foundations/mathematical-objects/linear-algebra-language-of-change.md)
and [](../../30-representations/representations.md).

## Define the mathematical objects

Let $t_0\le\cdots\le t_{K-1}$ be knots, $p\ge0$ the polynomial degree, and
$B_{i,p}$ the $i$th normalized basis function. Its support is contained in
$[t_i,t_{i+p+1}]$. Inside the active domain, bases are nonnegative and form a
partition of unity. A clamped open knot vector repeats each endpoint $p+1$
times.

The coefficient vector $c$ defines $S$. A design matrix has entries
$B_{ji}=B_{i,p}(x_j)$. A derivative-order-$m$ roughness functional measures the
integrated square of $S^{(m)}$.

## Derive the method

The degree-zero basis selects one half-open knot interval:

```{math}
:label: eq-bspline-zero
B_{i,0}(x)=
\begin{cases}
1,&t_i\le x<t_{i+1},\\
0,&\text{otherwise}.
\end{cases}
```

The Cox-de Boor recurrence builds higher degree from adjacent lower-degree
bases {cite:t}`deBoor1972`:

```{math}
:label: eq-cox-de-boor
B_{i,p}(x)=
\frac{x-t_i}{t_{i+p}-t_i}B_{i,p-1}(x)
+\frac{t_{i+p+1}-x}{t_{i+p+1}-t_{i+1}}B_{i+1,p-1}(x).
```

A zero denominator contributes zero, which is the standard repeated-knot
convention. Differentiating the spline gives a degree-$(p-1)$ spline with

```{math}
:label: eq-bspline-derivative
c'_i=p\frac{c_{i+1}-c_i}{t_{i+p+1}-t_{i+1}}.
```

The coefficient sensitivity is especially simple:
$\partial S(x)/\partial c_i=B_{i,p}(x)$. For smoothing, Jaxstro approximates

```{math}
:label: eq-bspline-roughness
R_m(c)=\int \left[S^{(m)}(x)\right]^2 dx
```

on a fixed sample grid. A caller may form an objective such as
$\lVert Bc-y\rVert_2^2+\lambda R_m(c)$, but $\lambda$ remains caller-owned.

Antiderivative increments obey
$d_{i+1}-d_i=c_i(t_{i+p+1}-t_i)/(p+1)$, enabling definite integrals by endpoint
subtraction.

## What the algorithm actually does

`bspline_basis` evaluates [](#eq-cox-de-boor) for every basis and clamps queries
to the active domain. `bspline_eval` contracts coefficients with that basis;
`bspline_eval_deboor` provides the equivalent local de Boor evaluator.
`bspline_derivative`, `bspline_antiderivative`, and `bspline_integral` transform
coefficients and knot vectors as above.

`open_uniform_knots` creates clamped equally spaced interior knots.
`adaptive_open_uniform_knots` places interior knots at sample quantiles; this is
deterministic preprocessing, not knot optimization. `fit_bspline_lstsq` solves
ordinary fixed-knot least squares. `tensor_product_design_matrix` computes a
row-wise Kronecker product. Degree and relevant axes are static under JIT.

Invalid concrete degree, knot order, coefficient shape, sample shape, or active
domain raises `ValueError`. Value-dependent knot checks cannot raise while the
knots are traced.

## What JAX differentiates

```{list-table} B-spline gradient contracts
:header-rows: 1
:label: tbl-bspline-gradient-contracts

* - Operation
  - Contract
  - Supported claim
  - Boundary
* - Coefficients at fixed knots
  - `smooth_pathwise`
  - AD returns the active basis vector.
  - Knots and degree remain fixed.
* - Interior query coordinate
  - `smooth_pathwise`
  - AD agrees with finite differences in a smooth knot span.
  - Query stays away from repeated knots and boundaries.
* - Clamped exterior coordinate
  - `known_zero`
  - Evaluator and analytic derivative are constant outside.
  - Saturation is not an inference direction.
* - Knot boundaries
  - `validation_only`
  - Multiplicity-specific continuity can be checked.
  - No universal knot gradient is claimed.
* - Quantile knot construction
  - `validation_only`
  - Deterministic placement can be reproduced.
  - Sorting and quantiles are preprocessing boundaries.
```

:::{warning}
Smoothness at a knot depends on multiplicity. Clamped exterior derivatives are
zero, while quantile knot construction and active-span selection are not smooth
inference paths. A least-squares solution also inherits rank and conditioning
limits from its design matrix.
:::

## Using it in Jaxstro

```python
from jaxstro.jaxconfig import enable_high_precision

enable_high_precision()  # before creating JAX arrays

import jax.numpy as jnp

from jaxstro.numerics.splines import (
    BSpline1D,
    bspline_basis,
    bspline_derivative,
    bspline_eval,
    open_uniform_knots,
)

knots = open_uniform_knots(0.0, 1.0, n_basis=6, degree=3)
coeffs = jnp.array([0.0, 0.25, 0.9, 0.7, 0.2, 0.1])
x = jnp.linspace(0.0, 1.0, 9)

basis = bspline_basis(knots, x, degree=3)
values = bspline_eval(knots, coeffs, x, degree=3)
derivative = bspline_derivative(knots, coeffs, x, degree=3)

spline = BSpline1D(knots, coeffs, degree=3)
wrapped_values = spline(x)

assert basis.shape == (9, 6)
assert jnp.allclose(basis.sum(axis=-1), 1.0)
assert jnp.allclose(values, wrapped_values)
assert jnp.all(jnp.isfinite(derivative))
```

## How to audit the result

Check basis nonnegativity, local support, and partition of unity. Compare basis
contraction against de Boor evaluation. Verify AD with respect to coefficients
against the basis vector and AD with respect to an interior query against both
the analytic derivative and central finite differences. For a fit, inspect
design rank, condition, residuals, and sensitivity to knots and regularization.

:::{figure} ../../10-theory/figures/bspline-local-support.webp
:name: fig-bspline-local-support
:alt: Six cubic B-spline basis curves with local support and their sum equal to one across the active domain

The public basis values show local support and partition of unity for one
open-uniform cubic configuration. They do not validate knot or smoothing-model
selection.
:::

Measured anchors are indexed in [](../../60-validation/validation.md).

## Where the claim stops

Jaxstro does not own smoothing-spline model selection, adaptive-knot optimization,
extrapolation, sparse tensor-product storage, uncertainty calibration, or a
domain-specific regularization policy. Deterministic quantile knots are not an
optimized adaptive spline.

## Connected ideas

:::{seealso}
Connect basis matrices to
[](../../10-foundations/mathematical-objects/linear-algebra-language-of-change.md),
representations to [](../../30-representations/representations.md), signatures
to [](../../50-api/approximation-integration/splines.md), the gradient taxonomy
to [](../methods.md#gradient-contracts), and evidence to
[](../../60-validation/validation.md). Direct table interpolation is in
[](./interpolation.md).
:::
