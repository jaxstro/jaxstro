---
title: One-dimensional interpolation
description: >-
  Linear, Hermite, natural cubic, and monotone cubic interpolation with
  explicit boundary and derivative contracts.
---

## The question this method answers

Given values known only at ordered coordinates, what value should represent the
table between samples? Interpolation fills finite gaps under an explicit shape
assumption. It does not discover unresolved physics or justify extrapolation.

:::{tip}
Use linear interpolation when honesty about unresolved curvature matters most.
Use natural cubic splines for global $C^2$ smoothness, and PCHIP-style monotone
cubics when preserving monotonicity and range matters more than global second-
derivative continuity.
:::

## Before computation: what should be true?

The coordinate array must be one-dimensional, contain at least two points, and
be strictly increasing. The selected value axis must have the same length. State
whether the data justify linearity, smooth curvature, known node derivatives, or
monotone shape. Choose clamping or extrapolation before computing.

:::{important}
Interpolation error depends on unresolved curvature and sample placement. A
smooth curve through all samples is not evidence that the curve is physically
correct between them.
:::

The meaning of coordinate units and scales is developed in
[](../../10-foundations/mathematical-objects/functions-units-scales.md); table
payloads connect to [](../../30-representations/representations.md).

## Define the mathematical objects

Let $x_0<\cdots<x_{n-1}$ be knots and let $y_i$ be scalar samples. The linear,
Hermite, and PCHIP implementations also allow array-valued payloads along a
selected axis; the natural-cubic implementation does not. For a query
$x\in[x_i,x_{i+1}]$, define the interval width $h_i=x_{i+1}-x_i$ and local
coordinate $t=(x-x_i)/h_i\in[0,1]$. Node derivatives are denoted $m_i$ and
secant slopes are $d_i=(y_{i+1}-y_i)/h_i$.

A boundary policy specifies the value outside $[x_0,x_{n-1}]`: clamp to endpoint
values or numerically continue the endpoint segment. A limiter branch is the
piecewise rule that chooses monotone cubic slopes from neighboring secants.

## Derive the method

The unique straight line through two adjacent samples is

```{math}
:label: eq-linear-interpolant
L_i(x)=(1-t)y_i+t y_{i+1}.
```

To match both endpoint values and endpoint derivatives, use the cubic Hermite
basis:

```{math}
:label: eq-hermite-interpolant
H_i(x)=h_{00}(t)y_i+h_{10}(t)h_i m_i
       +h_{01}(t)y_{i+1}+h_{11}(t)h_i m_{i+1},
```

where

```{math}
h_{00}=2t^3-3t^2+1,\quad h_{10}=t^3-2t^2+t,\quad
h_{01}=-2t^3+3t^2,\quad h_{11}=t^3-t^2.
```

For monotone PCHIP slopes, adjacent secants with different signs imply a zero
node derivative. Otherwise {cite:t}`FritschButland1984` gives the weighted
harmonic mean

```{math}
:label: eq-pchip-slope
m_i=\frac{w_1+w_2}{w_1/d_{i-1}+w_2/d_i},\qquad
w_1=2h_i+h_{i-1},\qquad w_2=h_i+2h_{i-1}.
```

Endpoint slopes receive additional sign and three-times-secant limiters. These
branches prevent a monotone table from gaining a new interior extremum.

The natural cubic instead solves for knot second derivatives $M_i$ with
$M_0=M_{n-1}=0$ and

```{math}
h_{i-1}M_{i-1}+2(h_{i-1}+h_i)M_i+h_iM_{i+1}
=6(d_i-d_{i-1}).
```

This creates a globally $C^2$ interpolant, but it need not preserve monotonicity
or remain inside the sample range {cite:t}`deBoor2001`.

## What the algorithm actually does

`interp1d` locates intervals with `searchsorted`, computes [](#eq-linear-interpolant),
and defaults to endpoint clamping. `cubic_hermite_interp` evaluates supplied
derivatives. `pchip_slopes` constructs limited slopes, and
`monotone_cubic_interp` passes them to the Hermite evaluator. Natural-cubic
coefficient construction uses `jnp.linalg.solve`; `eval_cubic_spline` evaluates
the selected interval in nested polynomial form and clamps its query first.

Array-valued payloads are supported by the linear, Hermite, and PCHIP paths.
By contrast, `natural_cubic_spline_coeffs`, `eval_cubic_spline`, and
`NaturalCubicSpline1D` support scalar one-dimensional `y` only. The current
natural-cubic coefficient routine does not cleanly reject an array-valued `y`;
it reaches incompatible core array assembly and raises `TypeError`. Treat that
failure as a limitation, not as array-payload support.

Concrete wrappers check the shapes they explicitly support and strictly
increasing grids. Value-dependent exceptions cannot fire on
tracers: eager validation is skipped while the grid is traced; the caller must supply a strictly increasing grid under `jax.jit`.
With `extrapolate=True`, the endpoint linear or Hermite segment is numerical continuation, not a physical guarantee.

## What JAX differentiates

```{list-table} Interpolation gradient contracts
:header-rows: 1
:label: tbl-interpolation-gradient-contracts

* - Operation
  - Contract
  - Supported claim
  - Boundary
* - Hermite values and supplied derivatives
  - `smooth_pathwise`
  - AD agrees with finite differences in a fixed segment.
  - The query remains away from knots and clamping.
* - Natural-spline values and interior query
  - `smooth_pathwise`
  - For scalar one-dimensional `y`, the coefficient solve and evaluation carry
    local gradients.
  - The selected interval remains fixed.
* - PCHIP inside a fixed limiter branch
  - `smooth_pathwise`
  - AD applies while secant signs and limiter decisions do not change.
  - A limiter transition changes the executed derivative rule.
* - Clamped exterior query
  - `known_zero`
  - Endpoint output is locally constant in the exterior query.
  - Saturation is not an inference direction.
* - Knots and limiter transitions
  - `validation_only`
  - Values, bounds, and one-sided behavior can be checked.
  - No universal smooth derivative is claimed.
```

:::{warning}
`searchsorted`, limiter predicates, and clamping define derivative boundaries.
Knots and limiter transitions are piecewise-smooth locations, not failed AD.
Gradients with respect to a clamped exterior query are zero by construction.
:::

## Using it in Jaxstro

```python
from jaxstro.jaxconfig import enable_high_precision

enable_high_precision()  # before creating JAX arrays

import jax.numpy as jnp

from jaxstro.numerics.interpolation import (
    MonotoneTabulatedFunction1D,
    cubic_hermite_interp,
    eval_cubic_spline,
    monotone_cubic_interp,
    natural_cubic_spline_coeffs,
    pchip_slopes,
)

x_grid = jnp.arange(5.0)
values = jnp.array([0.0, 0.01, 0.9, 0.91, 1.0])
x_new = jnp.linspace(0.0, 4.0, 801)
dydx = pchip_slopes(x_grid, values)

hermite = cubic_hermite_interp(x_grid, values, dydx, x_new)
natural = eval_cubic_spline(
    x_grid,
    natural_cubic_spline_coeffs(x_grid, values),
    x_new,
)
monotone = monotone_cubic_interp(x_grid, values, x_new)
table = MonotoneTabulatedFunction1D(x_grid, values)
wrapped_monotone = table(x_new)

assert natural.min() < -0.1
assert monotone.min() >= -1e-12
assert monotone.max() <= 1.0 + 1e-12
assert jnp.all(jnp.diff(monotone) >= -1e-12)
assert jnp.allclose(hermite, monotone)
assert jnp.allclose(wrapped_monotone, monotone)
```

Prepared `TabulatedFunction1D`, `MonotoneTabulatedFunction1D`, and
`NaturalCubicSpline1D` objects are registered PyTrees. Their table arrays are
dynamic leaves; interpolation axes stored by the monotone wrapper are static
auxiliary data. This PyTree behavior does not widen `NaturalCubicSpline1D`
beyond its scalar one-dimensional value contract.

## How to audit the result

Check exact reproduction at every knot, then probe midpoints. For monotone data,
audit output range and successive increments. For a smooth analytic function,
refine the grid and compare errors. Compare AD with central finite differences
only at interior queries whose interval and limiter branch remain fixed.

:::{figure} ../../10-theory/figures/interpolation-shape-contracts.webp
:name: fig-interpolation-shape-contracts
:alt: Two-panel comparison of natural cubic and PCHIP interpolation for the same monotone samples, showing natural-spline undershoot and nonnegative PCHIP increments

The same monotone samples demonstrate that natural-cubic smoothness and PCHIP
shape preservation are different contracts, not a universal ranking.
:::

Executable anchors are indexed in [](../../60-validation/validation.md).

## Where the claim stops

These are one-dimensional table primitives. They do not estimate interpolation
error, infer missing structure, validate a table's physical meaning, or handle
scattered data and multidimensional monotonicity. PCHIP preserves monotone shape
for monotone samples; it does not make noisy or biased data correct.

## Connected ideas

:::{seealso}
Connect interpolation to derivative meaning in
[](../../10-foundations/mathematical-objects/what-is-a-derivative.md), table
representations in [](../../30-representations/representations.md), exact owner
signatures in [](../../50-api/approximation-integration/interpolation.md), the
gradient taxonomy in [](../methods.md#gradient-contracts), and evidence in
[](../../60-validation/validation.md). Tensor-product tables continue in
[](./regular-grid.md), and fixed-knot smooth bases in [](./bsplines.md).
:::
