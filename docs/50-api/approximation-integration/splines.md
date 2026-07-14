---
title: B-splines
---

# B-splines

## Owner import path

`jaxstro.numerics.splines`

## Purpose

B-spline basis construction, evaluation, calculus, fitting, and tensor-product
design mechanics for fixed knot vectors.

## Public records and callables

`BSpline1D`, `open_uniform_knots`, `adaptive_open_uniform_knots`,
`bspline_basis`, `bspline_design_matrix`, `bspline_eval`,
`bspline_eval_deboor`, `bspline_derivative`, `bspline_antiderivative`,
`bspline_integral`, `bspline_roughness_penalty`, `fit_bspline_lstsq`, and
`tensor_product_design_matrix`.

## Shape and dtype expectations

Knots and coefficients are one-dimensional floating arrays along the spline
axis. Degree and axis are static; tensor-product designs require aligned sample
rows.

## JAX transforms and AD classification

Fixed-knot evaluation composes with JIT and smooth-pathwise AD away from knots.
Adaptive knot placement and rank or branch changes are preprocessing boundaries.

## Failure behavior

Invalid degree, knot order, coefficient shape, or unsupported axis raises for
concrete inputs. Fitting exposes the underlying linear-solve behavior.

## Contract and evidence links

See [](../../20-methods/approximation-integration/bsplines.md) and
[](../../60-validation/index.md).

## Canonical import example

```python
from jaxstro.numerics.splines import BSpline1D
```
