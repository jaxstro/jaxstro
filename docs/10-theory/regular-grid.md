---
title: Regular-grid interpolation
description: >-
  Static-rank multilinear interpolation for gridded scientific tables with
  explicit boundary policy.
---

Regular-grid interpolation is the table primitive for data that live on a
Cartesian product of one-dimensional axes. Atmosphere grids, calibration
surfaces, and tracks often have this shape: each coordinate axis is sorted, and
the value array stores samples on the tensor grid.

jaxstro exposes the generic numerical kernel:

```python
from jaxstro.numerics import regular_grid

y = regular_grid.regular_grid_interp((x_axis, y_axis), values, xi)
y2 = regular_grid.bilinear_interp(x_axis, y_axis, values, x_new, y_new)
y3 = regular_grid.trilinear_interp(x_axis, y_axis, z_axis, values, x_new, y_new, z_new)
```

The leading dimensions of `values` are the grid axes. Any trailing dimensions are
treated as payload axes, so one interpolation call can return vector-valued
quantities such as spectra, coefficient vectors, or multiple diagnostics.

## Boundary Policy

The boundary policy is explicit:

- `boundary="clamp"` clips query coordinates to the grid domain.
- `boundary="fill"` returns `fill_value` for any query point outside the domain.
- `boundary="reject"` raises eagerly when concrete query points are outside the
  grid.

`reject` is an eager/debug guard. Inside `jax.jit`, value-dependent Python
exceptions cannot fire on traced query coordinates, so callers should validate
domain membership before entering compiled model code when rejection is required.

## Multilinear Weights

For each dimension, the evaluator finds the enclosing interval and computes a
fractional coordinate:

```{math}
t_d = \frac{x_d - a_{d,i}}{a_{d,i+1} - a_{d,i}}.
```

The final value is the weighted sum over all `2^D` cell corners. In two
dimensions this is bilinear interpolation; in three dimensions it is trilinear
interpolation. The ND function uses the same static-rank corner sum.

## Differentiability

Inside a grid cell, multilinear interpolation is differentiable with respect to
both the grid values and the query coordinates. The validation suite compares
autodiff against finite differences for both paths.

At cell boundaries, the active cell changes through `searchsorted`, so gradients
are piecewise-defined. Tests therefore use interior query coordinates for
gradient claims and separate validation for exact affine recovery.

## What This Does Not Do

This primitive assumes regular-grid structure. It does not handle scattered data,
triangulations, adaptive meshes, multidimensional monotonicity constraints, or
domain-specific grid selection. Those policies belong in higher-level packages or
future, separately validated modules.
