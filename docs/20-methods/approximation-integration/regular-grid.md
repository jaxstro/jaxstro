---
title: Regular-grid interpolation
description: >-
  Static-rank multilinear interpolation for tensor-product tables with explicit
  payload, boundary, and derivative contracts.
---

## The question this method answers

Given values sampled on the Cartesian product of several ordered axes, what
value should represent an interior coordinate? Regular-grid interpolation uses
the surrounding $2^D$ cell corners, preserving any trailing payload shape.

:::{tip}
Use this method for tensor-product grids. If the samples are scattered, the mesh
is adaptive, or cross-axis monotonicity is required, choose a representation and
method that own those structures explicitly.
:::

## Before computation: what should be true?

Every axis must be one-dimensional, strictly increasing, and contain at least
two points. If axis $d$ has length $n_d$, the leading shape of `values` must be
`(n_0, ..., n_{D-1})`. The query's final axis must have length $D$. Choose
`clamp`, `fill`, or eager `reject` as a scientific boundary policy.

:::{important}
The fill sentinel is a numerical policy, not missing-data semantics. Decide
whether clamping, a finite sentinel, NaN, masking, or a restricted prior is
meaningful before interpolation enters a larger model.
:::

Tensor-product domains connect to
[](../../30-representations/fields/topology-and-discretization.md) and axis units
to [](../../10-foundations/mathematical-objects/functions-units-scales.md).

## Define the mathematical objects

For dimension $d$, let the axis be
$a_{d,0}<\cdots<a_{d,n_d-1}$. A query $\boldsymbol{x}\in\mathbb{R}^D$ lies in a
cell whose lower corner has indices $\boldsymbol{i}=(i_1,\ldots,i_D)$. The table
value at a corner is $f_{\boldsymbol{i}+\boldsymbol{b}}$, where
$\boldsymbol{b}\in\{0,1\}^D$. Trailing dimensions of each table entry are the
payload and are not interpolation axes.

## Derive the method

Normalize each coordinate inside its enclosing interval:

```{math}
:label: eq-regular-grid-coordinate
t_d=\frac{x_d-x_{d,i_d}}{x_{d,i_d+1}-x_{d,i_d}},\qquad 0\le t_d\le1.
```

In one dimension, the lower and upper weights are $1-t_d$ and $t_d$. Taking the
product of those independent weights across dimensions gives

```{math}
:label: eq-multilinear-interpolant
F(\boldsymbol{x})=
\sum_{\boldsymbol{b}\in\{0,1\}^D}
f_{\boldsymbol{i}+\boldsymbol{b}}
\prod_{d=1}^{D} t_d^{b_d}(1-t_d)^{1-b_d}.
```

The weights are nonnegative and sum to one inside a cell. Consequently the
interpolant reproduces constants and affine functions exactly and remains in
the convex hull of scalar corner values. This rectangular-table construction is
described by {cite:t}`WeiserZarantonello1988`.

## What the algorithm actually does

`regular_grid_interp` validates rank and static shapes, clips coordinates for
cell lookup, uses `searchsorted` on each axis, and loops in Python over the
statically known $2^D$ corner tuples while tracing the JAX arithmetic. Query
shape is `xi.shape[:-1]`; output shape is that query shape followed by
`values.shape[D:]`.

`boundary="clamp"` evaluates at the nearest endpoint coordinate.
`boundary="fill"` replaces the complete payload if any coordinate is outside.
`boundary="reject"` raises for a concrete outside query. Value-dependent eager validation is skipped while axes or queries are traced, so compiled callers own that
precondition. `fill_value` is static under `jax.jit`; the boundary policy and
grid rank are static too.
`bilinear_interp` and `trilinear_interp` broadcast coordinate arrays, stack
their final query axis, and call the same generic owner.

## What JAX differentiates

```{list-table} Regular-grid interpolation contracts
:header-rows: 1
:label: tbl-regular-grid-contracts

* - Operation
  - Contract
  - Supported claim
  - Boundary
* - Values at fixed axes and interior queries
  - `smooth_pathwise`
  - AD agrees with central finite differences for table values.
  - The active cell is fixed locally.
* - Interior query coordinates
  - `smooth_pathwise`
  - AD agrees with central finite differences inside one cell.
  - The query remains away from grid lines and boundaries.
* - Clamped or filled exterior coordinates
  - `known_zero`
  - Output is locally constant in the exterior query.
  - Saturation or sentinel selection is not inference.
* - Cell boundaries and axis locations
  - `validation_only`
  - Continuity and one-sided behavior can be checked.
  - `searchsorted` changes the active cell.
* - Reject validation
  - `validation_only`
  - Concrete invalid inputs can fail closed.
  - Value-dependent checks are skipped under tracing.
```

:::{warning}
Axis-location gradients are not claimed. Moving an axis changes both the local
scale and potentially the selected cell. At cell boundaries the value is
continuous, but the first derivative can change.
:::

## Using it in Jaxstro

```python
from jaxstro.jaxconfig import enable_high_precision

enable_high_precision()  # before creating JAX arrays

import jax.numpy as jnp

from jaxstro.numerics.regular_grid import bilinear_interp, regular_grid_interp

x_axis = jnp.array([0.0, 1.0, 3.0])
y_axis = jnp.array([-1.0, 2.0])
xx, yy = jnp.meshgrid(x_axis, y_axis, indexing="ij")
values = jnp.stack([3.0 * xx + yy, xx - 2.0 * yy], axis=-1)
xi = jnp.array([[0.25, 0.5], [2.5, 1.5]])

interpolated = regular_grid_interp((x_axis, y_axis), values, xi)
expected = jnp.stack(
    [3.0 * xi[:, 0] + xi[:, 1], xi[:, 0] - 2.0 * xi[:, 1]],
    axis=-1,
)
bilinear = bilinear_interp(
    x_axis, y_axis, values[..., 0], xi[:, 0], xi[:, 1]
)

outside = jnp.array([[-0.5, 0.5], [3.5, 1.0]])
clamped = regular_grid_interp(
    (x_axis, y_axis), values[..., 0], outside, boundary="clamp"
)
filled = regular_grid_interp(
    (x_axis, y_axis),
    values[..., 0],
    outside,
    boundary="fill",
    fill_value=-99.0,
)

assert jnp.allclose(interpolated, expected)
assert jnp.allclose(bilinear, expected[:, 0])
assert jnp.allclose(clamped, jnp.array([0.5, 10.0]))
assert jnp.array_equal(filled, jnp.array([-99.0, -99.0]))
```

## How to audit the result

Verify exact values at every grid node and exact recovery of a constant and an
affine payload. Within a chosen cell, compare AD coordinate and table-value
gradients to central finite differences. Test every boundary policy separately,
including whole-payload fill behavior and eager reject failure.

:::{figure} ../../10-theory/figures/regular-grid-contracts.webp
:name: fig-regular-grid-contracts
:alt: Unit-square interpolation query connected to four corners with measured bilinear weights, beside clamp and fill outputs across the grid boundary

The measured one-hot corner weights sum to one, while the boundary panel shows
the separate clamp and fill contracts. It is not a general error benchmark.
:::

The assertion-bearing map is in [](../../60-validation/validation.md).

## Where the claim stops

This primitive does not handle scattered data, triangulations, adaptive meshes,
missing-cell reconstruction, multidimensional monotonicity, or domain-specific
grid selection. Exact affine recovery does not bound error for a curved function
inside a coarse cell.

## Connected ideas

:::{seealso}
Connect tensor grids to
[](../../30-representations/fields/topology-and-discretization.md), conditioning
to [](../../10-foundations/models-and-computation/sensitivity-conditioning-identifiability.md),
signatures to [](../../50-api/approximation-integration/regular-grid.md), the
gradient taxonomy to [](../methods.md#gradient-contracts), and evidence to
[](../../60-validation/validation.md). One-dimensional shape-preserving methods
are in [](./interpolation.md).
:::
