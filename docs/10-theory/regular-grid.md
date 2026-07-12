---
title: Regular-grid interpolation
description: >-
  Static-rank multilinear interpolation for gridded scientific tables with
  explicit boundary and differentiation contracts.
---

Regular-grid interpolation is the table primitive for data that live on a
Cartesian product of one-dimensional axes. Atmosphere grids, calibration
surfaces, and tracks often have this shape: each coordinate axis is sorted, and
the value array stores samples on the tensor grid.

This complete example builds a two-component affine table. Multilinear
interpolation must recover both components exactly because an affine function
has no unresolved curvature inside a cell.

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

The leading dimensions of `values` correspond to the grid axes. Every trailing
dimension is payload, so one interpolation call can return a scalar, vector,
matrix, spectrum, or a batch of diagnostics. `bilinear_interp(...)` and
`trilinear_interp(...)` assemble two- and three-dimensional queries before
calling the same generic kernel.

:::{figure} ./figures/regular-grid-contracts.webp
:name: fig-regular-grid-contracts
:alt: Unit-square interpolation query connected to four corners with measured bilinear weights, beside clamp and fill outputs across the grid boundary

The left panel obtains each weight by interpolating a one-hot corner table with
the public bilinear API; the measured weights sum to one. The right panel sends
one affine slice through the public clamp and fill policies. This is a contract
visualization for one executable fixture, not a general interpolation-error
benchmark.
:::

## Learning objectives

After this chapter, you should be able to derive multilinear corner weights,
separate boundary policy from interpolation arithmetic, and identify which
preparation steps remain discrete under JAX transformations.

### Concept check: outside the table

Predict the value and coordinate gradient just outside a grid for clamp, fill,
and reject policies. Compute each policy, then audit whether the observed
gradient follows the documented policy rather than an imagined extrapolation.

## Boundary policy

The boundary policy is explicit:

- `boundary="clamp"` clips each query coordinate to the nearest grid endpoint.
- `boundary="fill"` returns `fill_value` for the complete payload whenever any
  coordinate lies outside the domain.
- `boundary="reject"` raises eagerly for a concrete outside query.

`reject` is an eager/debug guard. Value-dependent eager validation is skipped while axes or queries are traced, because a Python exception cannot depend on an
unknown traced value. Compiled callers that require rejection must validate the
axes and query domain before entering `jax.jit`. The `fill_value` is static under `jax.jit`; the boundary string is static too. Changing either value selects a
distinct compiled program.

The fill sentinel is a numerical policy, not missing-data semantics. A downstream
model must decide whether a finite sentinel, `NaN`, masking, or prior-domain
restriction is scientifically appropriate.

## Multilinear weights

For each dimension, the evaluator finds the enclosing interval and computes a
fractional coordinate:

```{math}
t_d = \frac{x_d - a_{d,i_d}}{a_{d,i_d+1} - a_{d,i_d}}.
```

The value is the weighted sum of all `2^D` corners of that cell:

```{math}
F(\boldsymbol{x}) =
\sum_{\boldsymbol{b}\in\{0,1\}^D}
f_{\boldsymbol{i}+\boldsymbol{b}}
\prod_{d=1}^{D}
t_d^{b_d}(1-t_d)^{1-b_d}.
```

This is the rectangular-table multilinear interpolant described by
{cite:t}`WeiserZarantonello1988`. In two dimensions the four products are the
familiar bilinear corner weights; in three dimensions there are eight trilinear
weights. The current implementation makes grid rank static so JAX sees a fixed
corner sum during tracing.

## Differentiation and execution boundaries

Regular-grid interpolation does not have one universal gradient contract:

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
  - The active cell and boundary policy are fixed locally.
* - Interior query coordinates
  - `smooth_pathwise`
  - AD agrees with central finite differences inside one cell.
  - The query remains away from grid lines and domain boundaries.
* - Clamped or filled exterior coordinates
  - `known_zero`
  - The selected output is locally constant with respect to an exterior query.
  - This is saturation or sentinel selection, not an inference direction.
* - Cell boundaries and axis locations
  - `validation_only`
  - Continuity, exact grid values, and named one-sided behavior can be checked.
  - `searchsorted` changes the active cell; no universal derivative is claimed.
* - Reject validation
  - `validation_only`
  - Concrete eager inputs can fail closed on invalid axes or outside queries.
  - Value-dependent checks are skipped under tracing; compiled callers own the
    precondition.
```

The validation suite compares AD with finite differences for both table values
and interior query coordinates. Axis-location gradients are not part of that
claim: changing an axis can change interval membership as well as the local
coordinate scale. At cell boundaries the interpolant is continuous, but its
first derivative can change between adjacent cells.

## What this does not do

This primitive assumes tensor-product grid structure. It does not handle
scattered data, triangulations, adaptive meshes, multidimensional monotonicity
constraints, missing-cell reconstruction, or domain-specific grid selection.
Those policies belong in higher-level packages or future, separately validated
modules.

## From explanation to evidence

For signatures and payload ownership, see
[](../40-api/index.md#jaxstro-numerics-regular-grid). The assertion-bearing test
map is in [](../60-validation/index.md). The five package-wide differentiation
labels are defined in [](./index.md#gradient-contracts).
