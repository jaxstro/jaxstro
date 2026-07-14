---
title: Regular-grid interpolation
---

# Regular-grid interpolation

## Owner import path

`jaxstro.numerics.regular_grid`

## Purpose

Static-rank multilinear interpolation on tensor-product grids.

## Public records and callables

`regular_grid_interp(points, values, xi, boundary="clamp")`,
`bilinear_interp`, and `trilinear_interp`.

## Shape and dtype expectations

Each point axis is one-dimensional and strictly increasing. Grid axes occupy
the leading dimensions of `values`; trailing dimensions are payload axes.

## JAX transforms and AD classification

Evaluation composes with JIT and AD inside fixed cells. Coordinate derivatives
are claimed only for branch-stable interior queries.

## Failure behavior

Boundary policy is explicit: clamp, whole-payload fill, or eager reject. Reject
validation is skipped for value-dependent queries while traced.

## Contract and evidence links

See [](../../20-methods/approximation-integration/regular-grid.md) and the
generated [](../research-infrastructure/contracts.md).

## Canonical import example

```python
from jaxstro.numerics.regular_grid import regular_grid_interp
```
