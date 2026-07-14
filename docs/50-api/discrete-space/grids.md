---
title: Grids and conservative binning
---

# Grids and conservative binning

## Owner import path

`jaxstro.numerics.grids`

## Purpose

Positive logarithmic grids, bin centers, and overlap-conservative rebinning.

## Public records and callables

`log_grid`, `geometric_bin_edges`, `bin_centers`, `geometric_bin_centers`, and
`conservative_rebin`.

## Shape and dtype expectations

Edges and values are one-dimensional floating arrays with compatible lengths.
Logarithmic inputs must be positive and ordered.

## JAX transforms and AD classification

Array evaluation composes with JIT on fixed shapes. Bin-index and overlap route
changes are discrete derivative boundaries.

## Failure behavior

Concrete invalid edge order, positivity, or shape raises. Rebinning preserves
integrated totals only over the overlapping domain.

## Contract and evidence links

See [](../../20-methods/discrete-space/grids.md) and
[](../../60-validation/index.md).

## Canonical import example

```python
from jaxstro.numerics.grids import conservative_rebin
```
