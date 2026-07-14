---
title: Grids and conservative bin transfer
description: >-
  Logarithmic coordinates, bin centers, and overlap-based transfer of integrated
  totals.
---

## The question this method answers

How can a researcher construct one-dimensional sample locations and move
integrated bin totals onto new edges without losing the total over the shared
domain? Grid geometry determines what each array entry represents and which
conservation statement is meaningful.

:::{important}
`conservative_rebin` interprets values as integrated bin totals, not pointwise
densities. Using the wrong interpretation changes the units and invalidates the
conservation claim.
:::

## Before computation: what should be true?

Edges must be one-dimensional and strictly increasing. `old_edges` must have
one more entry than `values`. Logarithmic grids also require positive endpoints
and a positive base other than one. Decide whether the new domain covers the
whole old domain or only an overlap.

:::{warning}
Shape failures are checked in Python. Monotonicity and positivity are checked for
concrete arrays, but value-dependent eager validation is skipped while traced.
Compiled callers still own ordered edges and valid logarithmic parameters.
:::

## Define the mathematical objects

A grid is an ordered coordinate set. Bin edges $e_0<\cdots<e_N$ define cells
$[e_i,e_{i+1}]$ with widths $\Delta e_i=e_{i+1}-e_i$. An arithmetic center is
$(e_i+e_{i+1})/2$; a geometric center is $\sqrt{e_ie_{i+1}}$ and requires
positive edges.

A per-bin total $v_i$ is an integral over cell $i$. Assuming that total is
uniformly distributed inside the cell gives density
$\rho_i=v_i/(e_{i+1}-e_i)$ for the purpose of overlap transfer.

## Derive the method

The length shared by new bin $j$ and old bin $i$ is

```{math}
:label: eq-grid-overlap
\ell_{ji}=\max\!\left(0,
\min(e'_{j+1},e_{i+1})-\max(e'_j,e_i)\right).
```

Integrating the old piecewise-constant density over each new cell gives

```{math}
:label: eq-conservative-rebin
v'_j=\sum_i v_i\frac{\ell_{ji}}{e_{i+1}-e_i}.
```

If the new edges cover the old domain exactly or more broadly, summing over $j$
partitions every old bin and therefore $\sum_jv'_j=\sum_iv_i$. If the domains
only partly overlap, the new sum equals only the old total inside that overlap.

## What the algorithm actually does

`log_grid(start, stop, num, base=10)` uses logarithms and `jnp.linspace`, with
output shape `(num,)`. `geometric_bin_edges` calls it with `n_bins + 1`.
`bin_centers` and `geometric_bin_centers` map $(N+1)$ edges to $N$ centers.
`num` and `n_bins` are Python integers that determine shape.

`conservative_rebin` forms a dense overlap matrix of shape
`(n_new, n_old)` and multiplies it by old piecewise-constant densities. New bins
outside the old domain receive zero. The computation scales with the product of
old and new bin counts; no sparse overlap search is used.

## What JAX differentiates

For fixed edges, conservative rebinning is a linear map of `values`, so AD gives
the exact overlap fractions. Grid endpoints and edges flow through `log`,
`linspace`, `minimum`, `maximum`, and division. Their derivatives are piecewise
and change when edges coincide or an overlap opens or closes. Treat edge topology
as fixed preprocessing unless that piecewise derivative is explicitly intended.

Integer counts and output shapes are static, not differentiable. Invalid traced
edges are not repaired by the runtime.

## Using it in Jaxstro

```python
import jax.numpy as jnp

from jaxstro.numerics.grids import (
    bin_centers,
    conservative_rebin,
    geometric_bin_edges,
)

old_edges = jnp.array([0.0, 1.0, 3.0])
values = jnp.array([2.0, 6.0])
new_edges = jnp.array([0.0, 0.5, 2.0, 3.0])
rebinned = conservative_rebin(old_edges, values, new_edges)

positive_edges = geometric_bin_edges(1.0, 100.0, n_bins=2)
centers = bin_centers(positive_edges)

assert jnp.allclose(rebinned, jnp.array([1.0, 4.0, 3.0]))
assert jnp.allclose(jnp.sum(rebinned), jnp.sum(values))
assert positive_edges.shape == (3,)
assert centers.shape == (2,)
```

## How to audit the result

1. State whether inputs are totals, averages, densities, or point samples.
2. Check edge ordering, shapes, units, and old/new domain coverage.
3. Hand-compute the overlap matrix for a small non-aligned fixture.
4. Compare the transferred total with the old total over the shared domain.
5. Check that bins outside the old domain receive zero.
6. Compare AD with the known linear overlap matrix for fixed edges.

:::{tip}
Include both a full-domain conservation fixture and a partial-overlap fixture.
The latter prevents an incorrect blanket claim that every rebin preserves the
entire old total.
:::

## Where the claim stops

Overlap transfer assumes a piecewise-constant density inside each old bin. It is
not a point-sample interpolator or a higher-order reconstruction. Conservation
is bounded to the shared domain. Sobol and Halton sequences remain on the
separate planned quasi-Monte-Carlo route and are not importable here.

## Connected ideas

:::{seealso}
Review units and sampled functions in
[](../../10-foundations/mathematical-objects/functions-units-scales.md), connect
grids to [](../../30-representations/fields/topology-and-discretization.md), and
use preprocessing audits from
[](../../40-workflows/scientific-ml/preprocessing.md).
The API is [](../../50-api/discrete-space/grids.md), and evidence routes through
[](../../60-validation/validation.md).
:::
