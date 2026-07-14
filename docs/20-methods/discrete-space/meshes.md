---
title: Structured 1D meshes
description: >-
  Cell and face geometry, finite-volume divergence, topology stencils, and
  conservative remapping for one-dimensional data.
---

## The question this method answers

How can a researcher represent one-dimensional cells and faces, compute a
finite-volume divergence, and remap cell averages while preserving an integrated
total? A mesh combines coordinate geometry with topology: which cells exist,
where their faces lie, and which cells are neighbors.

:::{important}
The current owner is Cartesian and one-dimensional. Here cell volumes equal
widths and face areas equal one; those are explicit geometry choices, not a
general curvilinear-mesh contract.
:::

## Before computation: what should be true?

Provide strictly increasing edges of shape `(n_cells + 1,)`. Cell values have
shape `(n_cells,)`; face fluxes have shape `(n_cells + 1,)`. Choose boundary
fluxes and decide whether a `-1` neighbor sentinel is appropriate for the
downstream stencil. For remapping, old values must be cell averages, not totals.

:::{warning}
`Mesh1D`, face geometry, divergence, and cell-to-face averaging do not fully
validate edge ordering or all mutual shapes. The conservative remap delegates
edge validation to the grid helper. Shape-compatible output is not proof that a
mesh was geometrically valid.
:::

## Define the mathematical objects

A cell is the interval between two adjacent edges. A face is an edge shared by
neighboring cells or lying on the domain boundary. In one Cartesian dimension,
cell $i$ has width and volume $\Delta x_i=x_{i+1/2}-x_{i-1/2}$, while each face
has unit area.

Topology records adjacency independently of distance. `cell_neighbors_1d`
returns left and right integer cell indices, using `-1` where a boundary has no
neighbor.

## Derive the method

Integrating the conservation law over cell $i$ gives the finite-volume
divergence

```{math}
:label: eq-finite-volume-divergence
(\nabla\cdot F)_i=\frac{F_{i+1/2}-F_{i-1/2}}{\Delta x_i}.
```

Multiplying by cell width and summing cancels every interior face flux:

```{math}
:label: eq-mesh-telescoping
\sum_i\Delta x_i(\nabla\cdot F)_i=F_{N+1/2}-F_{1/2}.
```

For old cell average $\bar{q}_i$, first form total
$Q_i=\bar{q}_i\Delta x_i$, conservatively transfer totals by overlap, then
divide by each new width:

```{math}
:label: eq-conservative-remap
\bar{q}'_j=\frac{1}{\Delta x'_j}
\sum_i Q_i\frac{\ell_{ji}}{\Delta x_i}.
```

When the new domain covers the old domain, this construction preserves
$\sum_j\bar{q}'_j\Delta x'_j=\sum_i\bar{q}_i\Delta x_i$.

## What the algorithm actually does

`structured_edges_1d(start, stop, n_cells=...)` returns `n_cells + 1` uniformly
spaced edges; `n_cells` is static because it sets array shape. `Mesh1D(edges)` is
an Equinox PyTree exposing computed `centers`, `widths`, and `volumes`.
`face_geometry_1d` returns positions and unit areas.

`cell_neighbors_1d(n_cells)` constructs integer arrays of shape `(n_cells,)`.
`cell_to_face_average(values)` copies the first and last cell values to boundary
faces and arithmetic-averages adjacent interior cells. This boundary copy is a
stencil convention, not a physical boundary condition.

`conservative_remap_1d` converts cell averages to integrated totals, calls
`conservative_rebin`, and divides by new widths.

## What JAX differentiates

For fixed valid edges, divergence, face averaging, and remapping are linear in
fluxes or cell averages. Their AD derivatives can be checked against the explicit
stencil or overlap matrix. Mesh edges are floating PyTree leaves, so JAX can
trace formulas through widths and overlaps, but edge-order changes and overlap
topology changes are piecewise boundaries.

Neighbor indices, `-1` sentinels, `n_cells`, and the number of faces are
discrete. AD does not differentiate mesh topology or array shape.

## Using it in Jaxstro

```python
import jax.numpy as jnp

from jaxstro.numerics.meshes import (
    Mesh1D,
    cell_neighbors_1d,
    conservative_remap_1d,
    divergence_1d,
)

edges = jnp.array([0.0, 0.5, 2.0, 4.0])
mesh = Mesh1D(edges)
face_flux = jnp.array([1.0, 3.0, 2.0, 5.0])
divergence = divergence_1d(face_flux, edges)
neighbors = cell_neighbors_1d(3)

old_edges = jnp.array([0.0, 1.0, 3.0])
old_averages = jnp.array([2.0, 4.0])
new_edges = jnp.array([0.0, 0.5, 2.0, 3.0])
remapped = conservative_remap_1d(old_edges, old_averages, new_edges)

assert jnp.allclose(
    jnp.sum(mesh.widths * divergence), face_flux[-1] - face_flux[0]
)
assert jnp.allclose(
    jnp.sum(remapped * jnp.diff(new_edges)),
    jnp.sum(old_averages * jnp.diff(old_edges)),
)
assert neighbors.left.tolist() == [-1, 0, 1]
assert neighbors.right.tolist() == [1, 2, -1]
```

## How to audit the result

1. Check edge ordering, array shapes, units, and the boundary convention.
2. Compare widths, centers, faces, and neighbor sentinels with a hand fixture.
3. Verify the telescoping flux identity on a nonuniform mesh.
4. Verify constant-field remapping and total conservation on non-aligned edges.
5. Compare remapped totals with the lower-level overlap matrix.
6. Compare AD with central differences in cell averages while edges stay fixed.

:::{tip}
Audit boundary fluxes separately from interior cancellation. Conservation can
fail scientifically because a boundary condition is wrong even when the
discrete divergence telescopes exactly.
:::

## Where the claim stops

This module does not own multidimensional connectivity, adaptive meshes,
curvilinear geometry, reconstruction, Riemann solvers, or physical boundary
conditions. Remapping preserves totals only over the shared domain and assumes
piecewise-constant old cell averages. Integer neighbors are not differentiable.

## Connected ideas

:::{seealso}
Build the linear stencil language in
[](../../10-foundations/mathematical-objects/linear-algebra-language-of-change.md),
connect geometry and topology through
[](../../30-representations/fields/topology-and-discretization.md), and follow
the evidence workflow in
[](../../40-workflows/reproducible-research/evidence-and-claim-boundaries.md).
The API is [](../../50-api/discrete-space/meshes.md), and validation evidence is
indexed from [](../../60-validation/validation.md).
:::
