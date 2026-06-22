---
title: Structured 1D meshes
description: >-
  Structured cell/face geometry, finite-volume stencil helpers, and
  conservative remapping for one-dimensional scientific grids.
---

`jaxstro.numerics.meshes` provides the smallest shared mesh layer that many
scientific workflows need before they simulate, remap, or conserve quantities.
The first slice is deliberately one-dimensional.

## Mesh Geometry

`structured_edges_1d(start, stop, n_cells=...)` constructs uniform cell edges.
`Mesh1D(edges)` exposes `centers`, `widths`, and `volumes`; in this Cartesian 1D
slice, volumes are widths.

`face_geometry_1d(edges)` returns face positions and unit face areas.
`cell_neighbors_1d(n_cells)` returns left/right neighbor indices with `-1`
sentinels at boundaries.

## Finite-Volume Helpers

`divergence_1d(face_flux, edges)` computes

```{math}
\frac{F_{i+1/2} - F_{i-1/2}}{\Delta x_i}.
```

`cell_to_face_average(values)` copies boundary values and averages neighboring
cell values to interior faces.

## Conservative Remap

`conservative_remap_1d(old_edges, old_cell_averages, new_edges)` treats inputs as
cell averages, converts them to integrated totals, calls the existing
`conservative_rebin` total-preserving helper, and converts back to new cell
averages.

Unit tests check conservation, constant-field preservation, boundary stencils,
and JAX transforms. Validation checks FD-vs-AD behavior with respect to input
cell averages.
