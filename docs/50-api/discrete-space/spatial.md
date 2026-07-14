---
title: Spatial indexing and pairs
---

# Spatial indexing and pairs

## Owner import path

`jaxstro.spatial`

## Purpose

Morton indexing, fixed-capacity binning, approximate candidate gathering, and
exact fixed-radius pair mechanics.

## Public records and callables

`morton_encode_3d`, `morton_decode_3d`, `wyhash32`,
`assign_particles_to_bins`, `assign_to_cells_linear`, `fill_bins`,
`fill_bins_exact`, `gather_candidates_from_bins`,
`gather_candidates_with_stencil`, `gather_candidates_two_stencil`,
`approx_knn_candidates`, and `gather_pairs_within_radius`.

## Shape and dtype expectations

Positions have a trailing length-three axis. Indices and Morton codes are
integer arrays; capacities, stencil sizes, and output shapes are static.

## JAX transforms and AD classification

Fixed-shape values compose with JIT and VMAP, but sorting, bins, candidates,
overflow, and pair membership are discrete and have no AD claim.

## Failure behavior

Capacity and overflow policy are explicit in masks and result contracts.
Approximate candidates are not silently promoted to exact neighbor results.

## Contract and evidence links

See [](../../20-methods/discrete-space/spatial.md), the generated
[](../research-infrastructure/contracts.md), and [](../../60-validation/index.md).

## Canonical import example

```python
from jaxstro.spatial import gather_pairs_within_radius
```
