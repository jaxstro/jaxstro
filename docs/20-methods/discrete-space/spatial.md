---
title: Spatial indexing and neighbor contracts
description: >-
  How jaxstro turns positions into fixed-shape spatial bins, approximate
  candidate pools, and exact fixed-radius neighbors without hiding overflow or
  differentiability boundaries.
---

Spatial acceleration begins with a change of question. Instead of comparing
every particle with every other particle, first ask which particles *could* be
nearby. That first answer is a candidate pool. It becomes an exact geometric
answer only after an explicit distance filter and a successful capacity check.

:::{admonition} Observable first: what did the query return?
:class: important

- A **candidate query** returns indices worth checking. It may include distant
  particles and can lose recall if a fixed-capacity bin overflows.
- An **exact fixed-radius query** returns every index satisfying its distance
  contract only when `did_overflow` is false.
:::

## Learning objectives

After this chapter, you should be able to separate candidate recall from exact
geometry, interpret capacity and overflow evidence, and identify why topology
construction is a discrete boundary rather than a differentiable interaction.

### Concept check: candidate is not neighbor

Predict which grid candidates fail the exact radius test. Compute both sets,
then audit recall, symmetry, cutoff inclusion, and overflow independently.

## The pipeline

```text
positions
   -> host-side cell assignment and ordering
fixed-capacity cells or Morton bins
   -> stencil gather
candidate pool
   -> distance filter + overflow check
exact fixed-radius neighbors
```

The boxes and masks in this pipeline have static shapes that compose well with
JAX. Choosing bins, sorting integer IDs, truncating capacity, and deciding which
stencil to inspect remain host-side, discrete preprocessing rather than a
smooth map from positions to neighbor identity.

## Morton bins: locality, not distance

A Morton or Z-order code interleaves the bits of integer cell coordinates into
one integer {cite:t}`Morton1966`. Nearby grid cells often remain near one another
in that ordering, which is useful for grouping and memory access. The code is
not a metric: subtracting two Morton IDs does not give a physical distance.

`assign_particles_to_bins(...)` maps a symmetric cubic box to Morton-ordered
bins. Its `Nbins_per_dim` contract is a positive power of two, no larger than
1024. This restriction makes the Morton IDs dense in
`[0, Nbins_per_dim**3)`. For arbitrary dense dimensions, use
`assign_to_cells_linear(...)`, which owns row-major cell IDs rather than Morton
ordering.

Particles outside the configured box clamp to boundary bins. Clamping keeps
indices valid; it does not create periodic boundaries.

## Capacity is part of the result

JAX kernels benefit from fixed shapes, so a cell array has a configured
capacity `Bcap` and unused slots carry a sentinel plus a Boolean mask.

- `fill_bins(...)` deterministically selects `Bcap` members when a bin is too
  full, but it returns no overflow flag. `fill_bins` cannot certify full recall after capacity overflow.
- `fill_bins_exact(...)` returns `did_overflow` along with the members and mask.
  "Exact" here means exact grouping when capacity is sufficient; it does not
  make a too-small `Bcap` lossless.

Never infer completeness from a full-looking fixed-shape array. Propagate the
overflow result or choose a configuration whose capacity is independently
validated for the point distribution.

## Candidate does not mean neighbor

`gather_candidates_from_bins(...)`,
`gather_candidates_with_stencil(...)`,
`gather_candidates_two_stencil(...)`, and
`approx_knn_candidates(...)` search nearby grid cells and return bounded
candidate arrays. The arrays can contain false positives because a cell stencil
covers a box-like region rather than the final physical criterion. Their
project-local keep counts and fallback thresholds are fixed-shape heuristics,
not a proof of exact k-nearest-neighbor recall for every distribution.

A consumer should:

1. mask sentinels;
2. compute the needed distances or other physical predicate;
3. apply the final filter or ranking; and
4. validate recall on representative uniform, boundary, and clustered clouds.

:::{figure} ../../10-theory/figures/spatial-neighbor-contracts.webp
:name: fig-spatial-neighbor-contracts
:alt: Two-panel spatial-neighbor diagram comparing a grid candidate pool with exact cutoff-filtered neighbors

The left panel is the real candidate set returned for focal particle 0 in the
JaxtroViz fixture; orange edges include false positives. The right panel applies
the public exact-radius query to the same positions. Only particles 1 and 2 meet
the cutoff, and the displayed overflow state is false. This one cloud explains
the contract; it is not a population-wide recall benchmark.
:::

## Exact fixed-radius neighbors

`gather_pairs_within_radius(...)` owns a stronger contract:

```text
{j : 0 < |x_i - x_j| <= cutoff}
```

The lower bound excludes self-indices and coincident particles. The upper bound
includes a pair exactly on the cutoff. The query is exact only when `did_overflow` is false.
Its other preconditions are load-bearing:

- `cell_size >= cutoff`, so the 27-cell stencil covers the cutoff sphere;
- `k_max` can hold every accepted neighbor per particle;
- `Bcap` can hold every member of a searched cell; and
- under `jit`, `dims` is supplied explicitly as static grid structure.

The current implementation uses open, clamped boundaries rather than periodic
wrapping. An exact result is exact for that stated geometry, not for a different
boundary convention.

## Worked exact-radius example

This standalone example checks three easy-to-confuse boundaries: a particle at
exactly `cutoff` is included, a coincident particle is excluded, and capacity is
sufficient.

```python
import jax.numpy as jnp

from jaxstro.spatial import gather_pairs_within_radius

positions = jnp.array(
    [
        [0.0, 0.0, 0.0],
        [0.25, 0.0, 0.0],
        [0.5, 0.0, 0.0],   # exactly at cutoff
        [1.25, 0.0, 0.0],
        [0.0, 0.5, 0.0],   # exactly at cutoff
        [0.0, 0.0, 0.0],   # coincident: excluded by r > 0
    ]
)

neighbors, mask, did_overflow = gather_pairs_within_radius(
    positions,
    origin=jnp.array([0.0, 0.0, 0.0]),
    cell_size=0.5,
    cutoff=0.5,
    k_max=5,
    Bcap=6,
    dims=(4, 2, 2),
)

focal_neighbors = set(map(int, neighbors[0][mask[0]].tolist()))
assert focal_neighbors == {1, 2, 4}
assert not bool(did_overflow)
```

The result is a discrete index set. Differentiating downstream force or density
kernels with respect to values at a *fixed* neighbor set may be meaningful;
differentiating through the change in neighbor identity is not the contract of
this preprocessing layer.

## Choosing the public path

```{list-table} Spatial query decision table
:header-rows: 1
:label: tbl-spatial-query-choice

* - Need
  - Public path
  - Required check
* - Morton-ordered grouping on a cubic power-of-two grid
  - `assign_particles_to_bins` -> `fill_bins` or `fill_bins_exact`
  - Capacity and boundary-clamping policy
* - Arbitrary dense grid dimensions
  - `assign_to_cells_linear` -> `fill_bins_exact`
  - Dense cell-count and capacity bounds
* - Bounded approximate kNN candidates
  - `approx_knn_candidates`
  - Recall against exact kNN on representative clouds
* - Exact open-boundary fixed-radius neighbors
  - `gather_pairs_within_radius`
  - Preconditions plus `did_overflow == False`
```

The public ownership map is in [](../../50-api/discrete-space/spatial.md), and quantitative evidence
belongs in [](../../60-validation/validation.md).
