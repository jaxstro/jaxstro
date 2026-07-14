---
title: Spatial indexing and neighbor contracts
description: >-
  Fixed-shape bins, candidate neighborhoods, and exact fixed-radius pairs with
  explicit overflow and topology boundaries.
---

## The question this method answers

How can a researcher avoid comparing every point with every other point while
still knowing whether a reported neighbor set is complete? Spatial indexing
first groups points into cells, gathers a bounded candidate neighborhood, and
then applies the physical distance criterion.

:::{important}
Candidate does not mean neighbor. A candidate set can contain false positives;
an exact fixed-radius set is complete only when its geometric preconditions hold
and `did_overflow` is false.
:::

## Before computation: what should be true?

Define the coordinate frame, boundary convention, cell size, cutoff, and fixed
capacities. For the exact query, `cell_size >= cutoff`, `Bcap` must hold every
searched cell, and `k_max` must hold every accepted neighbor per particle. The
current geometry is open and clamped, not periodic.

Morton binning requires `Nbins_per_dim` to be a positive power of two no larger
than 1024. Dense linear cells accept arbitrary positive dimensions and are the
owner used by exact fixed-radius gathering.

:::{warning}
Fixed output shapes can truncate information. `fill_bins` cannot certify full recall after capacity overflow because it returns no overflow flag. Use
`fill_bins_exact` and propagate its status when completeness matters.
:::

## Define the mathematical objects

A cell is a bounded spatial region assigned an integer ID. A neighborhood is a
set of nearby cells inspected for possible interactions. Topology describes
which cell or point identities are connected; it can change discontinuously
when a point crosses a cell face or distance cutoff.

A Morton or Z-order code interleaves integer coordinate bits into one integer
{cite:t}`Morton1966`. It often preserves locality in memory, but it is not a
distance metric. A mask marks which slots in a fixed-capacity array hold real
indices; a sentinel occupies unused slots. An overflow flag says that capacity
was insufficient and the stored set may be incomplete.

## Derive the method

If coordinate bits are $x_b,y_b,z_b\in\{0,1\}$, a three-dimensional Morton code
interleaves them as

```{math}
:label: eq-morton-interleave
m=\sum_{b=0}^{B-1}\left(x_b2^{3b}+y_b2^{3b+1}+z_b2^{3b+2}\right).
```

The exact fixed-radius target for focal point $i$ is

```{math}
:label: eq-fixed-radius-set
\mathcal{N}_i=\{j:0<\lVert x_i-x_j\rVert_2\le r_{\mathrm{cut}}\}.
```

With cubic cells at least as wide as the cutoff, any point in
$\mathcal{N}_i$ lies in the focal cell or one of its 26 adjacent cells. Gathering
that 27-cell stencil, masking invalid slots, and applying the exact distance test
therefore reproduces the brute-force set when neither cell nor neighbor capacity
overflows. The lower strict inequality excludes self and coincident points; the
upper inclusive inequality keeps points exactly on the cutoff.

## What the algorithm actually does

`assign_particles_to_bins` maps a symmetric cube to Morton IDs and clamps
off-box positions to boundary bins. `assign_to_cells_linear` produces dense
row-major IDs for arbitrary `(nx, ny, nz)`. Clamping keeps indices valid but does
not create periodic boundaries.

`fill_bins` deterministically retains `Bcap` hash-ranked members per cell.
`fill_bins_exact` stores the same fixed shape and also returns `did_overflow`.
`gather_candidates_from_bins`, stencil variants, and
`approx_knn_candidates` return bounded candidates that still require a physical
filter and recall audit.

`gather_pairs_within_radius` assigns dense cells, gathers a masked 27-cell
stencil, computes distances, and returns `(neighbors, mask, did_overflow)` with
shapes `(N, k_max)`, `(N, k_max)`, and `()`. It is exact only when
`did_overflow` is false and all stated preconditions hold. Its `dims=None` path
reads positions on the host and is eager-only; under `jit`, `dims` must be
provided explicitly as static grid structure. `k_max`, `Bcap`, dimensions, and
the concrete `cell_size >= cutoff` guard also determine traced structure or
host-side checks.

The result is exact only when `did_overflow` is false and those geometry and
capacity conditions all hold.

## What JAX differentiates

Cell assignment uses floor, clipping, integer encoding, sorting, masks, and
top-k selection. These are host-side, discrete preprocessing or discrete JAX
operations, not a smooth map from positions to neighbor identity. A JIT-compatible
spatial query does not thereby have a meaningful topology derivative.

Once a neighbor set is fixed, a downstream smooth distance, force, or density
kernel can be differentiated with respect to floating positions or values. That
conditional derivative excludes points where cell membership, cutoff inclusion,
ranking, or capacity status changes.

## Using it in Jaxstro

```python
import jax.numpy as jnp

from jaxstro.spatial import gather_pairs_within_radius

positions = jnp.array(
    [
        [0.0, 0.0, 0.0],
        [0.25, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.25, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0],
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

## How to audit the result

1. Compute all pairwise distances for a small cloud and compare every returned
   row with the brute-force set `0 < |x_i - x_j| <= cutoff`.
2. Include coincident points, exact-cutoff points, boundaries, and empty cells.
3. Check neighbor symmetry when the scientific relation should be symmetric.
4. Force cell overflow and neighbor overflow, then verify `did_overflow` changes.
5. Compare approximate candidate recall with brute-force neighbors on uniform,
   boundary-heavy, and clustered clouds.
6. Record boundary convention, cell size, dimensions, `Bcap`, and `k_max`.

:::{tip}
Treat `did_overflow == False` as evidence attached to a particular cloud and
configuration. Re-audit capacity when the population or spatial concentration
changes.
:::

:::{figure} ../../10-theory/figures/spatial-neighbor-contracts.webp
:name: fig-spatial-neighbor-contracts
:alt: Two-panel spatial-neighbor diagram comparing a grid candidate pool with exact cutoff-filtered neighbors

The left panel shows candidate false positives; the right applies the public
exact-radius predicate with no overflow. This fixture explains the contract but
is not a population-wide recall benchmark.
:::

## Where the claim stops

Morton locality does not imply physical distance. Candidate heuristics do not
guarantee exact k-nearest-neighbor recall. Fixed-radius exactness is conditional
on open clamped geometry, stencil coverage, and both capacities. None of these
queries defines periodic wrapping, differentiable topology, or a complete
many-body interaction model.

## Connected ideas

:::{seealso}
Review norms in
[](../../10-foundations/mathematical-objects/linear-algebra-language-of-change.md),
connect cells and topology to
[](../../30-representations/fields/topology-and-discretization.md), and use the
claim-boundary workflow in
[](../../40-workflows/reproducible-research/evidence-and-claim-boundaries.md).
The public owner map is [](../../50-api/discrete-space/spatial.md), and
quantitative evidence belongs in [](../../60-validation/validation.md).
:::
