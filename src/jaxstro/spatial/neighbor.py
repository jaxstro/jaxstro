"""
Approximate neighbor candidate gathering using spatial bins.

This module provides algorithms to efficiently identify candidate neighbors
for each particle without exhaustive O(N^2) pairwise distance calculations.
The approach uses a stencil of nearby grid cells to gather potential neighbors,
then filters by distance to produce an approximate candidate set.

**Key insight**: These functions return *candidate* neighbors, not exact k-NN.
The candidates are guaranteed to include all true neighbors within the stencil
radius, but may include extras. Downstream algorithms (e.g., integrators,
density estimators) should perform final distance filtering as needed.

Stencil strategies:
    - **3x3x3 = 27 cells**: Fast, covers ~1.5 bin widths in each direction
    - **5x5x5 = 125 cells**: Denser coverage for thin or clustered regions
    - **Two-stencil adaptive**: Uses coarse stencil by default, falls back
      to dense stencil for particles with insufficient coarse candidates

NOTE: This module was originally developed to support neighbor-list
construction in a direct N-body integrator. The algorithms are now refactored
to be backend-agnostic and live in `jaxstro.spatial`. Downstream projects
(e.g., N-body codes, IC builders) should import from `jaxstro.spatial.*`.

TODO: Use r_search and dx to cull candidates outside a physical/softening
radius, especially for applications where the interaction radius is known.

TODO: Add optional periodic boundary conditions when computing distances,
for use in cosmological boxes. Currently assumes simple Euclidean distances.

TODO(jaxstro): Add a higher-level `approx_knn` function that returns both
candidate indices and distances, and a small layer on top to support:
    - fixed-k neighbors (kNN)
    - neighbors within a given radius
to be consumed by N-body integrators, IC builders, and analysis code.

Provenance:
    The general idea of grid-/cell-based neighbour search and of maintaining a
    neighbour list to avoid the O(N^2) all-pairs cost is standard in collisional
    N-body work (see the references below). HOWEVER, the specific design choices
    in this module are ORIGINAL heuristics for this codebase, NOT taken from any
    paper, and the magic numbers are tuned, not derived:

      - the **two-stencil adaptive fallback** (coarse 3x3x3, then dense 5x5x5
        only for particles whose coarse pool is thin),
      - the dense-fallback trigger threshold ``2 * K_target``,
      - the per-bin keep counts ``K_bin_coarse = 18`` and ``K_bin_dense = 2``,
      - and the pool caps ``384`` (coarse) and ``512`` (dense), chosen for JIT
        shape stability, not from theory.

    Treat all of the above as project-local heuristics: they are reasonable
    defaults for roughly uniform-to-clustered point sets, but have not been
    validated against a literature benchmark. The cited papers motivate the
    neighbour-list *concept*, not these particular constants.

References:
    Makino, J. & Aarseth, S. J. (1992). PASJ, 44, 141.  (neighbour-scheme concept)
    Ahmad, A. & Cohen, L. (1973). J. Comp. Phys., 12, 389.  (Ahmad-Cohen lists)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from jaxstro.spatial.morton import MAX_BITS_3D, morton_decode_3d, morton_encode_3d

# =============================================================================
# Core Candidate Gathering: Fixed Stencil
# =============================================================================


def gather_candidates_from_bins(
    pos: Float[Array, "Nplus1 3"],
    bin_members: Int[Array, "Nbins Bcap"],
    bin_mask: Bool[Array, "Nbins Bcap"],
    bin_of: Int[Array, "N"],
    r_search: Float[Array, "N"],
    Nbins_per_dim: int,
    dx: float,
    Cand_max: int,
    K_bin: int,
) -> tuple[Int[Array, "N Cand_max"], Bool[Array, "N Cand_max"]]:
    """
    Gather approximate neighbor candidates from 27 neighboring bins (3x3x3 stencil).

    For each particle, this function:
    1. Identifies the 27 bins in the 3x3x3 neighborhood of the particle's bin
    2. Gathers all particles from those bins (up to Bcap per bin)
    3. Keeps the K_bin closest candidates per bin (by distance)
    4. Returns the top Cand_max overall candidates

    This is an *approximate* candidate set. All true neighbors within ~1.5 bin
    widths are guaranteed to be included, but the set may contain extras.

    Args:
        pos: Particle positions [N+1, 3]. The last row (index N) is a sentinel
            position used for safe indexing of empty bin slots.
        bin_members: Particle IDs in each bin [Nbins, Bcap]. Empty slots contain
            the sentinel value N.
        bin_mask: Valid slots [Nbins, Bcap]. True = valid particle, False = sentinel.
        bin_of: Bin assignment per particle [N], from assign_particles_to_bins.
        r_search: Search radius per particle [N]. Currently unused; reserved for
            future radius-based culling.
        Nbins_per_dim: Bins per dimension in the grid.
        dx: Bin size (L_box / Nbins_per_dim). Currently unused; reserved for
            future radius-based culling.
        Cand_max: Maximum candidates to return per particle.
        K_bin: Candidates to keep per bin before final selection. Must satisfy
            K_bin <= Bcap and Cand_max <= 27 * K_bin.

    Returns:
        cand_idx: Candidate particle indices [N, Cand_max]. Invalid slots contain N.
        cand_mask: Valid candidates [N, Cand_max]. True = valid, False = sentinel.

    Performance:
        O(N x 27 x Bcap) distance calculations, reduced by K_bin pre-filtering.
    """
    # Delegate to the general stencil function with width=3
    return gather_candidates_with_stencil(
        pos=pos,
        bin_members=bin_members,
        bin_mask=bin_mask,
        bin_of=bin_of,
        r_search=r_search,
        Nbins_per_dim=Nbins_per_dim,
        dx=dx,
        Cand_max=Cand_max,
        K_bin=K_bin,
        stencil_width=3,
    )


def gather_candidates_with_stencil(
    pos: Float[Array, "Nplus1 3"],
    bin_members: Int[Array, "Nbins Bcap"],
    bin_mask: Bool[Array, "Nbins Bcap"],
    bin_of: Int[Array, "N"],
    r_search: Float[Array, "N"],
    Nbins_per_dim: int,
    dx: float,
    Cand_max: int,
    K_bin: int,
    stencil_width: int = 3,
) -> tuple[Int[Array, "N Cand_max"], Bool[Array, "N Cand_max"]]:
    """
    Gather approximate neighbor candidates from (stencil_width^3) neighboring bins.

    Generalized version supporting configurable stencil sizes:
        - stencil_width=3: 27-cell (3x3x3), fast, ~1.5 bin widths coverage
        - stencil_width=5: 125-cell (5x5x5), denser, ~2.5 bin widths coverage

    The algorithm:
    1. Decode particle bin indices to (x, y, z) coordinates
    2. Generate all bin offsets in the stencil
    3. Gather particles from stencil bins
    4. Compute distances, keep K_bin closest per bin
    5. Return top Cand_max overall by distance

    Args:
        pos: Particle positions [N+1, 3] including sentinel at index N.
        bin_members: Particle IDs per bin [Nbins, Bcap], sentinel=N for empty.
        bin_mask: Valid slots [Nbins, Bcap].
        bin_of: Bin assignment per particle [N].
        r_search: Search radius per particle [N]. Currently unused.
        Nbins_per_dim: Bins per dimension.
        dx: Bin size. Currently unused.
        Cand_max: Maximum candidates to return per particle.
        K_bin: Candidates to keep per bin (pre-filter).
        stencil_width: Stencil size (3 for 27-cell, 5 for 125-cell, etc.)

    Returns:
        cand_idx: Candidate indices [N, Cand_max], sentinel=N for empty slots.
        cand_mask: Valid mask [N, Cand_max].

    Raises:
        AssertionError: If K_bin > Bcap or Cand_max > n_cells * K_bin
    """
    N = pos.shape[0] - 1  # Last row is sentinel
    Bcap = bin_members.shape[1]
    n_cells = stencil_width**3

    # Compile-time checks
    assert K_bin <= Bcap, f"K_bin ({K_bin}) must be <= Bcap ({Bcap})"
    assert Cand_max <= n_cells * K_bin, (
        f"Cand_max ({Cand_max}) must be <= {n_cells}*K_bin ({n_cells * K_bin})"
    )

    # Decode bin indices to (x, y, z)
    bx, by, bz = morton_decode_3d(bin_of, bits=MAX_BITS_3D, out_dtype=jnp.int32)

    # Generate neighbor bin offsets for stencil_width
    # For width=3: [-1, 0, 1]; for width=5: [-2, -1, 0, 1, 2]
    half = stencil_width // 2
    offs = jnp.arange(-half, half + 1, dtype=jnp.int32)  # shape: (stencil_width,)

    # Broadcast to get all stencil_width^3 combinations
    nbx = jnp.clip(
        bx[:, None] + offs[None, :], 0, Nbins_per_dim - 1
    )  # [N, stencil_width]
    nby = jnp.clip(by[:, None] + offs[None, :], 0, Nbins_per_dim - 1)
    nbz = jnp.clip(bz[:, None] + offs[None, :], 0, Nbins_per_dim - 1)

    # Cartesian product via broadcasting
    nbxg, nbyg, nbzg = jnp.broadcast_arrays(
        nbx[:, :, None, None],  # [N, stencil_width, 1, 1]
        nby[:, None, :, None],  # [N, 1, stencil_width, 1]
        nbz[:, None, None, :],  # [N, 1, 1, stencil_width]
    )

    # Flatten to [N, n_cells]
    neighbor_xyz = jnp.stack(
        [
            nbxg.reshape(N, n_cells),
            nbyg.reshape(N, n_cells),
            nbzg.reshape(N, n_cells),
        ],
        axis=-1,
    )  # [N, n_cells, 3]

    # Morton encode to bin IDs
    neighbor_bins = morton_encode_3d(neighbor_xyz.reshape(-1, 3), bits=MAX_BITS_3D)
    neighbor_bins = neighbor_bins.reshape(N, n_cells)  # [N, n_cells]

    # Gather members from bins (sentinel=N in empty slots)
    members = bin_members[neighbor_bins]  # [N, n_cells, Bcap]
    masks = bin_mask[neighbor_bins]  # [N, n_cells, Bcap]

    # Compute distances to all candidates
    p = pos[:N]  # [N, 3]
    q = pos[members]  # [N, n_cells, Bcap, 3] -- safe: sentinel=N exists in pos

    r2 = jnp.sum((p[:, None, None, :] - q) ** 2, axis=-1)  # [N, n_cells, Bcap]

    # Mask invalid and self
    is_self = members == jnp.arange(N, dtype=members.dtype)[:, None, None]
    valid = masks & ~is_self
    r2 = jnp.where(valid, r2, jnp.inf)

    # Per-bin top-K_bin by distance (ascending r2 -> descending -r2 for top_k)
    r2_for_topk = -r2  # Closest = most negative
    _, top_per_bin = jax.lax.top_k(r2_for_topk, K_bin)  # [N, n_cells, K_bin]

    # Gather selected indices and masks
    idx_per_bin = jnp.take_along_axis(
        members, top_per_bin, axis=2
    )  # [N, n_cells, K_bin]
    mask_per_bin = jnp.take_along_axis(valid, top_per_bin, axis=2)

    # Flatten to [N, n_cells*K_bin]
    idx_flat = idx_per_bin.reshape(N, n_cells * K_bin)
    mask_flat = mask_per_bin.reshape(N, n_cells * K_bin)

    # Final top-Cand_max by distance
    qf = pos[idx_flat]  # [N, n_cells*K_bin, 3]
    r2f = jnp.sum((p[:, None, :] - qf) ** 2, axis=-1)  # [N, n_cells*K_bin]

    # Score: valid candidates get -r2, invalid get -inf
    score = jnp.where(mask_flat, -r2f, -jnp.inf)

    # Top-Cand_max
    _, top_final = jax.lax.top_k(score, Cand_max)

    cand_idx = jnp.take_along_axis(idx_flat, top_final, axis=1)
    cand_mask = jnp.isfinite(jnp.take_along_axis(score, top_final, axis=1))

    # Pad with sentinel N
    cand_idx = jnp.where(cand_mask, cand_idx, N)

    return cand_idx, cand_mask


# =============================================================================
# Adaptive Two-Stencil Gathering
# =============================================================================


def gather_candidates_two_stencil(
    pos: Float[Array, "Nplus1 3"],
    bin_members: Int[Array, "Nbins Bcap"],
    bin_mask: Bool[Array, "Nbins Bcap"],
    bin_of: Int[Array, "N"],
    r_search: Float[Array, "N"],
    Nbins_per_dim: int,
    dx: float,
    K_target: int,
    K_bin_coarse: int = 18,
    K_bin_dense: int = 2,
) -> tuple[Int[Array, "N Cand_max"], Bool[Array, "N Cand_max"]]:
    """
    Adaptive two-stencil candidate gathering for robust coverage.

    This function addresses the "thin candidate pool" problem where some
    particles (especially in sparse or clustered regions) have too few
    candidates from the standard 27-cell stencil. The strategy:

    1. **Coarse pass** (27-cell, 3x3x3): Fast, typically sufficient for 50-70%
       of particles in uniform distributions.

    2. **Dense fallback** (125-cell, 5x5x5): For particles with < 2*K_target
       candidates from the coarse pass, use the denser stencil.

    3. **Per-particle selection**: Each particle uses either coarse or dense
       results based on its coarse candidate count.

    This ensures every particle has a sufficient candidate pool for downstream
    neighbor selection, fixing asymmetry issues caused by thin pools.

    Args:
        pos: Particle positions [N+1, 3] including sentinel at index N.
        bin_members: Particle IDs per bin [Nbins, Bcap].
        bin_mask: Valid slots [Nbins, Bcap].
        bin_of: Bin assignment per particle [N].
        r_search: Search radius per particle [N]. Currently unused.
        Nbins_per_dim: Bins per dimension.
        dx: Bin size. Currently unused.
        K_target: Target neighbor count. Used to determine the threshold
            (2*K_target) below which dense fallback is triggered.
        K_bin_coarse: Candidates per bin for 27-cell stencil (default: 18).
        K_bin_dense: Candidates per bin for 125-cell stencil (default: 2).
            Lower value due to more cells (125 vs 27).

    Returns:
        cand_idx: Candidate indices [N, Cand_max] where
            Cand_max = max(coarse_capacity, dense_capacity).
        cand_mask: Valid candidates [N, Cand_max].

    Note:
        The adaptive threshold (2*K_target) triggers dense fallback for
        ~30-50% of particles in typical clustered systems, ensuring all
        particles have adequate coverage.
    """
    N = pos.shape[0] - 1

    # Capacity for each stencil
    # 27-cell: 27*K_bin_coarse pool, capped at 384 for JIT performance
    # 125-cell: 125*K_bin_dense pool, capped at 512 for JIT performance
    Cand_coarse = min(27 * K_bin_coarse, 384)
    Cand_dense = min(125 * K_bin_dense, 512)
    Cand_max = max(Cand_coarse, Cand_dense)  # Use larger capacity for output

    # Coarse pass: 3x3x3 = 27 cells
    cand_coarse, mask_coarse = gather_candidates_with_stencil(
        pos=pos,
        bin_members=bin_members,
        bin_mask=bin_mask,
        bin_of=bin_of,
        r_search=r_search,
        Nbins_per_dim=Nbins_per_dim,
        dx=dx,
        Cand_max=Cand_coarse,
        K_bin=K_bin_coarse,
        stencil_width=3,
    )

    # Dense pass: 5x5x5 = 125 cells
    cand_dense, mask_dense = gather_candidates_with_stencil(
        pos=pos,
        bin_members=bin_members,
        bin_mask=bin_mask,
        bin_of=bin_of,
        r_search=r_search,
        Nbins_per_dim=Nbins_per_dim,
        dx=dx,
        Cand_max=Cand_dense,
        K_bin=K_bin_dense,
        stencil_width=5,
    )

    # Count candidates from coarse pass
    n_coarse = jnp.sum(mask_coarse, axis=1)  # [N]

    # Adaptive threshold: use dense for particles with thin coarse pools
    # Threshold of 2*K_target ensures adequate candidate pool for neighbor selection
    threshold = 2 * K_target
    use_dense = n_coarse < threshold

    # Pad arrays to Cand_max if needed (for consistent shape)
    if Cand_coarse < Cand_max:
        pad_coarse = Cand_max - Cand_coarse
        cand_coarse = jnp.pad(cand_coarse, ((0, 0), (0, pad_coarse)), constant_values=N)
        mask_coarse = jnp.pad(
            mask_coarse, ((0, 0), (0, pad_coarse)), constant_values=False
        )

    if Cand_dense < Cand_max:
        pad_dense = Cand_max - Cand_dense
        cand_dense = jnp.pad(cand_dense, ((0, 0), (0, pad_dense)), constant_values=N)
        mask_dense = jnp.pad(
            mask_dense, ((0, 0), (0, pad_dense)), constant_values=False
        )

    # Per-particle selection
    cand_idx = jnp.where(use_dense[:, None], cand_dense, cand_coarse)
    cand_mask = jnp.where(use_dense[:, None], mask_dense, mask_coarse)

    return cand_idx, cand_mask


# =============================================================================
# High-Level Convenience API
# =============================================================================


def approx_knn_candidates(
    pos: Float[Array, "Nplus1 3"],
    bin_members: Int[Array, "Nbins Bcap"],
    bin_mask: Bool[Array, "Nbins Bcap"],
    bin_of: Int[Array, "N"],
    Nbins_per_dim: int,
    *,
    K_target: int,
    use_two_stencil: bool = True,
    K_bin_coarse: int = 18,
    K_bin_dense: int = 2,
) -> tuple[Int[Array, "N Cand_max"], Bool[Array, "N Cand_max"]]:
    """
    High-level wrapper to get an approximate candidate set for kNN queries.

    This is a convenience API over `gather_candidates_two_stencil` (adaptive)
    or `gather_candidates_from_bins` (fixed 27-cell), providing sensible
    defaults for common use cases.

    Intended applications:
        - N-body neighbor lists (force calculation, block timesteps)
        - Local surface density estimators (Sigma_k)
        - IC analysis (mass segregation diagnostics, fractal clustering)
        - Cluster membership and structure analysis

    Args:
        pos: Particle positions [N+1, 3] including sentinel at index N.
        bin_members: Particle IDs per bin [Nbins, Bcap].
        bin_mask: Valid slots [Nbins, Bcap].
        bin_of: Bin assignment per particle [N].
        Nbins_per_dim: Bins per dimension.
        K_target: Target number of neighbors. Used to size the output and
            determine dense fallback threshold (if use_two_stencil=True).
        use_two_stencil: If True (default), use adaptive two-stencil
            gathering for robust coverage. If False, use fixed 27-cell.
        K_bin_coarse: Candidates per bin for 27-cell stencil.
        K_bin_dense: Candidates per bin for 125-cell stencil.

    Returns:
        cand_idx: Candidate particle indices [N, Cand_max].
        cand_mask: Valid candidates [N, Cand_max].

    Example:
        >>> # After building spatial grid
        >>> cand_idx, cand_mask = approx_knn_candidates(
        ...     pos_with_sentinel,
        ...     bin_members,
        ...     bin_mask,
        ...     bin_of,
        ...     Nbins_per_dim=16,
        ...     K_target=32,
        ... )
        >>> # Further process candidates to select exact k-NN
    """
    N = pos.shape[0] - 1

    # Placeholder for r_search and dx (not yet used, but keep for future)
    r_search = jnp.zeros(N)  # Unused placeholder
    dx = 1.0  # Unused placeholder

    if use_two_stencil:
        return gather_candidates_two_stencil(
            pos=pos,
            bin_members=bin_members,
            bin_mask=bin_mask,
            bin_of=bin_of,
            r_search=r_search,
            Nbins_per_dim=Nbins_per_dim,
            dx=dx,
            K_target=K_target,
            K_bin_coarse=K_bin_coarse,
            K_bin_dense=K_bin_dense,
        )
    else:
        # Fixed 27-cell stencil
        Cand_max = min(27 * K_bin_coarse, 384)
        return gather_candidates_from_bins(
            pos=pos,
            bin_members=bin_members,
            bin_mask=bin_mask,
            bin_of=bin_of,
            r_search=r_search,
            Nbins_per_dim=Nbins_per_dim,
            dx=dx,
            Cand_max=Cand_max,
            K_bin=K_bin_coarse,
        )
