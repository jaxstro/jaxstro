"""
Uniform grid binning utilities for spatial particle data.

This module provides efficient algorithms to:

1. **Assign particles to bins**: Map 3D positions to Morton-ordered bin indices
   on a uniform grid, with clamping at boundaries.

2. **Fill bins with overflow handling**: Populate fixed-capacity bin arrays
   using deterministic hash-based reservoir sampling when bins exceed capacity.

These utilities form the foundation for spatial queries (neighbor lists,
local density estimates) without O(N^2) pairwise distance calculations.

TODO: Add support for arbitrary box_min/box_max bounds (non-symmetric boxes)
and optional periodic wrapping instead of clamping for cosmological volumes.

TODO: Consider exposing a higher-level `build_uniform_grid` helper that
wraps assign_particles_to_bins + fill_bins, and returns a dataclass
with (bin_of, bin_members, bin_mask, dx, Nbins_per_dim).

TODO(jaxstro): Provide a `local_density` helper that uses `fill_bins` +
`approx_knn_candidates` to compute Sigma (k-th nearest neighbor surface density)
for use in mass segregation diagnostics (e.g. Allison+2009 Lambda_MSR) and
fractal IC characterization.

Example:
    >>> import jax.numpy as jnp
    >>> from jaxstro.spatial.grid import assign_particles_to_bins, fill_bins
    >>> # Random positions in [-2, 2]^3
    >>> pos = jnp.array([[0.0, 0.0, 0.0], [1.5, -1.0, 0.5]])
    >>> bin_of = assign_particles_to_bins(pos, L_box=4.0, Nbins_per_dim=8)
    >>> particle_ids = jnp.arange(2, dtype=jnp.int32)
    >>> bin_members, bin_mask = fill_bins(particle_ids, bin_of, Nbins=512, Bcap=10)
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from jaxstro.spatial.morton import MAX_BITS_3D, morton_encode_3d, wyhash32

# =============================================================================
# Bin Assignment
# =============================================================================


def assign_particles_to_bins(
    pos: Float[Array, "N 3"],
    L_box: float,
    Nbins_per_dim: int,
    box_center: float | Float[Array, "3"] = 0.0,
    symmetric: bool = True,
) -> Int[Array, "N"]:
    """
    Assign particles to Morton-ordered bins on a uniform grid.

    Maps particle positions to bin indices using a uniform grid, then encodes
    the 3D bin coordinates to 1D Morton codes for cache-coherent access.

    Grid layout:
        If symmetric=True (default), the box spans:
            [center - L_box/2, center + L_box/2]^3
        Particles outside this range are clamped to boundary bins.

    Args:
        pos: Particle positions [N, 3]
        L_box: Box size (cube side length)
        Nbins_per_dim: Bins per dimension (total bins = Nbins^3).
            Must be <= 1024 (10-bit Morton encoding limit).
        box_center: Center of the box. Scalar or [3] array. Default: 0.0
        symmetric: If True, box spans [center - L/2, center + L/2].
            Currently only symmetric=True is supported.

    Returns:
        bin_of: Morton bin ID per particle [N], in range [0, Nbins^3 - 1]

    Raises:
        AssertionError: If Nbins_per_dim > 1024
        NotImplementedError: If symmetric=False

    Example:
        >>> pos = jnp.array([[0.0, 0.0, 0.0]])  # Center of box
        >>> bin_of = assign_particles_to_bins(pos, L_box=4.0, Nbins_per_dim=8)
        >>> # Center maps to bin (4, 4, 4) in a 0-7 grid
        >>> bin_of
        Array([182], dtype=int32)

    Note:
        The Morton encoding preserves spatial locality: particles in nearby
        3D bins will have nearby 1D bin IDs, improving cache performance.
    """
    # Validate Nbins_per_dim against Morton encoding limit
    assert Nbins_per_dim <= 2**MAX_BITS_3D, (
        f"Morton encoder uses {MAX_BITS_3D} bits/dim "
        f"(max {2**MAX_BITS_3D}), got {Nbins_per_dim}"
    )

    if not symmetric:
        raise NotImplementedError(
            "Only symmetric=True is currently supported. "
            "TODO: Add support for arbitrary box_min/box_max bounds."
        )

    # Compute bin size
    dx = L_box / Nbins_per_dim

    # Map positions to bin indices (0 to Nbins_per_dim-1)
    # Positions: [center - L_box/2, center + L_box/2] -> bins: [0, Nbins_per_dim-1]
    pos_shifted = pos - box_center + L_box / 2
    bin_xyz = jnp.floor(pos_shifted / dx).astype(jnp.int32)

    # Clamp to valid range (handle boundary particles)
    bin_xyz = jnp.clip(bin_xyz, 0, Nbins_per_dim - 1)

    # Morton encode to 1D bin index
    bin_of = morton_encode_3d(bin_xyz, bits=MAX_BITS_3D)

    return bin_of


# =============================================================================
# Bin Filling with Overflow Handling
# =============================================================================


def fill_bins(
    particle_ids: Int[Array, "N"],
    bin_of: Int[Array, "N"],
    Nbins: int,
    Bcap: int,
    sentinel_N: Optional[int] = None,
) -> tuple[Int[Array, "Nbins Bcap"], Bool[Array, "Nbins Bcap"]]:
    """
    Fill spatial bins with particles using hash-based reservoir sampling.

    When a bin contains more than Bcap particles, this function keeps exactly
    Bcap particles using an unbiased, deterministic selection based on hash
    values. The algorithm is O(N log N) via a single lexicographic sort.

    Algorithm:
        1. Compute a 32-bit hash for each particle (mixed with bin ID)
        2. Sort particles by (bin_id, hash) lexicographically
        3. Take the first Bcap particles from each bin segment

    This approach is much faster than O(N x Nbins) per-bin scans for large N,
    and the hash-based selection ensures reproducible, unbiased overflow handling.

    Provenance:
        The "sort-by-(bin, hash) then take the first Bcap per segment" overflow
        rule is an ORIGINAL heuristic for this codebase (not taken from a
        specific paper). It is a deterministic, order-independent reservoir-style
        downsample: because the per-particle key mixes a hash of the particle ID
        with the bin ID (wyhash32, a standard non-cryptographic mixer), the
        retained subset is independent of input ordering and uniform over the
        bin's members, while remaining a single lax.sort (GPU/TPU-friendly).
        Treat the specific selection rule as project-local, not literature.

    Args:
        particle_ids: Particle indices [N]. These are the values stored in bins.
        bin_of: Bin assignment per particle [N], from assign_particles_to_bins.
        Nbins: Total number of bins (e.g., Nbins_per_dim^3 for a cubic grid)
        Bcap: Maximum particles per bin (capacity). Bins with more particles
            will have overflow handled via reservoir sampling.
        sentinel_N: Sentinel value for invalid/empty slots. Default: N
            (the number of particles). Empty slots are filled with this value.

    Returns:
        bin_members: Particle IDs in each bin [Nbins, Bcap].
            Empty slots contain sentinel_N.
        bin_mask: Boolean mask [Nbins, Bcap] where True = valid particle,
            False = empty slot (sentinel value).

    Example:
        >>> particle_ids = jnp.arange(20, dtype=jnp.int32)
        >>> bin_of = jnp.zeros(20, dtype=jnp.int32)  # All in bin 0
        >>> bin_members, bin_mask = fill_bins(particle_ids, bin_of, Nbins=8, Bcap=10)
        >>> jnp.sum(bin_mask[0])  # Bin 0 has 10 particles (overflow handled)
        Array(10, dtype=int32)

    Performance:
        O(N log N) via single lexicographic sort on (bin, hash).
        Memory: O(N) for sort, O(Nbins x Bcap) for output arrays.

    Note:
        The sentinel convention assumes that position arrays are padded to
        length N+1, with the sentinel position at index N. This allows safe
        indexing of pos[sentinel_N] without bounds errors.
    """
    N = particle_ids.shape[0]
    if sentinel_N is None:
        sentinel_N = N  # Expect position arrays padded to length N+1

    # 32-bit hash per particle, mixed with bin ID (deterministic, unbiased)
    bin_u32 = jnp.uint32(bin_of)
    pid_u32 = jnp.uint32(particle_ids)
    h = wyhash32(pid_u32 ^ (bin_u32 * jnp.uint32(0x9E3779B1)))  # uint32

    # Lexicographic key: (bin << 32) | hash
    # Ascending sort groups by bin, then LOWEST hash first (reservoir criterion)
    key64 = (jnp.uint64(bin_u32) << jnp.uint64(32)) | jnp.uint64(h)

    # Single sort: O(N log N)
    _, idx_sorted = jax.lax.sort_key_val(key64, particle_ids, dimension=0)
    bins_sorted = bin_of[idx_sorted]

    # Compute segment boundaries per bin
    counts = jnp.bincount(bins_sorted, length=Nbins).astype(jnp.int32)  # [Nbins]
    starts = jnp.cumsum(counts) - counts  # [Nbins]
    ends = starts + counts  # [Nbins]

    # Absolute positions for first Bcap particles per bin (static shape [Nbins, Bcap])
    offsets = jnp.arange(Bcap, dtype=jnp.int32)[None, :]  # [1, Bcap]
    abs_pos = starts[:, None] + offsets  # [Nbins, Bcap]

    # Validity mask: only positions < end are valid
    valid = abs_pos < ends[:, None]

    # Guard against empty bins when gathering (clamp to valid range)
    abs_pos_safe = jnp.clip(abs_pos, 0, jnp.maximum(N - 1, 0))

    # Gather sorted particle IDs
    picked = idx_sorted[abs_pos_safe]  # [Nbins, Bcap]

    # Use sentinel N for invalid slots
    bin_members = jnp.where(valid, picked, sentinel_N)
    bin_mask = valid

    return bin_members.astype(jnp.int32), bin_mask
