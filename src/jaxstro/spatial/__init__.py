"""
Spatial utilities for JAX-based astrophysical simulations.

This package provides efficient spatial data structures and algorithms for:

- **Morton (Z-order) encoding**: Cache-friendly 1D indexing of 3D grids that
  preserves spatial locality, improving memory access patterns for neighbor
  queries and domain decomposition.

- **Uniform grid binning**: O(N log N) particle-to-bin assignment with
  deterministic overflow handling via hash-based reservoir sampling.

- **Approximate neighbor candidate gathering**: Stencil-based queries over
  spatial bins to efficiently identify candidate neighbors without exhaustive
  O(N^2) distance calculations.

Intended applications:

- N-body integrators (neighbor lists, force evaluation)
- Initial condition builders (density profiles, mass segregation diagnostics)
- Cluster analysis (local density estimators, fractal dimension)
- Domain decomposition (MPI/GPU tiling based on Morton ordering)

References:
    Morton, G. M. (1966). "A computer oriented geodetic data base and a new
        technique in file sequencing." IBM Technical Report.
    Makino, J. & Aarseth, S. J. (1992). PASJ, 44, 141. (Neighbor list methods)

Example:
    >>> import jax.numpy as jnp
    >>> from jaxstro.spatial import (
    ...     assign_particles_to_bins,
    ...     fill_bins,
    ...     approx_knn_candidates,
    ... )
    >>> # Create random particle positions
    >>> pos = jax.random.uniform(jax.random.PRNGKey(0), (100, 3)) * 4.0 - 2.0
    >>> # Assign to spatial bins
    >>> bin_of = assign_particles_to_bins(pos, L_box=4.0, Nbins_per_dim=8)
    >>> # Fill bin arrays
    >>> particle_ids = jnp.arange(100, dtype=jnp.int32)
    >>> bin_members, bin_mask = fill_bins(particle_ids, bin_of, Nbins=512, Bcap=32)
"""

from __future__ import annotations

from jaxstro.spatial.grid import assign_particles_to_bins, fill_bins
from jaxstro.spatial.morton import morton_decode_3d, morton_encode_3d, wyhash32
from jaxstro.spatial.neighbor import (
    approx_knn_candidates,
    gather_candidates_from_bins,
    gather_candidates_two_stencil,
    gather_candidates_with_stencil,
)

__all__ = [
    # Morton encoding
    "morton_encode_3d",
    "morton_decode_3d",
    "wyhash32",
    # Grid binning
    "assign_particles_to_bins",
    "fill_bins",
    # Neighbor candidate gathering
    "gather_candidates_from_bins",
    "gather_candidates_with_stencil",
    "gather_candidates_two_stencil",
    "approx_knn_candidates",
]
