"""Focused public contracts for Morton-grid assignment."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jaxstro.spatial import assign_particles_to_bins


@pytest.mark.parametrize("bins_per_dim", [0, 3, 6, 1025])
def test_morton_grid_requires_positive_power_of_two(bins_per_dim: int) -> None:
    positions = jnp.zeros((1, 3))

    with pytest.raises(ValueError, match="positive power of two"):
        assign_particles_to_bins(
            positions,
            L_box=2.0,
            Nbins_per_dim=bins_per_dim,
        )


@pytest.mark.parametrize("bins_per_dim", [1, 2, 4, 16, 1024])
def test_supported_morton_grid_ids_fit_dense_allocation(bins_per_dim: int) -> None:
    corners = jnp.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
        ]
    )

    bin_ids = assign_particles_to_bins(
        corners,
        L_box=2.0,
        Nbins_per_dim=bins_per_dim,
    )

    assert jnp.all(bin_ids >= 0)
    assert jnp.all(bin_ids < bins_per_dim**3)
