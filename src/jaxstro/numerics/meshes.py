"""Structured one-dimensional mesh and finite-volume helpers."""

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .grids import conservative_rebin


class Mesh1D(eqx.Module):
    """Structured 1D mesh represented by monotonically increasing edges."""

    edges: Float[Array, "n_plus_1"]

    @property
    def centers(self) -> Float[Array, "n"]:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def widths(self) -> Float[Array, "n"]:
        return jnp.diff(self.edges)

    @property
    def volumes(self) -> Float[Array, "n"]:
        return self.widths


class FaceGeometry1D(NamedTuple):
    """Face positions and unit areas for a Cartesian 1D mesh."""

    positions: Float[Array, "n_plus_1"]
    areas: Float[Array, "n_plus_1"]


class CellNeighbors1D(NamedTuple):
    """Left and right cell-neighbor indices with ``-1`` at boundaries."""

    left: Int[Array, "n"]
    right: Int[Array, "n"]


def structured_edges_1d(
    start: float | Float[Array, ""],
    stop: float | Float[Array, ""],
    *,
    n_cells: int,
) -> Float[Array, "n_plus_1"]:
    """Return uniformly spaced 1D cell edges."""
    if n_cells < 1:
        raise ValueError("n_cells must be at least 1")
    return jnp.linspace(start, stop, n_cells + 1)


def face_geometry_1d(edges: Float[Array, "n_plus_1"]) -> FaceGeometry1D:
    """Return Cartesian 1D face positions and unit face areas."""
    edges = jnp.asarray(edges)
    return FaceGeometry1D(positions=edges, areas=jnp.ones_like(edges))


def cell_neighbors_1d(n_cells: int) -> CellNeighbors1D:
    """Return left/right neighbor stencils with ``-1`` boundary sentinels."""
    if n_cells < 1:
        raise ValueError("n_cells must be at least 1")
    cells = jnp.arange(n_cells, dtype=jnp.int32)
    left = jnp.where(cells == 0, -1, cells - 1)
    right = jnp.where(cells == n_cells - 1, -1, cells + 1)
    return CellNeighbors1D(left=left, right=right)


def divergence_1d(
    face_flux: Float[Array, "n_plus_1"],
    edges: Float[Array, "n_plus_1"],
) -> Float[Array, "n"]:
    """Return finite-volume divergence ``(F_right - F_left) / dx``."""
    face_flux = jnp.asarray(face_flux)
    widths = jnp.diff(jnp.asarray(edges))
    return (face_flux[1:] - face_flux[:-1]) / widths


def cell_to_face_average(values: Float[Array, "n"]) -> Float[Array, "n_plus_1"]:
    """Average cell values to faces, copying boundary cell values."""
    values = jnp.asarray(values)
    interior = 0.5 * (values[:-1] + values[1:])
    return jnp.concatenate([values[:1], interior, values[-1:]], axis=0)


def conservative_remap_1d(
    old_edges: Float[Array, "n_old_plus_1"],
    old_cell_averages: Float[Array, "n_old"],
    new_edges: Float[Array, "n_new_plus_1"],
) -> Float[Array, "n_new"]:
    """Conservatively remap 1D cell averages onto new edges."""
    old_edges = jnp.asarray(old_edges)
    new_edges = jnp.asarray(new_edges)
    old_cell_averages = jnp.asarray(old_cell_averages)
    old_totals = old_cell_averages * jnp.diff(old_edges)
    new_totals = conservative_rebin(old_edges, old_totals, new_edges)
    new_widths = jnp.diff(new_edges)
    return new_totals / new_widths


__all__ = [
    "Mesh1D",
    "FaceGeometry1D",
    "CellNeighbors1D",
    "structured_edges_1d",
    "face_geometry_1d",
    "cell_neighbors_1d",
    "divergence_1d",
    "cell_to_face_average",
    "conservative_remap_1d",
]
