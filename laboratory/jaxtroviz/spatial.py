"""Spatial-contract figures generated from the public jaxstro API."""

from __future__ import annotations

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from jaxstro.spatial import (
    assign_particles_to_bins,
    fill_bins_exact,
    gather_candidates_from_bins,
    gather_pairs_within_radius,
)

from .style import NEUTRAL, PALETTE, POSITIVE, setup_style

POSITIONS = jnp.array(
    [
        [-0.35, -0.25, 0.0],
        [-0.10, -0.20, 0.0],
        [0.10, -0.15, 0.0],
        [0.45, -0.10, 0.0],
        [-0.30, 0.25, 0.0],
        [0.05, 0.30, 0.0],
        [0.55, 0.35, 0.0],
        [0.85, 0.75, 0.0],
    ]
)
FOCAL_ID = 0
CUTOFF = 0.5


def _spatial_results() -> tuple[np.ndarray, np.ndarray, set[int], set[int], bool]:
    positions = POSITIONS
    n_particles = positions.shape[0]
    bins_per_dim = 4
    bin_ids = assign_particles_to_bins(
        positions,
        L_box=2.0,
        Nbins_per_dim=bins_per_dim,
    )
    members, member_mask, bin_overflow = fill_bins_exact(
        jnp.arange(n_particles, dtype=jnp.int32),
        bin_ids,
        Nbins=bins_per_dim**3,
        Bcap=n_particles,
    )
    positions_with_sentinel = jnp.concatenate(
        [positions, jnp.zeros((1, 3), dtype=positions.dtype)],
        axis=0,
    )
    candidates, candidate_mask = gather_candidates_from_bins(
        positions_with_sentinel,
        members,
        member_mask,
        bin_ids,
        r_search=jnp.full(n_particles, CUTOFF),
        Nbins_per_dim=bins_per_dim,
        dx=0.5,
        Cand_max=n_particles - 1,
        K_bin=n_particles,
    )
    neighbors, neighbor_mask, pair_overflow = gather_pairs_within_radius(
        positions,
        origin=jnp.array([-1.0, -1.0, -0.5]),
        cell_size=CUTOFF,
        cutoff=CUTOFF,
        k_max=n_particles - 1,
        Bcap=n_particles,
        dims=(4, 4, 2),
    )
    candidate_set = set(
        map(int, candidates[FOCAL_ID][candidate_mask[FOCAL_ID]].tolist())
    )
    neighbor_set = set(map(int, neighbors[FOCAL_ID][neighbor_mask[FOCAL_ID]].tolist()))
    did_overflow = bool(bin_overflow | pair_overflow)
    if candidate_set != {1, 2, 3, 4, 5}:
        raise RuntimeError(
            f"spatial figure candidate contract drifted: {candidate_set}"
        )
    if neighbor_set != {1, 2}:
        raise RuntimeError(
            f"spatial figure exact-neighbor contract drifted: {neighbor_set}"
        )
    if did_overflow:
        raise RuntimeError("spatial figure configuration must not overflow")
    return (
        np.asarray(positions),
        np.asarray(bin_ids),
        candidate_set,
        neighbor_set,
        did_overflow,
    )


def _format_panel(ax: plt.Axes) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_xticks(np.arange(-1.0, 1.01, 0.5))
    ax.set_yticks(np.arange(-1.0, 1.01, 0.5))
    ax.grid(True, color="#DCE3E8", linewidth=0.7, zorder=0)


def _annotate_ids(ax: plt.Axes, positions: np.ndarray, bin_ids: np.ndarray) -> None:
    for particle_id, ((x, y, _), bin_id) in enumerate(
        zip(positions, bin_ids, strict=True)
    ):
        ax.annotate(
            f"{particle_id} · z{bin_id}",
            (x, y),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=6.8,
            color=NEUTRAL,
        )


def build_spatial_neighbor_contracts() -> Figure:
    """Contrast a real grid candidate pool with exact cutoff filtering."""
    setup_style(font_scale=1.0)
    positions, bin_ids, candidates, neighbors, did_overflow = _spatial_results()
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.5), constrained_layout=True)
    focal = positions[FOCAL_ID, :2]

    left, right = axes
    _format_panel(left)
    left.set_title("Candidate pool", weight="bold")
    for candidate in sorted(candidates):
        target = positions[candidate, :2]
        left.plot(
            [focal[0], target[0]],
            [focal[1], target[1]],
            color=PALETTE[4],
            alpha=0.55,
            linewidth=1.0,
            zorder=1,
        )
    outside = sorted(set(range(len(positions))) - candidates - {FOCAL_ID})
    left.scatter(
        positions[outside, 0],
        positions[outside, 1],
        s=35,
        color="#B8C1C8",
        label="outside stencil pool",
        zorder=3,
    )
    candidate_ids = sorted(candidates)
    left.scatter(
        positions[candidate_ids, 0],
        positions[candidate_ids, 1],
        s=46,
        color=PALETTE[4],
        label="grid candidates",
        zorder=4,
    )
    left.scatter(
        *focal, s=70, color=PALETTE[0], marker="*", label="focal particle", zorder=5
    )
    _annotate_ids(left, positions, bin_ids)
    left.text(
        0.02,
        0.02,
        "fixed stencil + capacity\nmay include extras",
        transform=left.transAxes,
        fontsize=7.4,
        va="bottom",
        color="#754C24",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "#FFF7E5",
            "edgecolor": PALETTE[3],
        },
    )
    left.legend(loc="upper left", frameon=False)

    _format_panel(right)
    right.set_title("Exact radius filter", weight="bold")
    right.add_patch(
        Circle(
            focal,
            CUTOFF,
            facecolor=PALETTE[2],
            edgecolor=POSITIVE,
            alpha=0.10,
            linewidth=1.5,
            zorder=1,
        )
    )
    for neighbor in sorted(neighbors):
        target = positions[neighbor, :2]
        right.plot(
            [focal[0], target[0]],
            [focal[1], target[1]],
            color=POSITIVE,
            linewidth=1.7,
            zorder=2,
        )
    nonneighbors = sorted(set(range(len(positions))) - neighbors - {FOCAL_ID})
    right.scatter(
        positions[nonneighbors, 0],
        positions[nonneighbors, 1],
        s=35,
        color="#C5CBD0",
        label="rejected / outside cutoff",
        zorder=3,
    )
    neighbor_ids = sorted(neighbors)
    right.scatter(
        positions[neighbor_ids, 0],
        positions[neighbor_ids, 1],
        s=50,
        color=POSITIVE,
        label="exact neighbors",
        zorder=4,
    )
    right.scatter(
        *focal, s=70, color=PALETTE[0], marker="*", label="focal particle", zorder=5
    )
    _annotate_ids(right, positions, bin_ids)
    right.text(
        0.02,
        0.02,
        f"$0 < r \\leq {CUTOFF}$\ndid_overflow = {did_overflow}",
        transform=right.transAxes,
        fontsize=7.4,
        va="bottom",
        color="#1F665B",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "#EDF6F4",
            "edgecolor": POSITIVE,
        },
    )
    right.legend(loc="upper left", frameon=False)
    return figure
