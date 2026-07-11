"""Exact fixed-radius neighbour pair gathering (open boundary, dense linear cells)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from jaxstro.spatial.grid import assign_to_cells_linear, fill_bins_exact

__all__ = ["gather_pairs_within_radius"]


def gather_pairs_within_radius(
    pos: Float[Array, "N 3"],
    origin: Float[Array, "3"],
    cell_size: float,
    cutoff: float,
    k_max: int,
    Bcap: int | None = None,
    dims: tuple[int, int, int] | None = None,
) -> tuple[Int[Array, "N k_max"], Bool[Array, "N k_max"], Bool[Array, ""]]:
    """Exact neighbours within `cutoff`: {j : 0 < |x_i-x_j| <= cutoff}.

    27-cell stencil over a dense row-major grid of cells of side `cell_size`
    (>= cutoff required so the stencil covers the cutoff sphere). Fixed capacity
    k_max per particle; `did_overflow` is True if any particle has more in-radius
    neighbours than k_max OR any cell exceeds Bcap. General `cutoff` (a0 for g0,
    r_close for SDAR). NO top-K truncation of the true set unless did_overflow.

    Returns (nbr_idx [N,k_max] int32, nbr_mask [N,k_max] bool, did_overflow []).

    Preconditions / guarantees:
        (a) In-grid coverage holds for ANY dims and for off-box particles: cell
            assignment is non-expansive (clamped to [0, n-1] per axis), and the
            27-cell stencil is gated by an IN-RANGE offset mask computed before
            clamping. Each true neighbour cell c_j = c_i + off (off in {-1,0,+1})
            lies in [0, n-1], so exactly one in-range offset reaches it -- no
            true neighbour is dropped and no boundary cell is counted twice.
        (b) The `dims=None` auto-size path reads `pos` on the host (eager only).
            Under jit you MUST pass an explicit static `dims`.
        (c) Dense row-major cell ids are int32, so nx*ny*nz must stay under the
            int32 ceiling (~2.1e9 cells); size the grid accordingly.
        (d) Coincident particles (r == 0) are EXCLUDED by the `0 < r` contract;
            only {j : 0 < |x_i-x_j| <= cutoff} are returned.
    """
    if float(cell_size) < float(cutoff):
        raise ValueError(
            f"cell_size ({cell_size}) must be >= cutoff ({cutoff}) so the 27-cell "
            "stencil covers the cutoff sphere."
        )
    N = pos.shape[0]
    if Bcap is None:
        Bcap = min(N, max(k_max, 64))
    # Grid dims from the point cloud extent (open boundary + 1 cell margin).
    if dims is None:
        span = jnp.max(pos, axis=0) - origin
        dmax = [int(jnp.ceil(float(span[a]) / cell_size)) + 1 for a in range(3)]
        dims = (max(dmax[0], 1), max(dmax[1], 1), max(dmax[2], 1))
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    Ncells = nx * ny * nz
    cell_of = assign_to_cells_linear(pos, origin, cell_size, dims)  # [N]
    pids = jnp.arange(N, dtype=jnp.int32)
    members, mmask, did_bins = fill_bins_exact(
        pids, cell_of, Ncells, Bcap
    )  # [Ncells,Bcap]
    # sentinel row for safe gather of empty slots
    pos_s = jnp.concatenate([pos, jnp.zeros((1, 3), pos.dtype)], axis=0)  # [N+1,3]
    members = jnp.where(mmask, members, N)  # sentinel = N
    # decode each particle's cell -> (ix,iy,iz)
    iz = cell_of // (nx * ny)
    iy = (cell_of - iz * nx * ny) // nx
    ix = cell_of - iz * nx * ny - iy * nx
    offs = jnp.array([-1, 0, 1], dtype=jnp.int32)
    # 27 neighbour cells. Clamp keeps the gather index in-bounds, but a boundary
    # cell would otherwise be reached by TWO offsets that clamp to the same id
    # (e.g. -1 and 0 both -> 0), duplicating its members in `cand`. To avoid
    # that we gate candidates by the IN-RANGE stencil validity computed BEFORE
    # clamping: each true neighbour cell c_j = c_i + off with off in {-1,0,+1}
    # satisfies c_j in [0, n-1], so exactly one in-range offset reaches it. This
    # is provably lossless (no true neighbour dropped) while removing duplicates.
    ax = ix[:, None] + offs[None, :]  # [N,3] raw x-offsets (pre-clamp)
    ay = iy[:, None] + offs[None, :]
    az = iz[:, None] + offs[None, :]
    inx = (ax >= 0) & (ax <= nx - 1)  # [N,3] per-axis in-range masks
    iny = (ay >= 0) & (ay <= ny - 1)
    inz = (az >= 0) & (az <= nz - 1)
    jx = jnp.clip(ax, 0, nx - 1)  # [N,3] clamped ONLY to keep gather in-bounds
    jy = jnp.clip(ay, 0, ny - 1)
    jz = jnp.clip(az, 0, nz - 1)
    cx, cy, cz = jnp.broadcast_arrays(
        jx[:, :, None, None], jy[:, None, :, None], jz[:, None, None, :]
    )
    ncell = (cx + nx * (cy + ny * cz)).reshape(N, 27)  # [N,27] neighbour cell ids
    # [N,27] in-range mask via the same broadcast/Cartesian pattern as the cells
    mx, my, mz = jnp.broadcast_arrays(
        inx[:, :, None, None], iny[:, None, :, None], inz[:, None, None, :]
    )
    in_range = (mx & my & mz).reshape(N, 27)  # [N,27] True where offset is valid
    cand = members[ncell].reshape(
        N, 27 * Bcap
    )  # [N, 27*Bcap] candidate ids (sentinel=N)
    # broadcast the per-cell in-range mask over Bcap slots -> [N, 27*Bcap]
    cand_in_range = jnp.broadcast_to(in_range[:, :, None], (N, 27, Bcap)).reshape(
        N, 27 * Bcap
    )
    # distances
    d = pos[:, None, :] - pos_s[cand]  # [N, 27*Bcap, 3]
    r2 = jnp.sum(d * d, axis=-1)
    is_self = cand == pids[:, None]
    within = (
        (r2 > 0.0) & (r2 <= cutoff * cutoff) & (~is_self) & (cand != N) & cand_in_range
    )
    # count within per particle -> overflow if > k_max
    n_within = jnp.sum(within, axis=1)
    did_overflow = jnp.any(n_within > k_max) | did_bins
    # select the k_max closest within (all of them when n_within <= k_max)
    score = jnp.where(within, -r2, -jnp.inf)
    _, top = jax.lax.top_k(score, k_max)  # [N,k_max]
    nbr_idx = jnp.take_along_axis(cand, top, axis=1)
    nbr_mask = jnp.isfinite(jnp.take_along_axis(score, top, axis=1))
    nbr_idx = jnp.where(nbr_mask, nbr_idx, N).astype(jnp.int32)
    return nbr_idx, nbr_mask, did_overflow
