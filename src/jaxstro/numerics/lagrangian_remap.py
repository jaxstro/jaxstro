r"""Split and merge of cells on a one-dimensional Lagrangian (mass-coordinate) mesh.

A Lagrangian mesh is labelled by enclosed mass. Changing its resolution means splitting a
cell into two children or merging neighbours into one, and moving the conserved contents
with them. This module owns that operation and nothing physical: no refinement criterion,
no equation of state, no field names. Consumers (hydrax for shock refinement, stellax for
MESA's mesh plan) supply the criterion and decide which fields are extensive.

**Half cells.** Every old cell ``i`` is cut at its half-mass point into two half cells,
numbered ``2i`` and ``2i + 1``. A new cell is a contiguous run of half cells, so its
boundaries lie on old faces or on old half-mass points, which is MESA's rule
(``max_num_subcells = 2``). The run is the plan; the transfer sums the contents of the half
cells it covers. A plan may instead use ``subcells = 4`` equal-mass parts per old cell,
which lets a half cell be halved again (MESA's surface pass does this). Cells are ordered
centre-out (index 0 is the innermost cell) and the mesh
carries all ``n + 1`` faces, the inner one included.

1. **Plan.** A :class:`RemapPlan` gives, for each new slot, the first half cell and the
   number of half cells (``start``, ``length``). Runs tile the old mesh in order. Static
   capacity, so it traces under ``jit``. :func:`plan_from_actions` builds one from per-cell
   actions (keep, split in two, merge with the right neighbour); :func:`fixed_count_plan`
   chooses equal numbers of splits and merges, so array shapes never change. A caller with
   its own planner (stellax's MESA walk) builds a :class:`RemapPlan` directly.
2. **Transfer.** :func:`apply_plan` moves the faces, the masses and every extensive field.
   A run of whole old cells is a sum of old totals, so a kept cell (one whole cell) is a
   gather and is bitwise unchanged, and a merge is a plain sum. A run that starts or ends
   on a half-mass point sums half-cell contents.

The half-cell contents of an extensive field :math:`Q = \bar q\,\Delta m` use a linear
reconstruction of the specific value in mass, :math:`q(m) = \bar q + s\,(m - m_c)`. The
inner half holds

.. math:: Q_L = \tfrac12 Q - \tfrac18 s\,\Delta m^2, \qquad Q_R = Q - Q_L,

which is an equal share when :math:`s = 0` (piecewise-constant: children carry their
parent's specific values, MESA's rule). The outer half is the difference, so the two halves
sum to the parent to rounding. The half-mass face uses the same rule on the volume
coordinate: with :math:`\Delta\xi` the cell's extent in a coordinate in which volume is
linear (:math:`\xi = r^3` in spherical symmetry, :math:`\xi = x` in slab symmetry), the
inner half spans :math:`\tfrac12\Delta\xi - \tfrac18 s_v\,\Delta m^2`, with :math:`s_v` the
slope of the specific extent :math:`\Delta\xi/\Delta m`.

**Derivatives.** The plan is integer data and carries no derivative. A gradient through a
run that regrids records the plans in the forward pass and replays them unchanged (the
plan is the frozen hierarchy). :func:`apply_plan` is then a fixed composition of gathers,
sums and products, smooth in the faces, masses and fields. :func:`limited_slopes` is
piecewise smooth; its derivative is that of the selected limiter branch.
:func:`smooth_slopes` has no branch, so the transfer is smooth in the state everywhere.

**Provenance.** The split/merge structure (cuts at old faces and half-mass points, merges of
whole cells, exact copy of cells that need no change) follows the description of MESA's
mesh adjustment in Paxton et al. (2011, ApJS 192, 3) and the stellax design note
``stellax/docs/plans/2026-10-02-mesh-split-merge-design.md`` section 5. No MESA source code
was used. The half-cell share is derived above; the limiter is the monotonised central
limiter of van Leer (1977, J. Comput. Phys. 23, 276); the smooth limiter is eq (37) of
van Albada, van Leer & Roberts (1982, A&A 108, 76).
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

__all__ = [
    "KEEP",
    "MERGE_RIGHT",
    "SPLIT",
    "RemapPlan",
    "plan_from_actions",
    "fixed_count_plan",
    "limited_slopes",
    "smooth_slopes",
    "face_values",
    "apply_plan",
]

# Actions on old cells, for plan_from_actions.
KEEP = 0
SPLIT = 1
MERGE_RIGHT = 2


class RemapPlan(eqx.Module):
    """The integer record of one regrid: which half cells each new cell is made of.

    Old cell ``i`` consists of sub-cells ``k i .. k i + k - 1`` of equal mass, with
    ``k = subcells`` (2 by default: half cells ``2i`` inner and ``2i + 1`` outer).

    Attributes:
        start: first sub-cell of each new slot.
        length: number of sub-cells in each new slot; ``0`` marks an empty slot.
        n_active: number of new cells, including any that did not fit the capacity, so a
            caller detects an overflow by ``n_active > capacity``.
        max_length: static bound on ``length`` (``2 * subcells`` allows a merge of two
            whole cells).
        subcells: static number of equal-mass sub-cells per old cell. 2 is MESA's
            ``max_num_subcells``; 4 lets a half cell be halved again, which MESA's surface
            pass does (stellax, 2026-10-03: one face in 230,000 over a 1 Msun PMS-TAMS run,
            at the quarter point of the surface cell).

    Active slots come first and tile the old mesh in order:
    ``start[k + 1] = start[k] + length[k]``, ``start[0] = 0``. :meth:`is_valid` checks it.
    The plan has no floating-point content, so applying a recorded plan again in a replay
    reproduces the same topology change exactly.
    """

    start: Int[Array, " capacity"]
    length: Int[Array, " capacity"]
    n_active: Int[Array, ""]
    max_length: int = eqx.field(static=True, default=4)
    subcells: int = eqx.field(static=True, default=2)

    @property
    def capacity(self) -> int:
        return self.start.shape[0]

    @property
    def is_copy(self) -> Bool[Array, " capacity"]:
        """Slots that are one whole old cell, transferred bitwise."""
        k = self.subcells
        return (self.length == k) & (self.start % k == 0)

    @property
    def is_split(self) -> Bool[Array, " capacity"]:
        """Slots that are part of one old cell: a child of a split."""
        first = self.start // self.subcells
        last = (self.start + self.length - 1) // self.subcells
        return (self.length > 0) & (self.length < self.subcells) & (last == first)

    @property
    def is_merge(self) -> Bool[Array, " capacity"]:
        """Slots that contain parts of more than one old cell."""
        first = self.start // self.subcells
        last = (self.start + self.length - 1) // self.subcells
        return (self.length > 0) & (last > first)

    def is_valid(self, n_old: int) -> Bool[Array, ""]:
        """Whether the active runs tile ``n_old`` cells exactly and respect ``max_length``."""
        active = self.length > 0
        count = jnp.sum(active)
        ordered = jnp.all(active[:-1] | ~active[1:])
        end = self.start + self.length
        chained = jnp.all(jnp.where(active[1:], self.start[1:] == end[:-1], True))
        total = jnp.sum(self.length) == self.subcells * n_old
        bounded = jnp.all(self.length <= self.max_length)
        fits = self.n_active == count
        return ordered & chained & total & bounded & fits & (self.start[0] == 0)


def plan_from_actions(action: Int[Array, " n"], capacity: int) -> RemapPlan:
    """Build the plan for per-cell actions.

    Args:
        action: one of :data:`KEEP`, :data:`SPLIT`, :data:`MERGE_RIGHT` per old cell. A
            :data:`MERGE_RIGHT` cell absorbs its right neighbour, which must be
            :data:`KEEP` and must exist (the last cell cannot merge right). These
            preconditions are the caller's; :func:`fixed_count_plan` guarantees them.
        capacity: static number of new slots. Cells beyond it are dropped and
            ``n_active`` reports the produced count.

    New positions come from an exclusive cumulative sum of the per-cell emit counts (1
    kept, 2 split, 1 for a merge's left cell, 0 for the cell it absorbs).
    """
    action = jnp.asarray(action, dtype=jnp.int32)
    n = action.shape[0]
    absorbed = jnp.concatenate([jnp.zeros(1, bool), action[:-1] == MERGE_RIGHT])
    emit = jnp.where(absorbed, 0, jnp.where(action == SPLIT, 2, 1))
    first = jnp.cumsum(emit) - emit
    half = 2 * jnp.arange(n, dtype=jnp.int32)
    first_len = jnp.where(action == SPLIT, 1, jnp.where(action == MERGE_RIGHT, 4, 2))
    slot1 = jnp.where(emit > 0, first, capacity)
    slot2 = jnp.where(emit == 2, first + 1, capacity)
    start = jnp.zeros(capacity, jnp.int32)
    length = jnp.zeros(capacity, jnp.int32)
    start = start.at[slot1].set(half, mode="drop").at[slot2].set(half + 1, mode="drop")
    length = length.at[slot1].set(first_len, mode="drop").at[slot2].set(1, mode="drop")
    return RemapPlan(start=start, length=length, n_active=jnp.sum(emit))


def fixed_count_plan(
    split_score: Float[Array, " n"],
    merge_score: Float[Array, " n_minus_1"],
    split_threshold: float | Float[Array, ""],
    merge_threshold: float | Float[Array, ""],
    max_changes: int,
    split_eligible: Bool[Array, " n"] | None = None,
    merge_eligible: Bool[Array, " n_minus_1"] | None = None,
    size: Float[Array, " n"] | None = None,
    max_ratio: float | None = None,
) -> RemapPlan:
    """Choose splits and merges in equal number, so the cell count is unchanged.

    A cell is a split candidate when its ``split_score`` exceeds ``split_threshold``. A
    neighbour pair ``(i, i + 1)`` is a merge candidate when its ``merge_score`` is below
    ``merge_threshold``, neither cell is a split candidate, and the pair's score is a strict
    local minimum among the overlapping pairs ``(i - 1, i)`` and ``(i + 1, i + 2)``, ordered
    by score and then by index (an equal score goes to the left pair). The local-minimum rule
    makes the chosen merges disjoint without a sequential walk. Separate thresholds give the
    hysteresis that stops a cell being split and merged on alternate regrids.

    With ``size`` and ``max_ratio`` (MESA uses 2.5 on ``dq``): a cell larger than
    ``max_ratio`` times either neighbour is a split candidate ahead of every scored one, and
    a pair is not merged if the merged cell would exceed ``max_ratio`` times either outer
    neighbour.

    ``k = min(#split candidates, #merge candidates, max_changes)``; the ``k`` most demanding
    splits and the ``k`` smallest merge scores are applied. Every other cell is kept, and
    :func:`apply_plan` copies it bitwise.

    Args:
        split_score: per-cell refinement demand (a usage, a gradient, a shock sensor).
        merge_score: per-pair coarsening allowance (smaller merges first).
        split_threshold, merge_threshold: candidate thresholds.
        max_changes: static upper bound on splits (and on merges) per regrid.
        split_eligible, merge_eligible: optional masks; ``False`` excludes a cell or pair
            (a pinned face, a sink cell, a boundary).
        size, max_ratio: optional cell-ratio limit (both or neither).

    Returns:
        A :class:`RemapPlan` with capacity equal to the old cell count.
    """
    split_score = jnp.asarray(split_score, dtype=float)
    merge_score = jnp.asarray(merge_score, dtype=float)
    n = split_score.shape[0]
    if merge_score.shape != (n - 1,):
        raise ValueError(
            f"merge_score must have shape ({n - 1},), got {merge_score.shape}"
        )
    if not 0 < max_changes <= n // 2:
        raise ValueError(f"max_changes must lie in [1, {n // 2}], got {max_changes}")
    if (size is None) != (max_ratio is None):
        raise ValueError("size and max_ratio must be given together")
    if split_eligible is None:
        split_eligible = jnp.ones(n, bool)
    if merge_eligible is None:
        merge_eligible = jnp.ones(n - 1, bool)
    inf = jnp.inf

    split_cand = split_score > split_threshold
    priority = split_score
    if size is not None and max_ratio is not None:
        size = jnp.asarray(size, dtype=float)
        big = jnp.full(1, inf)
        inner = jnp.concatenate([big, size[:-1]])
        outer = jnp.concatenate([size[1:], big])
        too_big = (size > max_ratio * inner) | (size > max_ratio * outer)
        split_cand = split_cand | too_big
        priority = jnp.where(too_big, inf, split_score)
        merged = size[:-1] + size[1:]
        merge_eligible = (
            merge_eligible
            & (merged <= max_ratio * jnp.concatenate([big, size[:-2]]))
            & (merged <= max_ratio * jnp.concatenate([size[2:], big]))
        )
    split_cand = split_cand & split_eligible

    pair_blocked = split_cand[:-1] | split_cand[1:]
    pair_cand = merge_eligible & (merge_score < merge_threshold) & ~pair_blocked
    masked = jnp.where(pair_cand, merge_score, inf)
    left = jnp.concatenate([jnp.full(1, inf), masked[:-1]])
    right = jnp.concatenate([masked[1:], jnp.full(1, inf)])
    # Strict local minimum in the order (score, index): two overlapping pairs never both pass.
    pair_cand = pair_cand & (masked < left) & (masked <= right)

    k = jnp.minimum(jnp.minimum(jnp.sum(split_cand), jnp.sum(pair_cand)), max_changes)
    take = jnp.arange(max_changes) < k
    _, split_idx = jax.lax.top_k(jnp.where(split_cand, priority, -inf), max_changes)
    _, pair_idx = jax.lax.top_k(jnp.where(pair_cand, -merge_score, -inf), max_changes)
    action = jnp.zeros(n, jnp.int32)
    action = action.at[jnp.where(take, split_idx, n)].set(SPLIT, mode="drop")
    action = action.at[jnp.where(take, pair_idx, n)].set(MERGE_RIGHT, mode="drop")
    return plan_from_actions(action, capacity=n)


def _min(a, b):
    """``min`` by selection, so a tie sends the whole derivative to one argument."""
    return jnp.where(a <= b, a, b)


def limited_slopes(
    specific: Float[Array, " n ..."], dm: Float[Array, " n"], joint: bool = False
) -> Float[Array, " n ..."]:
    """Monotonised-central (MC) slopes of a specific value in the mass coordinate.

    ``s_i = minmod(2 s_L, 2 s_R, s_C)``: one-sided slopes over the centre-to-centre mass
    distances, the central slope over the sum of the two; zero in the first and last cells
    and wherever the one-sided slopes differ in sign. The half-cell means it produces,
    ``q_i -/+ s_i dm_i / 4``, are bounded by the neighbouring means
    (``|s_i| dm_i / 4 <= |q_{i+1} - q_i| dm_i / (dm_i + dm_{i+1})``), so a positive specific
    value stays positive in both children.

    Trailing axes of ``specific`` are limited independently, unless ``joint``: then every
    component of a cell takes the central slope times ONE factor, the smallest any component
    needs. For mass fractions that sum to one the central slopes sum to zero, so the
    limited slopes do too, and split children keep the sum (to rounding).

    Selections use ``jnp.where``, so the derivative is that of the selected branch.
    """
    q = jnp.asarray(specific)
    dm = jnp.asarray(dm)
    expand = (slice(None),) + (None,) * (q.ndim - 1)
    h = 0.5 * (dm[1:] + dm[:-1])[expand]
    d = (q[1:] - q[:-1]) / h
    s_l, s_r = d[:-1], d[1:]
    s_c = (q[2:] - q[:-2]) / (h[:-1] + h[1:])
    mag = _min(_min(2.0 * jnp.abs(s_l), 2.0 * jnp.abs(s_r)), jnp.abs(s_c))
    same = (s_l * s_r) > 0.0
    if joint and q.ndim > 1:
        nonzero = jnp.abs(s_c) > 0.0
        safe = jnp.where(nonzero, jnp.abs(s_c), 1.0)
        phi = jnp.where(
            same, jnp.where(nonzero, mag / safe, 1.0), jnp.where(nonzero, 0.0, 1.0)
        )
        axes = tuple(range(1, q.ndim))
        phi_cell = jnp.min(phi, axis=axes, keepdims=True)
        interior = phi_cell * s_c
    else:
        interior = jnp.where(same, jnp.sign(s_c) * mag, 0.0)
    zero = jnp.zeros_like(q[:1])
    return jnp.concatenate([zero, interior, zero], axis=0)


def smooth_slopes(
    specific: Float[Array, " n ..."],
    dm: Float[Array, " n"],
    k_eps: float,
    relative: bool = False,
    sum_to_zero: bool = False,
) -> Float[Array, " n ..."]:
    r"""Slopes of a specific value in mass from van Albada's smooth limiter.

    Van Albada, van Leer & Roberts (1982, A&A 108, 76) eq (37) averages the two one-sided
    differences of a cell as

    .. math:: \mathrm{ave}(a, b) = \frac{(b^2 + \epsilon^2)\,a + (a^2 + \epsilon^2)\,b}
              {a^2 + b^2 + 2\epsilon^2},

    a convex combination of ``a`` and ``b`` (weights ``b^2 + eps^2`` and ``a^2 + eps^2``), so
    ``|ave| <= max(|a|, |b|)``. It is smooth for ``eps > 0``: no selection, hence no branch to
    flip at near-equal differences. Where both differences are small against ``eps`` it tends
    to their mean, where one is much smaller it tends to that one, and at an extremum
    (``a b < 0``) it returns a small slope of the smaller one's sign rather than zero, so a
    child can exceed its neighbours by a fraction of the smaller difference (not TVD).

    ``a`` and ``b`` are the changes across the cell's own mass from the one-sided
    centre-to-centre slopes, made dimensionless before eq (37):

    - ``relative``: divided by the mean of the two cells each difference spans, eq (31.3) of
      the same paper, for positive values. Each relative difference is below 2 and the factor
      ``dm_i / h`` below 2, so ``|ave| < 4`` and the half-cell children
      ``q_i (1 -/+ ave / 4)`` stay positive for any neighbour mass ratio.
    - otherwise: divided by the root mean square of the three cells, for signed values.

    ``eps^2 = (k_eps dm_i / sum(dm))^3``: of order (cell size)^3 as the paper prescribes,
    with the cell size as a fraction of the total mass. ``k_eps`` is the caller's closure.

    ``sum_to_zero`` (relative, trailing axis of fractions summing to one): the slopes are
    projected to sum to zero, ``s_j - q_j sum_l s_l``, so the children of a split still sum
    to one while each component is conserved exactly. The projection can push a trace
    component's child below zero where the components' relative slopes differ by more than
    4; the caller owns the bound at zero.

    The first and last cells take zero slope. Returns slopes in mass, the ``slopes`` input
    of :func:`apply_plan`.
    """
    q = jnp.asarray(specific)
    dm = jnp.asarray(dm)
    expand = (slice(None),) + (None,) * (q.ndim - 1)
    h = (0.5 * (dm[1:] + dm[:-1]))[expand]
    w = dm[1:-1][expand]
    diff = q[1:] - q[:-1]
    if relative:
        d = diff / (0.5 * (q[1:] + q[:-1]))
        a, b = d[:-1] * w / h[:-1], d[1:] * w / h[1:]
        scale = q[1:-1]
    else:
        scale = jnp.sqrt((q[:-2] ** 2 + q[1:-1] ** 2 + q[2:] ** 2) / 3.0)
        a, b = diff[:-1] * w / h[:-1] / scale, diff[1:] * w / h[1:] / scale
    eps2 = ((k_eps * dm[1:-1] / jnp.sum(dm)) ** 3)[expand]
    ave = ((b * b + eps2) * a + (a * a + eps2) * b) / (a * a + b * b + 2.0 * eps2)
    interior = scale * ave / w
    zero = jnp.zeros_like(q[:1])
    s = jnp.concatenate([zero, interior, zero], axis=0)
    if sum_to_zero:
        frac = q / jnp.sum(q, axis=-1, keepdims=True)
        s = s - frac * jnp.sum(s, axis=-1, keepdims=True)
    return s


def face_values(plan: RemapPlan, face: Float[Array, " n_plus_1"]) -> Float[Array, " capacity_plus_1"]:
    """A face-centred field on the new faces, linear in mass between the parent's faces.

    ``face`` holds all ``n + 1`` old faces, centre-out. A new face on an old face takes that
    face's value exactly; one at sub-cell point ``j`` of old cell ``i`` takes
    ``f_i + (j / k)(f_{i+1} - f_i)`` (sub-cells are equal in mass). Empty slots repeat the
    outer face, as in :func:`apply_plan`.
    """
    face = jnp.asarray(face)
    n = face.shape[0] - 1
    k = plan.subcells
    end = jnp.minimum(plan.start + plan.length, k * n)
    i, j = end // k, end % k
    lo = face[i]
    hi = face[jnp.minimum(i + 1, n)]
    right = jnp.where(plan.length > 0, lo + (j / k) * (hi - lo), face[-1])
    return jnp.concatenate([face[:1], right])


def _parts(total, slope, dm, k):
    """The ``k`` equal-mass sub-cell contents of one extensive field, interleaved to ``kn``.

    Under a linear profile in mass, sub-cell ``j`` of ``k`` holds
    ``Q/k + s dm^2 (2j + 1 - k) / (2 k^2)`` (``k = 2``, ``j = 0``: ``Q/2 - s dm^2/8``). The
    last sub-cell is the parent minus the others, so they sum to the parent to rounding.
    """
    tail = (None,) * (total.ndim - 1)
    j = jnp.arange(k - 1, dtype=total.dtype)
    shares = jnp.broadcast_to(
        (total / k)[:, None], (total.shape[0], k - 1) + total.shape[1:]
    )
    if slope is not None:
        coeff = ((2.0 * j + 1.0 - k) / (2.0 * k * k))[(None, slice(None)) + tail]
        shares = shares + (slope * (dm * dm)[(slice(None),) + tail])[:, None] * coeff
    rest = total - jnp.sum(shares, axis=1)
    stacked = jnp.concatenate([shares, rest[:, None]], axis=1)
    return stacked.reshape((k * total.shape[0],) + total.shape[1:])


def _run_sum(plan, total, parts):
    """Each slot's content: a sum of whole old cells when its run is cell-aligned (so one
    cell is a bitwise gather), otherwise a sum of sub-cells."""
    n = total.shape[0]
    k = plan.subcells
    expand = (slice(None),) + (None,) * (total.ndim - 1)
    aligned = (plan.start % k == 0) & (plan.length % k == 0)
    tail = (None,) * (total.ndim - 1)
    # Sub-cell terms: slots x max_length, masked beyond each run's length.
    j = jnp.arange(plan.max_length)
    sub = jnp.minimum(plan.start[:, None] + j[None, :], k * n - 1)
    use = (j[None, :] < plan.length[:, None])[(slice(None), slice(None)) + tail]
    part = jnp.sum(jnp.where(use, parts[sub], 0.0), axis=1)
    # Whole-cell terms for aligned runs; adding zeros is exact, so a one-cell run is a copy.
    c = jnp.arange(-(-plan.max_length // k))
    cell = jnp.minimum((plan.start // k)[:, None] + c[None, :], n - 1)
    use_c = ((k * c)[None, :] < plan.length[:, None])[(slice(None), slice(None)) + tail]
    whole = jnp.sum(jnp.where(use_c, total[cell], 0.0), axis=1)
    return jnp.where(aligned[expand], whole, part)


def apply_plan(
    plan: RemapPlan,
    xi_face: Float[Array, " n_plus_1"],
    dm: Float[Array, " n"],
    extensive: Any,
    slopes: Any = None,
    volume_slope: Float[Array, " n"] | None = None,
) -> tuple[Float[Array, " capacity_plus_1"], Float[Array, " capacity"], Any]:
    """Apply a plan to the faces, the masses and every extensive field.

    Args:
        plan: from :func:`plan_from_actions`, :func:`fixed_count_plan` or the caller.
        xi_face: all ``n + 1`` faces, centre-out, in a coordinate in which cell volume is
            linear (``r**3`` in spherical symmetry, ``x`` in slab symmetry).
        dm: cell masses.
        extensive: a PyTree of arrays with leading axis ``n`` (cell totals: momentum,
            energy, species masses, ...). Every leaf is moved with the mass.
        slopes: ``None`` for piecewise-constant specific values (children carry the
            parent's), or a PyTree whose leaves are slopes of the SPECIFIC value
            (leaf / dm) in mass, or ``None`` per leaf. :func:`limited_slopes` gives bounded
            ones.
        volume_slope: slope in mass of ``diff(xi_face) / dm``; ``None`` cuts the extent in
            equal parts.

    Returns:
        ``(xi_face_new, dm_new, extensive_new)`` with ``plan.capacity`` cells. Empty slots
        have zero mass and contents, and their faces repeat the outer face.
    """
    xi_face = jnp.asarray(xi_face)
    dm = jnp.asarray(dm)
    k = plan.subcells
    dm_new = _run_sum(plan, dm, _parts(dm, None, dm, k))

    def move(slope, total):
        return _run_sum(plan, total, _parts(total, slope, dm, k))

    if slopes is None:
        new_fields = jax.tree.map(lambda q: move(None, q), extensive)
    else:
        new_fields = jax.tree.map(move, slopes, extensive, is_leaf=lambda x: x is None)

    # Faces at sub-cell resolution: old faces at multiples of k, sub-cell boundaries between.
    n = dm.shape[0]
    extent = _parts(jnp.diff(xi_face), volume_slope, dm, k).reshape(n, k)
    inner = jnp.cumsum(extent[:, :-1], axis=1)
    sub_faces = jnp.concatenate(
        [xi_face[:-1, None], xi_face[:-1, None] + inner], axis=1
    )
    sub_faces = jnp.concatenate([sub_faces.reshape(-1), xi_face[-1:]])
    end = jnp.minimum(plan.start + plan.length, k * n)
    right = jnp.where(plan.length > 0, sub_faces[end], xi_face[-1])
    return jnp.concatenate([xi_face[:1], right]), dm_new, new_fields
