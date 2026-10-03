---
title: Split and merge on a Lagrangian mesh
description: >-
  Changing the resolution of a one-dimensional mass-coordinate mesh by splitting
  cells at their half-mass points and merging neighbours, with exact copies of
  untouched cells, conservation to rounding, and a frozen plan for gradients.
---

## The question this method answers

A Lagrangian mesh follows the mass: each cell keeps its mass, and its faces move with
the fluid. The resolution a simulation needs moves too. A collapsing cloud needs cells
at an accretion shock that sweeps through the mass coordinate; a star needs cells where
a burning shell or a convective boundary sits. How can a researcher add cells where they
are needed and remove them where they are not, without disturbing the cells that need no
change, without losing conservation, and without breaking the gradient of the run?

`jaxstro.numerics.lagrangian_remap` answers this for one dimension. It owns the
operation only: which cells are split or merged is decided by the caller's criterion
(a shock sensor in hydrax, MESA's mesh functions in stellax).

:::{important}
The operation is local. A cell that is neither split nor merged is copied bitwise. A
global remap that moves every face, even slightly, changes every cell; in stellax that
injected sign-alternating entropy errors of up to $|dS/S| = 1.6\times10^{-4}$ into a
stellar core (stellax design note, 2026-10-02). Exact copies are the reason this method
exists.
:::

## Before computation: what should be true?

- The mesh carries all $n+1$ faces, ordered centre-out, in a coordinate $\xi$ in which
  cell volume is linear: $\xi = r^3$ in spherical symmetry, $\xi = x$ in slab symmetry.
- Cell masses $\Delta m_i > 0$ and every conserved field given as a cell **total**
  (momentum, energy, species mass), not as a density or a specific value.
- The caller decides which cells to split or merge and how often. A split adds no
  information: the children inherit the parent's content, shared by a reconstruction.
  It adds the ability to represent structure that the dynamics will create next.

## Define the mathematical objects

**Half cells.** Old cell $i$ is cut at its half-mass point into an inner half cell
$2i$ and an outer half cell $2i+1$, each of mass $\Delta m_i/2$. A new cell is a
contiguous run of half cells. Its boundaries therefore lie on old faces or on old
half-mass points, which is MESA's rule (at most two subcells per old cell).

**Plan.** A `RemapPlan` lists, for each new slot, the first half cell (`start`) and the
number of half cells (`length`): a kept cell is a run of 2 starting on an even index, a
split child is a run of 1, a merge of two whole cells is a run of 4. Runs tile the old
mesh in order. The plan is integer data with a static capacity. A plan built with
`subcells=4` cuts every old cell into four equal-mass parts instead, so a half cell can be
halved again; MESA's surface pass needs this once in about 230,000 faces of a 1 Msun
pre-main-sequence to TAMS run (stellax, 2026-10-03). Sub-cell $j$ of $k$ holds
$Q/k + s\,\Delta m^2 (2j + 1 - k)/(2k^2)$, which is the formula below for $k = 2$.

## Derive the method

Let the specific value $q = Q/\Delta m$ be reconstructed linearly in mass across cell
$i$, $q(m) = \bar q_i + s_i (m - m_{c,i})$, with $m_{c,i}$ the cell's mass centre. The
inner half covers $m \in [m_{c,i} - \Delta m_i/2,\ m_{c,i}]$, so its content is

```{math}
:label: eq-half-cell-share
Q_{2i} = \int_{-\Delta m/2}^{0} (\bar q + s\,\mu)\,d\mu
       = \tfrac12 Q_i - \tfrac18 s_i\,\Delta m_i^2,
\qquad Q_{2i+1} = Q_i - Q_{2i}.
```

The outer half is formed as the difference, so the halves sum to the parent to
rounding. With $s_i = 0$ the share is equal and the children carry the parent's
specific value, which is MESA's rule for a split. The half-mass face uses the same
formula with $Q \to \Delta\xi_i$ and $s \to s_v$, the slope of the specific extent
$\Delta\xi/\Delta m$ (the specific volume up to the geometry constant).

A new cell's content is the sum of its half cells. When the run starts and ends on old
faces it is computed instead as the sum of the old totals, so a one-cell run is the old
value itself (a gather, bitwise) and a two-cell merge is $Q_i + Q_{i+1}$.

**Slopes.** `limited_slopes` uses the monotonised-central limiter in mass,
$s_i = \operatorname{minmod}(2 s_L, 2 s_R, s_C)$, zero in the end cells and where the
one-sided slopes differ in sign. The half-cell mean is $\bar q_i \mp s_i\Delta m_i/4$,
and $|s_i|\Delta m_i/4 \le |\bar q_{i+1} - \bar q_i|\,\Delta m_i/(\Delta m_i + \Delta m_{i+1})$,
so each child's mean lies between the neighbouring means and a positive specific value
stays positive. For mass fractions, `joint=True` scales every component's central slope
by one factor per cell, so the slopes sum to zero and the children keep
$\sum_k X_k = 1$.

**Predicted accuracy.** Take a smooth monotone $q$ on a uniform mesh of spacing $h$.
Expanding about the cell centre, the cell mean is $q + q''h^2/24 + O(h^4)$ and the inner
half-cell mean is $q - q'h/4 + q''h^2/24 - q'''h^3/192 + O(h^4)$. Piecewise-constant
reconstruction gives the cell mean to both children, an error of $q'h/4$: **first
order**. The central slope of the cell means is $q' + \tfrac{5}{24} q''' h^2 + O(h^4)$,
so the linear rule's child mean is $q - q'h/4 + q''h^2/24 - \tfrac{5}{96}q'''h^3$. The
$q''$ terms agree, and the error is $\tfrac{3}{64}\,q'''h^3$: **third order**.

## What the algorithm actually does

```{list-table} Public callables
:header-rows: 1

* - Callable
  - Role
* - `plan_from_actions(action, capacity)`
  - per-cell `KEEP`, `SPLIT`, `MERGE_RIGHT` to a plan, by an exclusive cumulative sum
    of emit counts (static shapes)
* - `fixed_count_plan(split_score, merge_score, ...)`
  - equal numbers of splits and merges, so $N$ is unchanged
* - `limited_slopes(specific, dm, joint=False)`
  - MC slopes of a specific value in mass
* - `apply_plan(plan, xi_face, dm, extensive, slopes, volume_slope)`
  - the transfer: faces, masses and every leaf of a PyTree of totals
* - `RemapPlan`
  - the record; `is_copy`, `is_split`, `is_merge`, `is_valid(n_old)`
```

`fixed_count_plan` marks a cell as a split candidate when its score exceeds a threshold
and a neighbour pair as a merge candidate when its score is below a second, lower
threshold (hysteresis), excluding pairs that touch a split candidate. A pair is kept
only if its score is a strict local minimum among the overlapping pairs, ordered by
score and then by index; two overlapping pairs cannot both pass, so the merges are
disjoint without a sequential walk. It then applies the $k$ strongest splits and the $k$
weakest merges, with $k$ the smaller candidate count bounded by a static `max_changes`.
Optional masks exclude cells (a sink cell, a pinned face), and an optional cell-ratio
limit (MESA uses 2.5) forces a split of a cell larger than that ratio times a neighbour
and blocks a merge that would create one.

A caller with its own planner builds a `RemapPlan` directly; stellax's port of MESA's
surface-to-centre walk is that case. A padded capacity larger than the cell count is
allowed: empty slots have zero mass and repeat the outer face.

## Measured accuracy

:::{figure} ../figures/lagrangian-remap-contracts.webp
:name: fig-lagrangian-remap-contracts
:alt: Left, a tanh front on 32 cells before and after four fixed-count regrids, with the faces concentrated at the front. Right, the split error against N for piecewise-constant and limited-linear reconstruction, with order 1 and order 3 guide lines.

Left: a front $q = 1 + \tanh((m - 0.5)/0.02)$ on 32 uniform-mass cells (slab,
unit density), after four passes of `fixed_count_plan` (split above a neighbour jump of
0.1, merge below 0.02, at most 4 changes per pass, ratio limit 2.5). The cell count
stays 32; the face ticks show the cells moving to the front. Right: maximum relative
error of the inner child's mean for $q = e^{2m}$ when every cell is split, interior
cells.
:::

```{list-table} Split error for q = exp(2m), every cell split, interior cells (measured 2026-10-03)
:header-rows: 1

* - Cells $N$
  - piecewise constant, max rel. error
  - order
  - limited linear, max rel. error
  - order
* - 32
  - 1.59e-2
  -
  - 1.16e-5
  -
* - 64
  - 7.87e-3
  - 1.011
  - 1.44e-6
  - 3.012
* - 128
  - 3.92e-3
  - 1.006
  - 1.80e-7
  - 3.006
* - 256
  - 1.96e-3
  - 1.003
  - 2.24e-8
  - 3.003
* - 512
  - 9.78e-4
  - 1.001
  - 2.80e-9
  - 3.001
```

[](#fig-lagrangian-remap-contracts) shows both effects: cells gathering at a front at fixed
count, and the split error falling at the predicted orders. Both orders match the
prediction above. The half-mass face position converges at
1.000 and 3.000. Conservation of mass, extent and every field holds to $10^{-14}$
relative on a random non-uniform mesh with splits and merges; the transfer's AD
derivative matches central differences to $3.3\times10^{-13}$ relative
(`tests/validation/test_lagrangian_remap_order.py`).

## What JAX differentiates

The plan is integers and has no derivative. The decision to split or merge is a
discrete event: perturbing the inputs can flip it, and no smooth derivative exists
across that flip. A gradient through a run that regrids is the derivative at fixed
topology: record each plan in the forward pass and replay the same plans
(the frozen hierarchy). `apply_plan` is then a fixed composition of gathers, sums and
products, smooth in the faces, the masses and the fields. `limited_slopes` is
piecewise smooth; selections use `jnp.where`, so at a limiter switch the whole
derivative goes to the selected branch. Every loop is static (`max_length` is at most
4), so the transfer runs inside a `lax.scan` step.

## Using it in Jaxstro

```python
import jax.numpy as jnp

from jaxstro.numerics import lagrangian_remap as lr

xi_face = jnp.linspace(0.0, 1.0, 9) ** 3          # spherical: xi = r**3
dm = jnp.full(8, 0.125)
energy = dm * jnp.array([1.0, 1.0, 1.0, 4.0, 9.0, 9.0, 9.0, 9.0])

q = energy / dm
jump = jnp.abs(jnp.diff(q))
split_score = jnp.concatenate([jump[:1], jnp.maximum(jump[:-1], jump[1:]), jump[-1:]])
merge_score = jump + jnp.concatenate([jump[1:], jnp.zeros(1)])
plan = lr.fixed_count_plan(split_score, merge_score, 2.0, 0.5, max_changes=2)

xi_new, dm_new, energy_new = lr.apply_plan(
    plan, xi_face, dm, energy, lr.limited_slopes(q, dm),
    lr.limited_slopes(jnp.diff(xi_face) / dm, dm),
)
assert jnp.allclose(jnp.sum(energy_new), jnp.sum(energy), rtol=1e-14)
assert dm_new.shape == dm.shape
assert bool(plan.is_valid(8))
```

## How to audit the result

1. Check `plan.is_valid(n_old)` and `plan.n_active <= plan.capacity` before applying a
   plan built by hand.
2. Check that the cells the plan keeps are bitwise equal to their old values
   (`plan.is_copy`), faces included.
3. Check the totals of mass, extent and every field to rounding, and the outer face
   exactly.
4. For a field that must stay positive, check the children; with independent limiting
   check the sum of mass fractions, or use `joint=True`.
5. For a gradient, record the plans and replay them, and compare AD with central
   differences at fixed plans; an FD step that flips a plan is not a valid reference.

## Where the claim stops

- One dimension only. Multi-dimensional refinement needs a block or tree hierarchy and
  is not provided.
- A split cuts a cell at its half-mass point only; a cell can be split into at most two
  per regrid, and repeated regrids refine further.
- No refinement criterion, no equation of state, no field semantics. Quantities that are
  not linear in the transferred totals (temperature from internal energy, kinetic energy
  from momentum) are the caller's: a split shares momentum and total energy with the
  same mass fraction under piecewise-constant reconstruction, so each child keeps its
  parent's velocity and specific internal energy, but a limited-linear momentum share
  changes the kinetic energy and must be paired with an energy rule by the caller.
- Potential-energy changes from moving mass in a gravitational field are not
  corrected (MESA caps them at 5 %; stellax applies its own correction).

## Connected ideas

:::{seealso}
The fixed-mesh remap and finite-volume stencils are in
[](./meshes.md); the API is [](../../50-api/discrete-space/lagrangian-remap.md), and
validation evidence is indexed from [](../../60-validation/validation.md).
:::
