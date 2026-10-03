---
title: Lagrangian split and merge
---

# Lagrangian split and merge

## Owner import path

`jaxstro.numerics.lagrangian_remap`

## Purpose

Split and merge of cells on a one-dimensional mass-coordinate mesh: an integer plan
record, planners, bounded slopes and a conservative transfer that copies untouched
cells bitwise. No refinement criterion and no physics.

## Public records and callables

`RemapPlan` (`start`, `length`, `n_active`, static `max_length` and `subcells`; `is_copy`,
`is_split`, `is_merge`, `is_valid`), `plan_from_actions`, `fixed_count_plan`,
`limited_slopes`, `apply_plan`, and the action codes `KEEP`, `SPLIT`, `MERGE_RIGHT`.

## Shape and dtype expectations

`xi_face` has shape `(n + 1,)`, centre-out, in a coordinate linear in volume; `dm`
has shape `(n,)`; every leaf of `extensive` has leading axis `n` and is a cell total.
Plans have a static capacity; `fixed_count_plan` returns capacity `n`. Old cell `i`
is sub-cells `k i` to `k i + k - 1` of equal mass, `k = subcells` (2 by default; 4 lets a
half cell be halved again, with `max_length = 8`).

## JAX transforms and AD classification

Planning and transfer trace under `jit` and `vmap` with static shapes and static
loops of at most `max_length` (4) terms. The plan is a discrete, non-differentiable
record; replay a recorded plan for gradients. `apply_plan` is smooth in faces,
masses and fields at a fixed plan; `limited_slopes` is piecewise smooth with
`jnp.where` branch selection.

## Failure behavior

`fixed_count_plan` raises on a mismatched `merge_score` shape, on `max_changes`
outside `[1, n // 2]`, and when only one of `size` and `max_ratio` is given. A plan
that overflows its capacity reports `n_active > capacity`; a hand-built plan should be
checked with `is_valid(n_old)`, because traced code cannot raise on it.

## Contract and evidence links

See [](../../20-methods/discrete-space/lagrangian-remap.md) and
[](../../60-validation/validation.md). Evidence: `tests/unit/test_lagrangian_remap.py`,
`tests/validation/test_lagrangian_remap_order.py`.

## Canonical import example

```python
from jaxstro.numerics.lagrangian_remap import apply_plan, fixed_count_plan
```
