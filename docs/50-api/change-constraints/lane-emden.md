---
title: Lane-Emden self-gravitating spheres
---

# Lane-Emden self-gravitating spheres

## Owner import path

`jaxstro.numerics.lane_emden`

## Purpose

Integrate the Lane-Emden equation for a self-gravitating sphere in dimensionless
form and return its tabulated structure. Two branches are provided: the polytropic
equation `theta'' + (2/xi) theta' = -theta^n` and the isothermal (Bonnor-Ebert)
equation `psi'' + (2/xi) psi' = e^{-psi}`.

## Public records and callables

`solve_isothermal(...)` and `solve_polytrope(...)` return `LaneEmdenSolution`
(fields `xi`, `y`, `dy`, `m`, `dm`). `polytrope_xi1(...)` returns the scalar first
zero of `theta` -- the polytrope's outer edge -- as a differentiable event root.

## Shape and dtype expectations

Every `LaneEmdenSolution` field is a length-`n_points` floating array on a strictly
increasing dimensionless-radius grid that starts just off the origin at `XI_0 = 1e-6`.
The polytropic index `n` is a traced scalar; `xi_max` and `n_points` are static (they
size the output grid).

## JAX transforms and AD classification

`jit`, `vmap`, and `grad` are supported. The gradient in `n` flows through the
adaptive diffrax solve, and through `polytrope_xi1` via the implicit function theorem
on the `diffrax.Event` root -- not through a grid `argmin`. `xi_max` and `n_points`
are static and must not be differentiated.

## Failure behavior

Past the first zero the polytropic source `theta^n` is floored at zero and is no
longer physical; callers integrate only to `xi_1` (`solve_polytrope(n, xi_max=xi_1)`).
For `n >= 5` the sphere has infinite extent, no event fires, and `polytrope_xi1` runs
to `xi_search_max`; that regime must be rejected by the caller.

## Contract and evidence links

See the generated provenance cards in
[](../research-infrastructure/source-provenance/lane_emden.md) and the owner tests in
`tests/unit/test_lane_emden.py` and `tests/validation/test_lane_emden_gradients.py`.

## Canonical import example

```python
from jaxstro.numerics.lane_emden import solve_isothermal, solve_polytrope, polytrope_xi1
```
