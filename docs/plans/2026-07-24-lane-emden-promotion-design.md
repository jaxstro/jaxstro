# Promote the Lane-Emden solver to jaxstro (shared foundation)

**Date:** 2026-07-24
**Status:** Design approved (Anna), ready to implement
**Scope:** jaxstro (new module), progenax (repoint + delete), hydrax (consume)

## Problem

hydrax M2 (self-gravity) needs the **isothermal Lane-Emden** solution to build a
Bonnor-Ebert sphere as the hydrostatic static-gate initial condition. progenax already
has a mature, differentiable Lane-Emden solver
(`experimental/gravoturb/profiles/lane_emden.py`), but it lives inside progenax's
experimental gravoturb package.

hydrax must **not** import it directly:

- The gravoturb design has **progenax call hydrax** (gravoturb ages its gas by evolving
  it in hydrax). So the intended dependency arrow is **progenax → hydrax**. If hydrax
  imported progenax, that would create a **cycle** the moment M4 coupling lands.
- It also lives in `experimental/` (unstable API) and would drag progenax's whole
  experimental surface into hydrax for one validation IC.

## Decision

**Promote the Lane-Emden solver to `jaxstro`** — the shared foundation both packages
already depend on (progenax's `bonnor_ebert.py` *already* imports `jaxstro.numerics.*`).
This is the DRY, one-way-arrow solution: one canonical implementation, hydrax and
progenax both consume it, no cycle.

Two sub-decisions, both settled with Anna:

1. **Backend = move the code as-is (diffrax + optimistix).** jaxstro currently depends
   on neither; adding them was the only real objection to a literal move, and Anna
   waived it ("install is fine"). Moving the proven diffrax solver (adaptive `Tsit5`,
   `rtol=1e-8`, gradients known to flow — same idiom as `profiles/king`) is lower-risk
   than reimplementing on jaxstro's fixed-step RK4, gives better accuracy for free, and
   standardises diffrax as jaxstro's canonical adaptive ODE backend. jaxstro's own
   `numerics/ode.py` (fixed-step RK4/scan) stays for lightweight in-house ODEs.
2. **Scope = full module** (isothermal + polytropic + `polytrope_xi1`). It is a single
   file carrying both branches; moving the whole thing achieves complete DRY and lets
   *both* progenax `BonnorEbertProfile` **and** `PolytropeProfile` delegate to jaxstro,
   fully retiring progenax's copy.

## What moves, and where

- `progenax/.../gravoturb/profiles/lane_emden.py` → **`jaxstro/src/jaxstro/numerics/lane_emden.py`**
  (verbatim solver: `LaneEmdenSolution`, `solve_isothermal`, `solve_polytrope`,
  `polytrope_xi1`, plus the private RHS/series-seed helpers).
  - **Home rationale:** jaxstro already keeps a specific-physics equation solver in
    `numerics/` — `numerics/kepler.py` (Kepler's equation). `numerics/lane_emden.py`
    sits right beside it. Export via `numerics/__init__.py`.
- `progenax/tests/experimental/unit/test_lane_emden.py` → **jaxstro tests** (adapt the
  import path; it *is* the test suite for the promoted code).
- **Deps:** add `diffrax` and `optimistix` to jaxstro `pyproject.toml` core deps.
- **Provenance:** register the equation (isothermal + polytropic Lane-Emden) and impl in
  jaxstro's provenance system, with an **arrival gate** (see Testing).

## progenax repoint (exact callsites, from grep)

Source (3):
- `profiles/polytrope.py:26` — `from jaxstro.numerics.lane_emden import polytrope_xi1, solve_polytrope`
- `profiles/bonnor_ebert.py:41` — `from jaxstro.numerics.lane_emden import solve_isothermal`
- `profiles/__init__.py:23` — re-export the 4 symbols from jaxstro (decide: keep the
  gravoturb re-export as a convenience, or update the one test that uses it). Since
  CLAUDE.md forbids compat shims, prefer updating `test_profiles_gradients.py` to import
  `polytrope_xi1` from jaxstro and drop the re-export — confirm during implementation.

Then **delete** `profiles/lane_emden.py`. progenax keeps its own `_scaling.py`
(`half_mass_xi`, `interp_flat`, `require`) — gravoturb-specific, does not move.

No new dependency direction: progenax → jaxstro already exists. progenax keeps `diffrax`
(other profiles — king/michie/limepy — use it directly).

## hydrax consumption (M2 Task 4)

hydrax `lagrangian` builds the **physical** BE-in-hydro-units IC from
`jaxstro.numerics.lane_emden.solve_isothermal`: scale `r = ξ r_0`,
`r_0 = c_s / sqrt(4π G ρ_c)`, build the Lagrangian mass grid + isothermal state truncated
at `ξ_max`, with confining `P_ext = c_s² ρ(R)`. hydrax does **not** reuse progenax's
`BonnorEbertProfile` wrapper (unit-mass normalisation + `r_h` parametrisation is
IC-sampling machinery hydrax doesn't need).

## Testing / arrival gate

- **jaxstro:** the moved `test_lane_emden.py` must pass unchanged (behaviour-preserving
  move). Provenance gate: small-ξ series (`ψ ≈ ξ²/6 − ξ⁴/120`) and the literature
  critical Bonnor-Ebert constants (ξ_crit ≈ 6.451, contrast ≈ 14.04, mass coeff ≈ 1.18)
  as **evidence**, and polytrope analytic zeros (`ξ_1 = √6` at n=0, `π` at n=1).
- **progenax (regression gate):** `test_bonnor_ebert.py`, `test_profiles_gradients.py`,
  and the polytrope tests must all still pass after the repoint — this proves the DRY
  swap is behaviour-preserving.
- **hydrax:** M2 Task 4/6 (BE profile + static gate) consume the jaxstro solver.

## Rollout order (3 repos, feature branch in each)

1. **jaxstro** `feat/lane-emden-promotion`: add module + tests + deps + provenance; green.
2. **progenax** `feat/lane-emden-from-jaxstro`: repoint 3 imports, delete `lane_emden.py`,
   update/decide `__init__` re-export; run experimental test suite green.
3. **hydrax** `m2-self-gravity`: M2 Task 4 consumes jaxstro; continue M2 plan.

Code review after jaxstro (step 1) and after progenax (step 2), before hydrax builds on
top. jaxstro and progenax are on `main` and get their own feature branches; progenax has
an unrelated dirty `uv.lock` (left alone).
