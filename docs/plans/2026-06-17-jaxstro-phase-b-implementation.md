# jaxstro Baseline + Phase B (Consolidation) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this
> plan task-by-task (fresh subagent per task + code-review subagent between tasks). Apply the
> research-workflow gates named per task (gradient-validation, numerical-method-validation,
> provenance-of-constants, verification-gate). Ratified strategy:
> `docs/plans/2026-06-17-jaxstro-consolidation-and-release-strategy.md`.

**Goal:** Establish a release-grade build baseline, then consolidate the ecosystem's generic
differentiable numerics into jaxstro (parity-preserving), landing `PhotometricUnits` first to
unblock fluxax Phase 2.

**Architecture:** Thin SoTA foundation (D2). Hoist generic primitives into `jaxstro.numerics` /
`jaxstro.testing`; migrate sibling callsites in *separate, parity-preserving* commits that keep the
numerics byte-identical and run the sibling suites. JAX-native throughout; every differentiable
public function FD-vs-AD grad-checked.

**Tech Stack:** JAX (`jax.numpy`, `lax.scan`, `jit`/`vmap`/`grad`), equinox, jaxtyping, hatchling,
uv, pytest. Constants host-side; numerics differentiable.

**Branch:** `feature/consolidate-harden-release` (already created; baseline `c97b80d`).

**Working dirs:** jaxstro `~/projects/jaxstro-dev/jaxstro`; siblings `../progenax`, `../fluxax`.

---

## Conventions (every task)

- **Verify command** (after T0): `env -u VIRTUAL_ENV uv run --no-sync pytest <paths> -q`.
  Before T0 completes, use: `PYTHONPATH=src /Users/anna/miniforge3/envs/astro/bin/python -m pytest …`.
- **TDD**: write the failing test → run it red → minimal implementation → run green → commit.
- **Never weaken a test/tolerance to pass.** Fix the root cause (CLAUDE.md).
- **Parity discipline (hoists)**: jaxstro implementation must reproduce the origin's numerics; add a
  parity test comparing jaxstro output to the original function on shared inputs *before* deleting the
  origin. Sibling migration is a **separate commit** that runs the sibling's suite.
- **Provenance**: every constant/coefficient/formula carries a comment citing its authority
  (gate: `provenance-of-constants`).
- **Commit trailer** ends with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
  Stage files explicitly (never `git add -A`). Do NOT push/merge — Anna's separate words only.
- **Grad-check** = finite-difference vs `jax.grad`, prefer `jacrev`; never `jacfwd`/`hessian`
  through a custom_vjp/diffrax path (gate: `gradient-validation`).

---

## Task 0 — Release-grade build baseline (D6, D7)

**Files:**
- Modify: `pyproject.toml`
- Replace: `LICENSE` (BSD-3-Clause → Apache-2.0; copy text from `../progenax/LICENSE`)
- Create: `src/jaxstro/py.typed` (empty)
- Create: `uv.lock` (generated)

**Steps:**
1. `pyproject.toml`: switch `[build-system]` to hatchling
   (`requires = ["hatchling"]`, `build-backend = "hatchling.build"`); add
   `[tool.hatch.build.targets.wheel] packages = ["src/jaxstro"]`.
2. `license = "Apache-2.0"` + `license-files = ["LICENSE"]`; replace `LICENSE` with Apache-2.0 text
   (verify it's the standard Apache-2.0 body, copyright "Anna Rosen").
3. Classifiers: add `"Development Status :: 4 - Beta"`, `"Intended Audience :: Science/Research"`,
   `"Programming Language :: Python :: 3.10/3.11/3.12/3.13"`, `"Topic :: Scientific/Engineering ::
   Astronomy"`; remove the BSD classifier; add `keywords = ["jax","astrophysics","units","constants",
   "coordinates","numerics","differentiable"]`.
4. Add `equinox>=0.11.0` to `dependencies` (D3).
5. Fix the uv config: replace the broken `[tool.uv] default-groups = ["dev"]` with a valid
   `[dependency-groups] dev = [...]` (PEP 735) OR move dev deps there; ensure `uv` parses.
6. Ensure `py.typed` ships: with hatchling + `packages=["src/jaxstro"]` it's included; confirm.
7. Generate the lock + env: `env -u VIRTUAL_ENV uv lock` then `env -u VIRTUAL_ENV uv sync`.
8. **Verify**: `env -u VIRTUAL_ENV uv run --no-sync pytest -q` → full existing suite green
   (~230 tests). `env -u VIRTUAL_ENV uv run --no-sync python -c "import jaxstro; print(jaxstro.__version__)"`.
9. **Commit** (`pyproject.toml LICENSE src/jaxstro/py.typed uv.lock`):
   `build: hatchling + Apache-2.0 + py.typed + classifiers + working uv lock`.

**Gate:** `verification-gate` (paste the green suite + import as evidence).

---

## Task 1 — PhotometricUnits + photometric constants (UNBLOCKS fluxax Phase 2 — priority)

Per `../fluxax/docs/plans/2026-06-17-photometric-units-design.md` (read it first).

**Files:**
- Modify: `src/jaxstro/constants.py` (add `JY_CGS`, `AB_ZEROPOINT_JY`, `AB_ZEROPOINT_CGS`)
- Create: `src/jaxstro/units.py` addition or `src/jaxstro/photometric.py` — `PhotometricUnits`
  dataclass + presets `SOLAR_PHOTOMETRIC`, `CGS_PHOTOMETRIC` (match the fluxax design doc's chosen
  module location/name).
- Test: `tests/test_photometric.py`

**Steps:**
1. **Failing test**: `JY_CGS == 1e-23` (erg s⁻¹ cm⁻² Hz⁻¹), `AB_ZEROPOINT_JY == 3631.0`,
   `AB_ZEROPOINT_CGS == 3.631e-20`; `PhotometricUnits` resolves luminosity/radius/flux-density choices
   to constant multiples; presets exist. Run red.
2. Implement constants with provenance comments (**Oke & Gunn 1983**, ApJ 266, 713) and the
   `PhotometricUnits` dataclass (parallel to `UnitSystem`, host-side floats → constant multiplies).
3. **Grad-check**: a flux conversion using `PhotometricUnits` is differentiable (FD vs AD).
4. Run green; export from `jaxstro.units`/top-level per the design doc; update `__all__`.
5. **Commit**: `feat(units): PhotometricUnits + Jy/AB photometric constants (Oke & Gunn 1983)`.

**Gates:** `provenance-of-constants`, `gradient-validation`.
**Note:** fluxax migration to consume these is a *fluxax-side* change, tracked separately (do not
edit fluxax in this task beyond confirming import compatibility).

---

## Task 2 — Reconcile `cumulative_trapz` (dx-outside) + progenax migration

**Files:**
- Modify: `src/jaxstro/numerics/integration.py` (`cumulative_trapz`)
- Test: `tests/test_numerics.py` (extend), new `tests/test_integration_parity.py`
- (separate commit) Modify: `../progenax/src/progenax/numerics.py` + callsites

**Steps:**
1. **Failing parity test**: jaxstro `cumulative_trapz(y, dx=h)` (uniform) equals progenax's
   `cumulative_trapezoid(y, dx=h)` **dx-outside** (cumsum-then-multiply) byte-for-byte; and the
   `x`-array path still works for non-uniform grids. Run red.
2. Reimplement the uniform path as **dx-outside** (cumsum first, multiply after); keep `x`-array
   (non-uniform) path. Document the ~1-ulp relationship to the old dx-inside path.
3. Grad-check `cumulative_trapz` (FD vs AD) both paths.
4. Run green (jaxstro). **Commit**: `refactor(numerics): cumulative_trapz uniform path → dx-outside`.
5. **Separate commit (progenax)**: replace progenax's local `cumulative_trapezoid` with
   `from jaxstro.numerics.integration import cumulative_trapz` (alias if needed); update the 21
   callsites. Run progenax suite: `cd ../progenax && env -u VIRTUAL_ENV uv run --no-sync pytest -q`.
   Allow the documented ~1-ulp drift at the 3 former dx-inside sites (their existing test budgets).
   **Commit in progenax**: `refactor(numerics): use jaxstro cumulative_trapz (parity)`.

**Gate:** `numerical-method-validation` (convergence/accuracy), parity evidence.

---

## Task 3 — Hoist `inverse_cdf_draw` + progenax migration

**Files:**
- Create: `src/jaxstro/numerics/sampling.py` (`inverse_cdf_draw`) + export in `numerics/__init__.py`
- Test: `tests/test_sampling.py`
- (separate commit) progenax callsites (16, 6 files)

**Steps:**
1. **Failing test**: `inverse_cdf_draw(weight, grid, unif, reg=1e-30)` reproduces progenax semantics
   (uniform-grid trapezoid CDF, `+reg` guard → finite at zero total weight, `jnp.interp`); parity vs
   the progenax original on shared inputs. Run red.
2. Implement (pure JAX, differentiable); provenance/algorithm note (inverse-CDF / PPF sampling).
3. **Grad-check** w.r.t. `weight` and `unif` (FD vs AD).
4. Run green. **Commit**: `feat(numerics): hoist differentiable inverse_cdf_draw`.
5. **Separate commit (progenax)**: import from jaxstro, update 16 callsites, run progenax suite.

**Gate:** `gradient-validation`, parity evidence.

---

## Task 4 — Hoist generic Newton-PPF base (D8) + progenax migration

**Files:**
- Modify: `src/jaxstro/numerics/rootfinding.py` (add generic `newton_ppf` w/ initial guess + fixed
  iterations) + export
- Test: `tests/test_numerics.py` (extend)
- (separate commit) `../progenax/src/progenax/imf/base.py` — use jaxstro; **keep**
  `_ppf_newton_chabrier` local in `chabrier.py`.

**Steps:**
1. **Failing test**: generic fixed-iteration Newton-PPF solves a known CDF→PPF (e.g. exponential)
   to tolerance; differentiable w.r.t. `u` and params. Run red.
2. Implement generic base (decoupled from IMF; takes logpdf/cdf/bounds or callables); `lax.scan`,
   safe division, clip to bounds.
3. Grad-check. Run green. **Commit**: `feat(numerics): generic Newton-PPF base solver`.
4. **Separate commit (progenax)**: `BaseIMF.ppf` uses jaxstro base; Chabrier specialization stays.
   Run progenax suite + IMF distribution validation.

**Gate:** `gradient-validation`, `numerical-method-validation`.

---

## Task 5 — Quadrature factory (D5) + progenax-experimental migration

**Files:**
- Create: `src/jaxstro/numerics/quadrature.py` — `gauss_legendre_nodes(n)`, `gauss_hermite_nodes(n)`
  (probabilists'), `hermite_e_basis(g, n_max)`, `hermite_coefficients(map_fn, n_max, n_quad)`
  + export
- Test: `tests/test_quadrature.py`
- (separate commit) `../progenax/src/experimental/gravoturb_fdf/theory/gaussianization.py` + the 1
  import in `inference/projected_logp.py`; keep `bm19_*` wrappers local.

**Steps:**
1. **Failing tests**: GL nodes integrate polynomials exactly to degree `2n-1`; GH (probabilists')
   integrate Gaussian moments exactly; Hermite-e recurrence correct; `hermite_coefficients` matches a
   known expansion. Parity vs progenax `_gauss_hermite` (byte-identical — same numpy call). Run red.
2. Implement: host-side node generation (numpy `leggauss`/`hermgauss`, frozen to `jnp.asarray`,
   static `n`) → constants; provenance (**Golub & Welsch 1969**, numpy polynomial). Document *why*
   host-side gen is correct/performant (setup-only; grad flows through integrand values).
3. **Grad-check**: `hermite_coefficients` differentiable w.r.t. params inside `map_fn` (FD vs AD).
4. Run green. **Commit**: `feat(numerics): Gaussian quadrature factory (GL/GH + Hermite basis)`.
5. **Separate commit (progenax)**: experimental `gaussianization.py` imports jaxstro; run the
   experimental suite (`PYTHONPATH`/`[experimental]`).

**Gates:** `numerical-method-validation`, `gradient-validation`, `provenance-of-constants`.

---

## Task 6 — Grad-check utility → `jaxstro.testing` (dedup verbatim core)

**Files:**
- Create: `src/jaxstro/testing/__init__.py`, `src/jaxstro/testing/grad_audit.py`
  (`audit_entry_point`, `Case`, `AuditResult`, `EdgeConfig`)
- Test: `tests/test_grad_audit.py`
- (separate commits) fluxax + progenax: replace their `tests/validation/grad_audit/core.py` with an
  import from `jaxstro.testing`; keep per-package case registries local.

**Steps:**
1. **Failing test**: `audit_entry_point` classifies a clean/known-zero/known-blocked case correctly
   (port the existing `core.py` test). Run red.
2. Move the verbatim engine into `jaxstro.testing.grad_audit` (no behavior change).
3. Run green. **Commit**: `feat(testing): grad-check audit engine (dedup fluxax/progenax core)`.
4. **Separate commits**: fluxax + progenax import from jaxstro; run **fluxax 338-gate** and progenax
   grad-audit suite to confirm parity.

**Gate:** `verification-gate`.

---

## Task 7 — Harden existing numerics to release-grade (split into sub-commits)

**Files (per sub-task, test-first each):**
- `tests/test_linear_algebra.py` (NEW) — cover `norm2`, `project_onto`, `condition_number`; fix
  `project_onto(eps=0)` NaN (default `eps>0` or guard); document `condition_number` non-diff.
- `tests/test_rng.py` (NEW) — shape/type/determinism for `split_key`, `split_tree`, `fold_in_indices`.
- `tests/test_numerics.py` — add `compensated_*` **accuracy** test (catastrophic cancellation, e.g.
  `[1e16,1,-1e16,1] → 2.0`); add `simpson` tests + uniform-spacing validation; `interp1d`
  monotonic-x validation.
- FD-vs-AD grad-checks added for every differentiable public numeric (`stats`, `interpolation`,
  `integration`, `rootfinding`, `quadrature`, `sampling`, `compensated`, `linear_algebra`).
- Provenance pass in `constants.py`: fix the **Julian-vs-tropical year** comment (`3.15576e7 s` is the
  Julian year); add per-constant cites where missing (`A_RAD`, `SIGMA_SB`); cite or mark-original the
  `fill_bins` reservoir + two-stencil heuristics in `spatial/`.
- Upgrade `numerics/*` to full jaxtyping annotations.

**Process:** each bullet is its own red→green→commit cycle. Suggested commits:
`test(numerics): cover linear_algebra + fix project_onto NaN`,
`test(numerics): rng + compensated accuracy + simpson`,
`test(numerics): FD-vs-AD grad-checks across differentiable primitives`,
`fix(constants): correct Julian-year provenance + per-constant citations`,
`refactor(numerics): full jaxtyping annotations`.

**Gates:** `gradient-validation`, `numerical-method-validation`, `provenance-of-constants`.

---

## Task 8 — Constants round-out (D11)

**Files:** Modify `src/jaxstro/constants.py`; Test `tests/test_constants.py` (extend).

**Steps:**
1. **Failing test**: new constants present with correct CODATA values (e.g. Thomson cross-section
   `SIGMA_T = 6.6524587e-25` cm², fine-structure `ALPHA_FS = 7.2973525693e-3`). Run red.
2. Add each with a provenance comment to its CODATA 2018 source; add to `__all__`.
3. Run green. **Commit**: `feat(constants): round out fundamental constants (CODATA 2018)`.

**Gate:** `provenance-of-constants`.

---

## After Phase B (separate plans — do NOT start without Anna)
- **`jaxstro.quantity`** (D13): own design brainstorm (strategy §10) → its own plan. 0.1.0 only if no
  delay, else 0.2.0.
- **Phase C** — release hardening: 3-tier test reorg + markers, CI (`.github/workflows`: pytest+ruff+
  mypy, `JAX_ENABLE_X64=1`, `uv lock --check`, sibling smoke-test), MyST docs site (strategy §9),
  progenax-grade `CLAUDE.md`, `CHANGELOG`.
- **Phase D** — release prep: full gate + sibling smoke-tests green; STATUS+brain; **Anna's merge-go
  → push word → tag**.

---

## Definition of complete (this plan)
1. T0 baseline green; every hoisted/new numeric has unit + FD-vs-AD grad-check tests, 100% local pass.
2. Each sibling migration parity-verified with its suite green (progenax; fluxax 338-gate).
3. Provenance pass complete (no uncited constant/heuristic).
4. `PhotometricUnits` landed (fluxax Phase 2 unblocked).
5. Phase-B completion doc under `.claude-work/`; STATUS + brain updated.
