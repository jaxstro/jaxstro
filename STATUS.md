# jaxstro — status

next: jaxstro.params + Phase B PUSHED to origin/main (2026-06-17, c409e9a — in sync). NEXT: Phase C
(release hardening) in a FRESH session — 3-tier test reorg, CI (GitHub Actions), MyST/Diátaxis docs,
CHANGELOG, progenax-grade CLAUDE.md. Brainstorm→writing-plans→subagent-driven, same as params.
blocker: none (TAG/release still held — Phase D deferred pending release-staging decision: jaxstro
namespace? rename jaxstro→jaxstro-core?).
due:

## jaxstro.params — DONE (merged to local main 2026-06-17, b6d65b3, 14 commits, 444 tests green)
Equinox-only selective-inference utility (ADR-0009): `Parameterization` (leaf-aligned free/fixed
marking + PyTree↔vector bridge + log_det_jacobian) + Identity/Exp/Softplus/Sigmoid bijectors
(analytic, extreme-u-stable log-dets). No new core dep; optax/numpyro under `[ml]` only.
Validated: real progenax.PlummerProfile recovery via optax (exact) + numpyro (toy). Surfaced+documented
the cached-derived-leaf caveat (from_vector replaces leaves, does not re-run __init__ → fit the leaf the
observable reads). Also bumped core dep floors to current stable (jax 0.10.1/py311). Per-task + whole-arc
reviews all cleared. Branch feature/jaxstro-params KEPT (merged, not pushed → not deleted per Anna's rule).
distrax build-vs-reuse: ours for v1; revisit at Informax (ADR-0009 addendum + brain note).
Cross-repo TODO in brain inbox: bump siblings' pyproject floors to current stable + requires-python>=3.11.

## Phase B working decisions (Anna-ratified 2026-06-17)
- Sibling test cadence: run TARGETED progenax subset per migration; ONE full progenax suite gate at Phase-B end before merge-go (full suite ≈ 60 min).
- Canonical name: `cumulative_trapz` everywhere (no `cumulative_trapezoid` alias).
- progenax coverage.json staleness gate: defer regen to a single Phase-B-end maintenance commit.

## Current focus
Executing Phase B plan (docs/plans/2026-06-17-jaxstro-phase-b-implementation.md) via
superpowers:subagent-driven-development (fresh subagent per task + code-review between tasks).

Branch `feature/consolidate-harden-release`:
- `5a2791e` T0 — hatchling + Apache-2.0 + py.typed + classifiers + working uv.lock (234 tests green)
- `694ed43` brain-spoke handshake wired into CLAUDE.md/AGENTS.md (CLAUDE.md un-ignored)
- `dcdc7cf` README relicense BSD→Apache-2.0 (code-review follow-up)
- `5ae194b` T1 — PhotometricUnits + Jy/AB constants (Oke & Gunn 1983); 246 tests green
  → **fluxax Phase 2 dependency LANDED** (import: `from jaxstro.units import PhotometricUnits, SOLAR_PHOTOMETRIC, CGS_PHOTOMETRIC`)
- `4a84b0b` T1 hardening — poison AB linear flux scale w/ NaN (fail loud); 248 tests green
- `c9686ea` T2 — cumulative_trapz uniform path → dx-outside (progenax parity); 261 tests green
  progenax side (branch `feature/adopt-jaxstro-numerics`): `e2afe2a` migrate 21 callsites,
  `bb351ad` rename → cumulative_trapz / drop alias (122 targeted tests green)
- `030e189` T3 — hoist differentiable inverse_cdf_draw → numerics/sampling.py; 269 tests green
  progenax: `bf359df` use jaxstro inverse_cdf_draw (7 invocations/5 files — plan's "16/6" was wrong;
  no shim; 154 targeted tests green)

## Downstream dependency
**fluxax Phase 2 unblocked** once T1 merges to main — `PhotometricUnits` + `JY_CGS`/`AB_ZEROPOINT_JY`/
`AB_ZEROPOINT_CGS` now exist on this branch.

## Hub-side TODO (needs Anna's go — ~/brain edit)
Register jaxstro in `~/brain/roots.yml` so federate.py pulls this STATUS.md into the dashboard/standup
(siblings are registered; jaxstro is not yet).

## Open
- [x] Phase A: audit + strategy ratification (D1–D13)
- [x] T0 baseline tooling
- [x] T1 PhotometricUnits (fluxax unblocker)
- [x] T2 reconcile cumulative_trapz (dx-outside) + progenax migration + rename
- [x] T3 hoist inverse_cdf_draw + progenax migration
- [x] T4 hoist generic Newton-PPF base + progenax IMF migration ✅ reviewed
  jaxstro `d698fff` (274 tests) · progenax `f5e15f9` (309 targeted; Chabrier specialization kept local)
- [x] T5 quadrature factory (GL/GH + Hermite) + progenax-experimental migration ✅ reviewed
  jaxstro `b3b7081` (287 tests) · progenax `6906c20` (28 experimental targeted)
  note: gauss_hermite built from physicists' hermgauss+√2 for byte-parity; verified = probabilists' rule
- [x] T6 grad-audit engine → jaxstro.testing (dedup fluxax+progenax) ✅ reviewed (all 3 repos)
  jaxstro `b1361fc` (296 tests) · fluxax `d9e8298` on feature/use-jaxstro-testing (42 grad-audit + 338 full)
  · progenax `58a8938` (125 grad-audit). Shared engine sets Direction=str; per-pkg registries stay local.
- [x] T7a numerics coverage + project_onto NaN fix ✅ reviewed (APPROVE)
  `483db6` + `7996c6a` — 350 tests (+54). Bonus: norm2/project_onto static_argnames fix; new
  checks.try_concrete_bool; simpson/interp1d wrapper+core eager validation.
- [x] T7b grad-check sweep + provenance + full jaxtyping ✅ reviewed (APPROVE)
  `69de872`+`22192c4`+`a117d86` — 380 tests (+30); ruff+mypy clean on numerics.
  VALUE FIX: A_RAD 7.565767e-15 → 7.565733250e-15 (= 4σ/c exactly; old test was vacuous) — reviewer verified.
  bisect: structurally zero grad wrt params (documented). x64 fail-loud guard added.
- [x] T7c condition_number 0.0 → +inf (Anna-approved) + E731 lint fix → `8d61801`; ruff check src/ clean; 381 tests
- [x] T8 constants round-out (CODATA 2018) ✅ reviewed — `28f4193`; +5 constants
  (ALPHA_FS, E_ESU, R_E, SIGMA_T, R_GAS); 389 tests; internal-consistency cross-checks pass
- [ ] FINAL whole-arc review (T1–T8 integration) (running) → then Phase-B completion doc
- [ ] T7b grad-checks across primitives + provenance (Julian-year) + jaxtyping + deferred follow-ups
- [ ] T8 constants round-out (CODATA 2018)

## Merge ordering (Phase D)
jaxstro lands FIRST (siblings import jaxstro.numerics + jaxstro.testing). Sibling branches:
progenax `feature/adopt-jaxstro-numerics`, fluxax `feature/use-jaxstro-testing`.

### Deferred follow-ups (batch into T7 / Phase-B end)
- delete pre-existing unused import in progenax profiles/api.py
- add skip-if-not-x64 guard to grad-check tests (fail loud if precision regresses) — T7 standardizes this
- widen newton_ppf type hints on u/x0/lo/hi to include float (cosmetic) — T7 jaxtyping pass
- correct plan/D-log: T3 was 7 invocations/5 files (not 16/6)
- progenax coverage.json regen (single commit, Phase-B end)
- progenax full-suite gate (Phase-B end, ~60 min)
- [ ] T4 hoist generic Newton-PPF base + progenax migration
- [ ] T5 quadrature factory + progenax-experimental migration
- [ ] T6 grad-check util → jaxstro.testing (fluxax/progenax dedup)
- [ ] T7 harden existing numerics (grad-checks, tests, provenance, jaxtyping)
- [ ] T8 constants round-out (CODATA 2018)
- [ ] Phase C: release hardening (3-tier tests, CI, MyST docs, CHANGELOG)
- [ ] Phase D: release prep (sibling smoke-tests, merge-go, push, tag)
