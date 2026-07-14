# jaxstro — status

next: Review the semantic Validation and Project ownership migration. Do not begin Task 10 until Task 9 is accepted.

previous: Researcher-first documentation Task 9 completed on local main (2026-07-14). Public validation evidence now lives under semantic numerical, data, and reusable-method groups; project direction, development, decisions, release, and bibliography now share one Project owner. All 14 ADRs retain stable public routes while remaining hidden from primary navigation, and the public Instructor section was retired. Routed research-software prose is ASCII-clean and free of course-facing framing except for four exact, file-scoped `Diátaxis` proper-name occurrences enforced by the language ratchet. The SOTA assessment and scorecard now audit research workflow, contracts, evidence, limitations, and prioritized capability ownership. The full integration suite passed 254 tests, and the strict docs gate built and audited 163 unique routes.

previous: Researcher-first documentation Task 8 completed on local main (2026-07-14). The omnibus API was replaced by a route-first `/api` landing and 38 current owner pages, including `jaxstro.numerics.types` and the post-plan `jaxstro.numerics.kepler` owner. Generated contract and source-provenance pages now live under `docs/50-api/research-infrastructure`; the source-provenance pages preserve the established `/constants` and `/atmospheres` meanings while the owner references use `/constants-api` and `/atmospheres-api`. Contract, evidence, and source-card artifacts regenerate deterministically from current runtime ownership. All routed API links, TOC entries, and manifest routes were migrated, no Task 8 API page retains an opaque generated index route, and runtime exports were unchanged. The review-fix focused suite passed 23 tests, the full integration suite passed 249 tests, and the strict shared docs gate rendered and audited 164 unique routes.

previous: Researcher-first documentation Tasks 3 through 7 completed on local main (2026-07-13). Numerical methods now separate current Jaxstro owners from delegated ecosystem methods and planned QMC/signal capabilities; scientific representations separate current units, geometry, spectra, and parameter/state owners from planned uncertainty and deferred field abstractions; research workflows organize scientific ML, data pipelines, differentiable research, reproducible research, and executable investigations around representation -> computation plan -> execution -> audit -> evidence -> claim. Task 7 hard-cut the teaching-oriented executable registry to a deterministic schema-2 research-workflow registry with no aliases. The combined affected suite passed 60 tests, the strict MyST build rendered 127 pages, the exact route/link/accessibility audit passed, and independent review found no remaining findings.

previous: Researcher-first documentation Tasks 1 and 2 completed on local main (2026-07-13). Start Here now includes beginner JAX orientation, first-principles transformed-program examples, a neutral NumPy/JAX/Jaxstro decision table, documentation-use routes, and an executable luminosity map. Foundations now uses semantic Mathematical objects and Models and computation groups with stable public routes, tailored use-case openings, research-only language, ASCII prose plus LaTeX math, and explicit non-deficit readiness framing. Both slices passed focused and affected tests, isolated strict MyST/rendered-route gates, and independent review with no remaining findings.

previous: Added a living future modules and capabilities roadmap to the public Development Log (2026-07-12). It inventories implemented numerical and infrastructure families, preserves the Jaxstro-versus-Informax ownership boundary, and converts the proposed scientific-ML, iterative-solver, nonlinear-system, quasi-Monte-Carlo, adaptive-quadrature, signal, uncertainty-propagation, and field-method expansions into definition-of-done checklists. Focused documentation contracts and the strict rendered-site gate passed.

previous: GitHub Actions Node-runtime maintenance completed (2026-07-12). Active workflows now use the Node 24 releases of checkout, setup-node, upload-pages-artifact, and deploy-pages; setup-uv is pinned to the immutable v8.1.0 release SHA. Focused workflow contracts passed (7 tests), all 3 workflow files parsed, and Pages run 29224678647 built in 44 s and deployed in 8 s with zero build or deploy annotations. The live homepage returned HTTP 200.

previous: Release closeout published to `origin/main` and GitHub Pages (2026-07-12). The full local release gate passed with 1,352 non-slow tests, 162 ML-integration tests, a clean Python 3.13 wheel import, fresh provenance/contract/evidence/curriculum registries, and 88 strict documentation routes. Pages run 29224056600 built and deployed successfully after the docs jobs were repaired to provision Python 3.13 plus locked dependencies through `setup-uv`; focused workflow contracts now prevent recurrence. Live HTTP and rendered-content checks passed for the homepage, foundations, investigations, rootfinding, and package assessment scorecard.

previous: The executable foundations curriculum completed on local main (2026-07-12). Optional ungraded readiness routing now leads through units and scales, models and dimensionality, linear algebra, derivatives, probability, inference, conditioning, and differentiable programs without replacing any module section. Repository-owned root, finite-power-law, and interpolation investigations execute public APIs and resolve to scientific contracts, validation targets, indexed evidence where available, explicit limitations, instructor guidance, and a claim-calibrated rubric. Generated curriculum coverage and freshness checks are wired into local and documentation gates.

previous: Unified evidence infrastructure completed on local main (2026-07-12). Computational evidence, source provenance, and scientific policy remain deliberately distinct classes in one index. Rootfinding, implicit derivatives, and spectra now use one strict frozen envelope with units, environment policy, informational metrics, explicit passing comparisons, deterministic emit/check, and human reports. Contract links name exact passing artifact gates; provenance digests cover all rendered card content. Method-specific thresholds remain method-owned. The focused tests, Ruff, MyPy, and rendered docs gate passed. Independent C3 reviews found no Critical issues, and every Important finding was resolved.

previous: Safeguarded scalar rootfinding and the separate finite power-law alpha=-1 correctness slice
completed on local main (2026-07-12). Jaxstro now owns typed sign-bracket evidence, deterministic
guarded IQI/secant/midpoint proposals, fixed-shape trace/result types, and a value-first fixed-scan
wrapper. Scalar execution stops residual evaluations after convergence or nonfinite exhaustion; VMAP
preserves values/shapes but not a physical per-lane cost mask, so expensive batched callers should
use lax.map. The finite power-law normalization/logpdf/CDF/PPF now share smooth expm1(x)/x and
log1p(x)/x removable-singularity kernels through alpha=-1, with analytic and central-FD derivative
gates. A separate `implicit_bracketed_root(f, args, ...)` now provides a fail-closed
`lax.custom_root` IFT derivative behind uniqueness, smooth-branch, convergence,
finiteness, residual, bracket-width, and slope-conditioning certificates. No
while-loop or domain-specific runtime logic was added.

public-hardening: Slices A, D0, B, and C are complete. Local Slice D now has a least-privilege
Pages workflow, a `/jaxstro` base-path gate that validates 61 rendered routes, public citation and
contributor metadata, an authorization-gated release checklist, and sdist exclusions for tests,
laboratory/agent state, internal audits, and plans. The final base-path DOM gate passed in 10.97 s.
The full local release gate passed in 225.68 s: 1,138 core tests passed with 23 expected optional-
dependency skips, 114 ML-integration tests passed, and the clean-wheel import smoke passed.

publication: GitHub Pages is configured for Actions and live at https://jaxstro.github.io/jaxstro/.
Corrected run 29188207759 built/gated/uploaded the 61-page static artifact in 27 s and deployed it
in 8 s. Live verification returned 200 for the homepage, release checklist, and compiled base-path
CSS. The first run's 404 exposed and permanently ratcheted the `_build/site` versus `_build/html`
artifact boundary.

blocker:
due:

## Phase C — DONE (2026-06-17, branch feature/phase-c-release-hardening, HEAD 40f936e)
Release hardening: 3-tier tests + CI + mystmd docs + CHANGELOG + hardened CLAUDE.md. All 5 tasks +
whole-arc independently reviewed (APPROVE). Gate green end-to-end: `bash scripts/check.sh` →
ALL LOCAL GATES PASSED (lock-check, ruff, mypy, pytest 447, ml-integration 24, wheel-smoke);
`myst build` 25 pages 0 warnings; partition unit393/integration24/validation30=447. Zero Actions
minutes spent. See `.claude-work/PHASE_C_COMPLETE.md`.
- `344e510` T1 3-tier tests (git mv + conftest path-auto-marking + markers/strict + guard + README)
- `5eadbd1` T2.0 resolve pre-existing ruff/format debt (root-cause; no config relaxed)
- `7439d1a` T2 dormant 6-job CI (workflow_dispatch only) + scripts/check.sh local gate
- `a69c588`+`ee50fbc` T3 mystmd Diátaxis site (v1 core + 12 ported ADRs + honest stubs)
- `1ab1e01` T4 CHANGELOG (keep-a-changelog: Phase B + jaxstro.params)
- `40f936e` T5 progenax-depth CLAUDE.md + README staleness fix (3.11+/jax0.10.1/equinox core)

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
  VALUE FIX: A_RAD 7.565767e-15 → 7.565733250e-15 (stored-precision rounding of 4σ/c;
  old test was vacuous) — reviewer verified.
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
