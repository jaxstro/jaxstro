# jaxstro — Consolidation, Hardening & Release Strategy

**Date**: 2026-06-17
**Status**: DRAFT — awaiting Anna's final ratification
**Author**: drafted with Claude (Opus 4.8), supervised by Anna Rosen
**Companion audit**: `.claude-work/state-of-jaxstro-2026-06-17.md`

This is the ratified *strategy* (the "what" and "why"). A granular implementation plan
(the "how", task-by-task) follows via `superpowers:writing-plans` after ratification.

---

## 1. Vision

jaxstro is the **foundation layer** of the differentiable-astrophysics ecosystem:
units, constants, coordinates, astrometry, and **generic, differentiable, dependency-free
numerics** that downstream packages (progenax, fluxax, gravax; planned startrax, stellax)
reuse instead of reinventing. It sits at the bottom of the dependency DAG and must stay the
**lightest** node precisely because everything depends on it.

**Guiding principle (RATIFIED): thin SoTA foundation, not a façade.**
Own generic dependency-free primitives; adopt the ecosystem foundation pair
(`jaxtyping + equinox`); refuse to absorb solver libraries (`diffrax`/`optimistix`/
`lineax`/`quadax`/`interpax`) — those are peer/upper layers each domain package pulls in
directly. This mirrors the Kidger ecosystem topology (`jaxtyping → equinox → {lineax,
optimistix, diffrax}`), which is SoTA *because* the foundation stays thin.

### Why thin beats a curated all-deps façade
- **Blast radius**: a units-only consumer must not inherit diffrax+optimistix+numpyro.
- **Version decoupling**: solver-library churn must not force a jaxstro release + re-lock of
  every downstream package.
- **API stability**: a foundation promises stability; wrapping fast-moving solver APIs imports
  their instability and doubles the maintenance surface.
- **No lag**: callers reach SoTA solvers directly, with jaxstro as no bottleneck.

---

## 2. Ratified decisions (this brainstorm)

| # | Decision | Choice |
|---|----------|--------|
| D1 | `units.DEFAULT` for the foundation | **CGS** (keep uncommitted diff; fix `AGENTS.md` to match) |
| D2 | Numerics dependency posture | **Thin foundation, not façade** |
| D3 | Adopt `equinox` as base dep | **Yes** (`jaxtyping + equinox`) |
| D4 | Inter-package topology | **Per-package editable path-sources** (independent release cadence); standardize tooling; add sibling smoke-test CI |
| D5 | Quadrature factory (GL + GH nodes + Hermite basis) | **Include in 0.1.0**; migrate progenax-experimental only; **defer fluxax GL-12** (hot PSF path) |
| D6 | uv project setup ("workspace like siblings") | **Standardize jaxstro as a standalone uv project** (hatchling + `uv.lock` + `py.typed` + `[tool.uv]`); no root workspace |
| D7 | License | **Apache-2.0** (ecosystem policy; replaces BSD-3-Clause) |
| D8 | Newton-PPF generic base | **Hoist** to jaxstro (keep Chabrier specialization local) |
| D9 | Docs framework | **Diátaxis spine + knowledge-web meta sections**, numerics-theory-first, pedagogical for research students (see §9) |
| D10 | Committed 0.1.0 public API surface | **Accepted as §6** (no experimental/private tier except `jaxstro.quantity`) |
| D11 | Constants round-out | **Add in 0.1.0** — a few more well-cited fundamental constants (σ_T, α, …), each with provenance |
| D12 | IFT (Information Field Theory) | **Reject from core** — adopt `NIFTy.re` at the inference/upper layer if/when needed; future-ADR. Heavy framework ⇒ posture violation (D2). |
| D13 | Unit-aware arrays | **Build our own** `jaxstro.quantity`: pure-`equinox` `Quantity` (value + dimension), explicit `.to()/.value` unwrap at JAX boundaries. **Zero new deps** (no unxt / quax / astropy). **Opt-in + experimental**; no sibling forced to migrate. Own design brainstorm pending (§10). |

### Why D13 = build-our-own (pure-equinox), not unxt/quax
- `unxt` is SoTA but (a) uses an **astropy backend** (heavy dep in the foundation everything
  depends on — violates D2) and (b) **subsumes jaxstro's `units` module** (overlapping unit
  systems). `quax` is the transparent-dispatch *mechanism* but adds `quax`+`plum-dispatch` and a
  permanent **primitive-coverage treadmill**.
- A pure-`equinox` `Quantity` is **philosophically identical to jaxstro's ethos** (explicit units,
  host-side resolution, unwrap-to-raw-array at the compute boundary), needs **no new deps**, and
  sidesteps the primitive long-tail. If transparent interop is ever needed, add a thin `quax`
  adapter *at the consuming layer*, never in the foundation.
- `Quantity` (dimensions-on-values) and `UnitSystem` (code-unit scaling) are **complementary**, not
  competing — defined boundary: `Quantity` at I/O edges, `UnitSystem` for the dimensionless interior.

### Future ADRs (seeded, post-0.1.0)
- Unit-aware arrays design (the `jaxstro.quantity` MVP — see §10).
- IFT → adopt `NIFTy.re` at the inference layer (not the foundation).
- `jax.scipy.special` curated namespace (revisit when startrax/stellax want it).
- Unit-*aware* arrays vs `unxt` interop revisited if transparent flow becomes a hard requirement.

---

## 3. Scope of jaxstro 0.1.0

### Tier 1 — consolidate + harden (land now)
1. **Reconcile `cumulative_trapz`** (21 progenax callsites). One function: `x` (non-uniform) **or**
   scalar `dx` (uniform); standardize the uniform path to **dx-outside** (cumsum-then-multiply,
   matches progenax's 18-site majority + marginally better numerics). Migrate progenax off its
   local `cumulative_trapezoid`; **parity-test all 21 sites** (allow documented ~1 ulp drift at
   the 3 former dx-inside sites, with their existing test budgets).
2. **Hoist `inverse_cdf_draw`** (16 callsites) — generic differentiable inverse-CDF sampler.
3. **Hoist Newton-PPF generic base** (R1) — keep Chabrier specialization in progenax.
4. **Quadrature factory** (D5) — new `jaxstro.numerics.quadrature`: `gauss_legendre_nodes(n)`,
   `gauss_hermite_nodes(n)` (probabilists'), `hermite_e_basis(g, n_max)`, `hermite_coefficients`.
   Host-side node generation (Golub–Welsch via numpy, frozen to jnp; static `n`) → constants in
   the jitted/grad path. Migrate progenax-experimental `gaussianization.py`.
5. **Grad-check utility** — dedup the *verbatim* `grad_audit/core.py` in fluxax+progenax into
   `jaxstro.testing` (`audit_entry_point`, `Case`, `AuditResult`, `EdgeConfig`); per-package
   registries stay local.
6. **`PhotometricUnits` + photometric constants** — `JY_CGS`, `AB_ZEROPOINT_JY`,
   `AB_ZEROPOINT_CGS` (Oke & Gunn 1983); a `PhotometricUnits` dataclass parallel to `UnitSystem`
   with presets, host-side resolution to constant multiplies. **Unblocks fluxax Phase 2.**
7. **Harden existing numerics to release-grade**:
   - FD-vs-autodiff grad-checks on every differentiable public function.
   - Fill test gaps: `linear_algebra`, `rng`, `compensated` (accuracy/cancellation), `simpson`.
   - Input validation: `interp1d` monotonic-x, `simpson` uniform-spacing.
   - Grad-hazard fixes: `project_onto(eps=0)` NaN; document `condition_number` non-diff.
   - Provenance pass: fix Julian-vs-tropical-year comment; per-constant citations; cite
     `fill_bins` reservoir + two-stencil heuristics (or mark as original method).
   - Upgrade numerics modules to full jaxtyping.
8. **Constants round-out** (D11) — add a few more well-cited fundamental constants (e.g. Thomson
   cross-section σ_T, fine-structure α), each with a provenance comment to its CODATA/IAU source.

### Tier 1b — experimental, opt-in (ships but nobody must adopt)
- **`jaxstro.quantity`** (D13) — pure-`equinox` `Quantity` (value + dimension), explicit unit
  conversion + dimensionless-guards + **trace-time errors on incompatible-unit `+`**, full
  grad-checks. Marked experimental; **no sibling migration**. Scoped MVP operator set (see §10);
  the long tail grows later. Lands in 0.1.0 **only if it does not delay the release** — else 0.2.0.

### Tier 3 — defer (YAGNI)
- fluxax 2D interp (**0 callsites** — dead code; hoist only when a consumer exists).
- `jax.scipy.special` re-export; benchmark harness; fluxax GL-12 migration (D5).

---

## 4. Multi-package maintenance strategy (durable)
1. **Layered DAG, no cycles.** jaxstro depends on nothing in-ecosystem. Only genuinely generic,
   reused, stable-API code moves up into it; domain logic stays in domain packages.
2. **Standardize tooling, not domain code**: hatchling, `py.typed`, shared ruff/mypy config, CI
   template, `uv lock --check`, CHANGELOG, **semver + deprecation policy** (downstream pins
   `jaxstro>=X,<Y`).
3. **Parity-preserving hoists**: move byte-identical numerics, parity-test vs origin, run sibling
   suites (progenax; fluxax 338-gate) before/after. A sibling-smoke-test CI job makes a
   jaxstro-side break fail in jaxstro's CI.
4. **Independent release cadence** via per-package path-sources (D4).

---

## 5. Release hardening (progenax-grade)
- **License** → Apache-2.0 (LICENSE + `license = "Apache-2.0"` + `license-files`).
- **Build** → hatchling; `py.typed`; `[tool.uv]`; `uv.lock`.
- **Metadata** → classifiers (Development Status :: 4 - Beta, Python 3.10–3.13, Topic ::
  Astronomy, Intended Audience :: Science/Research), keywords, project URLs.
- **Tests** → 3 tiers (`tests/unit`, `tests/integration`, `tests/validation`) + pytest markers.
- **CI** → `.github/workflows`: pytest + ruff + mypy, `JAX_ENABLE_X64=1`, `uv lock --check`,
  sibling smoke-test job.
- **Docs** → MyST site (R2).
- **CHANGELOG**, progenax-grade `CLAUDE.md` (Definition-of-Complete, Common Issues, Critical
  Formulas, Debugging Checklist).

---

## 6. Committed public API surface for 0.1.0 (R3 — proposal)
- **Stable/committed**: `jaxstro.units` (UnitSystem, named systems + aliases, `get_units`,
  `DEFAULT`, `DEFAULT_UNITS`, **PhotometricUnits**), `jaxstro.constants` (incl. photometric),
  `jaxstro.astrometry`, `jaxstro.coords`, `jaxstro.jaxconfig`.
- **Stable-but-evolving**: `jaxstro.numerics` (stats, interpolation, rootfinding, integration
  incl. reconciled `cumulative_trapz` + `quadrature` + `inverse_cdf_draw` + Newton-PPF, checks,
  compensated, linear_algebra, rng), `jaxstro.spatial` (morton, grid, neighbor).
- **Testing utility**: `jaxstro.testing` (grad-check) — public, semi-stable.
- **No experimental/private tier** at release.

---

## 7. Sequencing (maps to brief Phases A–D)
- **Phase A (this doc)** — audit + clean baseline + ratified strategy. Then: branch off `main`,
  triage dirty tree (commit D1 + `AGENTS.md` fix), set up uv/hatchling/py.typed baseline (D6).
- **Phase B** — consolidate generic numerics (Tier 1.1–1.6) + sibling migrations (parity-tested),
  each with unit + grad-check + (where physical) validation tests.
- **Phase C** — release hardening (§5): 3-tier tests, CI, metadata, docs, CHANGELOG, CLAUDE.md.
- **Phase D** — full gate green + sibling smoke-test (gravax/progenax/fluxax import + pass against
  new jaxstro); then Anna's merge-go → `main`, then her separate push word, then tag.

**Execution discipline**: `superpowers:writing-plans` → `superpowers:subagent-driven-development`
(fresh subagent per task + code-review subagent between tasks + final whole-arc review). Research
gates for numerics (gradient-validation, numerical-method-validation, provenance-of-constants,
verification-gate). One feature branch off `main`; per-task commits; no push/merge without Anna's
separate explicit words.

---

## 8. Definition of complete (per phase, progenax standard)
1. 3-tier pytest suite, 100% local pass, edge cases; every public numeric grad-checked (FD vs AD).
2. Validation script(s) with expected-vs-measured tables + provenance + pass/fail thresholds + plots.
3. `myst build` 0 content warnings (docs phase).
4. Per-phase completion doc under `.claude-work/`.
5. Sibling smoke-test green; full local gate green; STATUS + brain updated; THEN merge-go → push → tag.

---

## 9. Docs architecture (RATIFIED — D9)

The docs site is the project's **single source of truth and onboarding path** — a knowledge web
(theory ↔ API ↔ validation ↔ reference), serving a **new graduate student** and **future-you**
simultaneously. Framework = **Diátaxis** as the spine for the *user-doc* quadrants; the
decision-log / validation / dev-log / release / bibliography are the **project-meta web** around
them (not stretched into "Explanation"). Skeleton (Anna's numbered scheme, ratified):

```
docs/
  index.md             Landing — dual front door ("Learn the methods" ↔ "Look up the API") + routed paths
  00-getting-started/  [Tutorial]      install · jaxconfig/float64 first-run · "first safe-math + root-find" · glossary
  10-theory/           [Explanation]   HYBRID A+C — index.md = the "how to write AD-safe scientific
                       numerics" thesis (§9.1) that fans out into per-method pages:
                         stable-stats · compensated-sum · interpolation · rootfinding (scan not while_loop) ·
                         Newton–Cotes (cumulative-trapz dx-outside) · Gaussian quadrature (Golub–Welsch, GL/GH) ·
                         inverse-CDF & PPF · linear-algebra · PRNG · units & dimensional analysis · why CGS ·
                         photometric units (Oke & Gunn 1983) · coordinates & astrometry · spatial (Morton, neighbors)
  20-architecture/     [Explanation]   software-design "why": JAX-native functional/PyTree, units policy,
                       one-way dependency rule, thin-foundation posture — narrative prose that {cite}s the ADRs.
                       (Boundary: 10 = math of the methods; 20 = shape of the software.)
  30-decisions/        [meta]          numbered ADR log (passive-adr) — D1–D13 + reconstructed/backdated ADRs (§9.2)
  40-api/              [Reference]     per-module API: constants · units · quantity · numerics
                       (stats/roots/quad/interp/integration) · coordinates · spatial · jaxconfig · testing · photometric
  50-howto/            [How-to]        recipes: add a constant with provenance · write an AD-safe reduction ·
                       grad-check a function · choose a quadrature rule · consume jaxstro from a sibling package
  60-validation/       [meta]          Property | Tolerance | Measured | Anchor tables · AD/grad audit · convergence
  90-development-log/  [meta]          dated, newest-first
  95-release/          [meta]          changelog · versioning + deprecation policy
  99-bibliography/     [meta]          references.bib + per-paper notes (CODATA, IAU, Oke & Gunn, Neumaier,
                                       Morton, Golub–Welsch)
```

Two deliberate departures from progenax: a **first-class `30-decisions` ADR section** (adjacent to
`20-architecture` so narrative-why → atomic-why read as one rationale cluster — ADRs are *not*
tucked under architecture), and an **explicit dual-front-door landing**.

### 9.1 `10-theory/index.md` — the AD-safe-numerics thesis (principles-first spine)
A principles-first essay; each principle bridges to the per-method page(s) that exemplify it:
1. **Differentiability is a design constraint** — every primitive survives `jax.grad`; FD-vs-AD checked.
2. **Fixed iteration, not convergence loops** — `lax.scan`/fixed steps, never `while_loop` (bounded, diff'able).
3. **Guard singularities without killing gradients** — the `where`-trap (NaN backprops through the dead
   branch) → double-`where`/safe-denominator pattern.
4. **Saturation is a silent gradient killer** — `clip`/`min`/`max`/`floor` zero the boundary gradient;
   know intended vs. bug.
5. **Floating point is part of the math** — cancellation, log-domain, compensated summation, `log1p`/`expm1`.
6. **Non-diff ops are forbidden in the graph** — `argmax`/`argsort`/`sort`, integer casts, data-dependent
   shapes; isolate discrete (spatial) ops from differentiable paths.
7. **Quadrature & sampling differentiate through the integrand/values, not the nodes** — fixed rules.
8. **Precision discipline** — float64 opt-in (`jaxconfig`), matmul precision, when it bites.
9. **Correctness over comfort** — every constant cited; every method validated vs. analytic/known; "it ran"
   ≠ "it's correct."
10. **Vectorize & compose** — `vmap` over Python loops; pure functions, immutable PyTrees.

### 9.2 Reconstructed/backdated ADRs (discipline)
Decisions already baked into the code (float64-only, minimal-deps, CGS/CODATA constants, one-way
dependency rule, JAX-native functional style, …) are retrofitted as ADRs marked
`Status: Accepted (reconstructed <date> from code/git)`, dated from git/code **evidence** (never
invented), with any uncertain rationale flagged `inferred`. Document what the code demonstrably
decided — never a fabricated history.

Page conventions follow `research-workflow:docs-writing-voice` + its `page-anatomy` reference
(frontmatter minimum, orient-in-one-sentence, link-outward, anchor-numbers, honesty markers).
Voice prose drafted with `docs-writing-voice`; clarity pass with `elements-of-style`; MyST syntax
via `myst-expert`.

---

## 10. `jaxstro.quantity` design (PENDING — own brainstorm before implementation, D13)

Decided: **pure-`equinox`, zero-dep, explicit, opt-in/experimental.** Still to design in a focused
sub-brainstorm before any code:
- **Dimension representation** — exponent vector over base dimensions (M, L, T, Θ, …); equality &
  algebra (`+`/`-` require equal dims → trace-time error otherwise; `*`/`/` add/subtract exponents;
  `**` scales).
- **Unit registry** — lightweight, reusing `constants` + `UnitSystem` scales; `.to(unit)` as a
  constant multiply; no astropy.
- **MVP operator set** — `+ - * / **`, comparisons, `matmul`, reshape/broadcast/indexing,
  dimensionless-guards on transcendentals (`sin`, `exp`, `log` require dimensionless).
- **Grad semantics** — `Quantity` is an `equinox.Module` PyTree; grads flow through `.value`;
  full FD-vs-AD grad-checks.
- **Boundary with `UnitSystem`** — `Quantity` at I/O edges; unwrap to raw array + `UnitSystem` for
  the dimensionless compute interior.
- **Timing** — 0.1.0 experimental MVP only if it does not delay the release; else 0.2.0 headline.
