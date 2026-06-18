# jaxstro Phase C — Release Hardening — Design

**Date:** 2026-06-17
**Branch:** `feature/phase-c-release-hardening` (off `main` @ `90d10e5`)
**Method:** superpowers:subagent-driven-development — fresh subagent per task +
independent code-reviewer between tasks + final whole-arc review.
**Authority:** ratifies the Phase C scope already settled in
`docs/plans/2026-06-17-jaxstro-consolidation-and-release-strategy.md` (§5 hardening,
§8 Definition-of-Complete, §9 docs architecture / ADR-0005). This doc records the
*execution* decisions taken in the Phase C brainstorm; it does not re-open ratified ADRs.

Phase C is **release hardening, not the release**. Pushing, tagging, PyPI, GitHub Pages,
the release-staging decision (namespace vs `jaxstro-core` rename), and bumping sibling
pyproject floors are all **Phase D / out of scope**.

---

## Baseline (must not regress)

- `env -u VIRTUAL_ENV uv run --no-sync pytest -q` → **444 passed**
  (`--extra ml` variant for the params/integration tier).
- `ruff check src/` clean · `ruff format --check src/` clean · `mypy src/jaxstro` clean.
- Never weaken a test or tolerance to pass — fix the root cause.

---

## The five tasks (sequenced)

### T1 — 3-tier test reorg
Move the 17 flat `tests/*.py` into `tests/{unit,integration,validation}/` with
`git mv` (preserves blame). Add `tests/conftest.py` that **auto-applies** the
`unit`/`integration`/`validation` marker from each test's directory (no hand-marking).
Declare markers (`unit`, `integration`, `validation`, `slow`) in
`pyproject.toml [tool.pytest.ini_options]`. Write `tests/README.md` mirroring progenax:
Quick Start · tier table · what-each-tier-validates · references.

**Proposed classification** (ratified):
- **unit** — `units`, `constants`, `coords`, `numerics`, `checks`, `linear_algebra`,
  `quadrature`, `sampling`, `rng`, `astrometry`, `spatial`, `photometric`, `jaxconfig`,
  and existing `unit/test_params_{parameterization,transforms,transformed}.py`.
- **integration** — `integration_parity`, `grad_audit`, existing
  `integration/test_params_optax.py`.
- **validation** — `grad_checks` (FD-vs-AD numerical-truth checks).

**Gate:** 444 pass via **both** `-m <tier>` and `tests/<tier>/`; sum across tiers = 444
(no test lost or duplicated).

### T2 — CI (dormant) + local gate
**Constraint:** GitHub Actions minutes are exhausted, so CI must cost **zero** minutes
this phase. Author/commit `.github/workflows/tests.yml` with the full six-job design but
`on: workflow_dispatch` **only** — the `pull_request:` trigger present-but-commented, so it
never auto-runs. Flipping it on is a one-line Phase-D switch (when usage resets).

Six jobs: `lock-check` (`uv lock --check`) · `lint` (ruff check + ruff format --check +
mypy, py3.13) · `test-matrix` py311/312/313 (`-m "not slow"`, `--extra dev`) ·
`ml-integration` (py3.13, `--extra ml`, `tests/integration`) · `wheel-smoke` (build wheel
→ clean venv → import) · `tests` aggregator gate (branch-protection name).
`env: JAX_ENABLE_X64=1` + XLA low-mem + `OMP_NUM_THREADS=1`.

The **real Phase-C gate** is `scripts/check.sh`, which runs every job's exact commands
locally. This is what we verify against; "CI is correct" never depends on spending minutes.

### T3 — Docs (mystmd site)
Build the full §9 directory skeleton + `myst.yml` + dual-front-door `index.md`. Write
**real** content for the v1 core: `00-getting-started/` (install · jaxconfig/float64
first-run · first safe-math + root-find) · `10-theory/index.md` (the AD-safe-numerics
thesis, §9.1) + 2 exemplar method pages (`rootfinding`, `cumulative-trapz`) ·
`30-decisions/` (port the 11 existing ADRs) · `40-api/` reference landing. Remaining
sections (`20-architecture`, `50-howto`, `60-validation`, `90-development-log`,
`95-release`, `99-bibliography`) get honest **"planned" stub** index pages so the web
exists and builds clean. Prose via `research-workflow:docs-writing-voice`; clarity pass via
`elements-of-style`; syntax via `myst-expert`.

`mystmd` is a Node/pip build tool, **not** a package dependency — no thin-foundation
conflict; ADR-0005 already covers it.

**Local gate:** `myst build` → **0 content warnings**; every cross-ref/citation resolves.
(`myst build` as a CI job and a Pages deploy workflow are deferred to Phase D.)

### T4 — CHANGELOG.md
keep-a-changelog format, heading `0.1.0 (unreleased)`, covering **Phase B** (T0–T8: the
hatchling/Apache-2.0/py.typed baseline, PhotometricUnits, cumulative_trapz reconcile,
inverse_cdf_draw + Newton-PPF hoists, quadrature factory, grad-audit engine →
`jaxstro.testing`, numerics hardening, constants round-out, A_RAD correction) and
**jaxstro.params** (Parameterization + bijectors). Entries trace to real commits.

### T5 — CLAUDE.md hardening + README fix
Harden the package `CLAUDE.md` to progenax depth, adapted to infrastructure (not physics):
- **AD-safe-numerics patterns** (the `where`-trap, fixed-iteration not `while_loop`,
  saturation as a silent gradient killer).
- **Critical Invariants** — `cumulative_trapz` dx-outside · Gauss-Hermite probabilists'-
  via-physicists parity · `condition_number` → `+inf` sentinel · `bisect` structurally-zero
  grad caveat · `params` cached-derived-leaf caveat.
- **Definition-of-Complete**, **Common Issues**, **Debugging Checklist**,
  **Provenance discipline** sections.

Also fix the **stale README** badges/requirements (Python 3.10+ → 3.11+, JAX 0.4.28+ →
0.10.1, status badge) to match `pyproject.toml`.

---

## Per-task verification

- Baseline pytest (444) + `--extra ml` variant after every task; ruff/format/mypy clean on
  any task touching code or config.
- **T1:** tier selection works both ways; tier sum = 444.
- **T2:** run each job's exact commands locally via `scripts/check.sh`; YAML-validity check;
  careful read-review of workflow logic. (Cannot and will not run Actions.)
- **T3:** `myst build` → 0 content warnings.
- **T4/T5:** prose review; CHANGELOG entries map to commits; CLAUDE.md carries the
  load-bearing infra sections; README matches pyproject.

## Execution discipline (HITL constraints)

- Stage files **explicitly by name** (never `git add -A`).
- Commit trailer ends:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Fresh subagent per task + independent **code-reviewer** between tasks + final whole-arc
  review. Concise update to Anna after each task.
- **No push, no merge** without Anna's separate explicit words (merge and push are different
  words). Branch is kept until merged **and** pushed.
- Completion doc `.claude-work/PHASE_C_COMPLETE.md`; update `STATUS.md` (`next:`/`blocker:`);
  `brain "…"` capture at milestones (brain is pull-only).
- JAX-native only; no new core deps without an ADR + Anna's approval.

## Explicitly deferred to Phase D / out of scope

`myst build` CI job · GitHub Pages deploy workflow · standalone `validation/validate_*.py`
CLIs + plots · push / tag / PyPI · release-staging (namespace vs `jaxstro-core` rename) ·
sibling pyproject floor bumps (Anna does these herself).
