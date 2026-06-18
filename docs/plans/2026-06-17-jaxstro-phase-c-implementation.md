# jaxstro Phase C — Release Hardening — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or
> superpowers:subagent-driven-development) to implement this plan task-by-task.

**Goal:** Harden jaxstro to a release-grade package — 3-tier test architecture, a
zero-cost dormant CI + local gate, a mystmd docs site (scaffold + v1 core), a CHANGELOG,
and a progenax-depth CLAUDE.md — without regressing the 444-test green baseline.

**Architecture:** Five independent-ish tasks. T1 (test reorg) must land before T2 (CI
references the tier dirs). T3/T4/T5 are independent. Each task: small steps, explicit file
paths, a verification command with expected output, then a commit. CI is
`workflow_dispatch`-only (Actions minutes exhausted) and verified by running its exact
commands locally via `scripts/check.sh`.

**Tech Stack:** Python 3.11+, JAX (x64), equinox, jaxtyping; uv + hatchling; pytest +
markers; ruff + mypy; GitHub Actions (dormant); mystmd (Node/pip CLI, not a package dep).

**Design doc:** `docs/plans/2026-06-17-jaxstro-phase-c-design.md`.

---

## Conventions for every task

- **Baseline command** (the gate, run from repo root
  `/Users/anna/projects/jaxstro-dev/jaxstro`):
  ```bash
  env -u VIRTUAL_ENV uv run --no-sync pytest -q              # expect: 444 passed
  env -u VIRTUAL_ENV uv run --no-sync --extra ml pytest -q   # params/ml tier too
  ```
- **Lint/type gates** (any task touching `src/` or config):
  ```bash
  env -u VIRTUAL_ENV uv run --no-sync ruff check src/        # All checks passed
  env -u VIRTUAL_ENV uv run --no-sync ruff format --check src/
  env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro       # Success: no issues
  ```
- **Never weaken a test or tolerance to pass** — fix the root cause.
- **Stage files explicitly by name** — never `git add -A` / `git add .`.
- **Commit trailer** ends every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  ```
- **No push, no merge** — Anna's separate explicit words only.

---

## Task 1 — 3-tier test reorg

**Files:**
- Move (git mv): the 16 flat `tests/test_*.py` → `tests/{unit,integration,validation}/`
- Modify: `tests/conftest.py` (add path→marker auto-marking)
- Modify: `pyproject.toml` (`[tool.pytest.ini_options]` markers + `--strict-markers`)
- Create: `tests/validation/test_suite_structure.py` (regression guard)
- Create: `tests/README.md`
- Create (if missing): `tests/unit/__init__.py`, `tests/integration/__init__.py`,
  `tests/validation/__init__.py`

**Classification (ratified — sums to 444):**

| Tier | Files | Tests |
|------|-------|-------|
| unit | astrometry(12), checks(39), constants(40), coords(28), jaxconfig(4), linear_algebra(19), numerics(62), photometric(14), quadrature(13), rng(19), sampling(8), spatial(35), units(47) + existing unit/params_parameterization(11), params_transformed(9), params_transforms(33) | **393** |
| integration | grad_audit(10), integration_parity(13) + existing integration/params_optax(1) | **24** |
| validation | grad_checks(27) | **27** |

**Step 1 — Write the structural guard test (RED).**
Create `tests/validation/test_suite_structure.py`:
```python
"""Regression guard: all tests live in a tier dir and carry a tier marker."""
import pathlib

import pytest

TIERS = ("unit", "integration", "validation")
TESTS_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_no_flat_test_modules():
    """No test_*.py may sit directly under tests/ — every test belongs to a tier."""
    flat = sorted(p.name for p in TESTS_ROOT.glob("test_*.py"))
    assert flat == [], f"flat test modules must move into a tier dir: {flat}"


def test_tier_dirs_populated():
    for tier in TIERS:
        d = TESTS_ROOT / tier
        assert d.is_dir(), f"missing tier dir: {d}"
        assert list(d.glob("test_*.py")), f"tier {tier} has no test modules"


@pytest.mark.validation
def test_this_module_is_marked_validation(request):
    """Sanity: the path→marker conftest hook actually applied a tier marker."""
    own = {m.name for m in request.node.iter_markers()}
    assert "validation" in own
```

**Step 2 — Run it to confirm it fails.**
```bash
env -u VIRTUAL_ENV uv run --no-sync pytest tests/validation/test_suite_structure.py -q
```
Expected: FAIL (`test_no_flat_test_modules` — flat modules still present; dir may not exist
yet). This proves the guard has teeth.

**Step 3 — Add markers + strict-markers to `pyproject.toml`.**
Replace the `[tool.pytest.ini_options]` block with:
```toml
[tool.pytest.ini_options]
minversion = "8.0"
addopts = "-ra --strict-markers"
testpaths = ["tests"]
markers = [
    "unit: fast isolated unit tests (functional correctness, shapes, bounds, round-trips)",
    "integration: cross-module / JAX-transform tests (jit, grad, vmap, parity)",
    "validation: scientific validation (FD-vs-AD numerical truth, convergence)",
    "slow: expensive tests excluded from the fast gate",
]
```

**Step 4 — Add path→marker auto-marking to `tests/conftest.py`.**
Append to the existing file (keep the float64 `pytest_configure`):
```python
import pathlib

import pytest

_TIERS = ("unit", "integration", "validation")


def pytest_collection_modifyitems(config, items):
    """Auto-apply the tier marker (unit/integration/validation) from each test's path."""
    for item in items:
        parts = pathlib.Path(str(item.fspath)).parts
        for tier in _TIERS:
            if tier in parts:
                item.add_marker(getattr(pytest.mark, tier))
                break
```

**Step 5 — Create tier `__init__.py` files (if absent) and move files with `git mv`.**
```bash
cd /Users/anna/projects/jaxstro-dev/jaxstro
touch tests/unit/__init__.py tests/integration/__init__.py tests/validation/__init__.py
git add tests/unit/__init__.py tests/integration/__init__.py tests/validation/__init__.py

# unit tier
git mv tests/test_astrometry.py tests/test_checks.py tests/test_constants.py \
       tests/test_coords.py tests/test_jaxconfig.py tests/test_linear_algebra.py \
       tests/test_numerics.py tests/test_photometric.py tests/test_quadrature.py \
       tests/test_rng.py tests/test_sampling.py tests/test_spatial.py \
       tests/test_units.py tests/unit/

# integration tier
git mv tests/test_grad_audit.py tests/test_integration_parity.py tests/integration/

# validation tier
git mv tests/test_grad_checks.py tests/validation/
```
**Note:** if any moved file imports a sibling via `from test_x import ...` or a relative
path, fix the import after the move (grep first:
`grep -rn "import test_" tests/` and `grep -rn "from \." tests/`). Expected: none, but
verify.

**Step 6 — Run the structural guard again (GREEN).**
```bash
env -u VIRTUAL_ENV uv run --no-sync pytest tests/validation/test_suite_structure.py -q
```
Expected: PASS.

**Step 7 — Verify the full suite + tier partition.**
```bash
env -u VIRTUAL_ENV uv run --no-sync --extra ml pytest -q                       # 444 +1 guard = 447 passed
env -u VIRTUAL_ENV uv run --no-sync pytest -m unit -q --co | grep -c "::"      # 393
env -u VIRTUAL_ENV uv run --no-sync pytest -m integration -q --co | grep -c "::"  # 24
env -u VIRTUAL_ENV uv run --no-sync pytest -m validation -q --co | grep -c "::"   # 27 + 3 guard = 30
env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit -q --co | grep -c "::"   # 393
```
Expected: full suite green (now **447**: the 444 baseline + 3 new guard tests); `-m unit`
== `tests/unit` count; the three `-m` tiers + guard sum to the collected total. The 444
*original* tests are all still present and green.

**Step 8 — Write `tests/README.md`** (mirror progenax: Quick Start · tier table · what
each tier validates · references). Keep counts described as "see CI / `pytest --co` for the
live number" to avoid drift. Include the `-m <tier>` and `tests/<tier>/` selection idioms
and the `env -u VIRTUAL_ENV uv run --no-sync` invocation.

**Step 9 — Lint/type/baseline gates.**
```bash
env -u VIRTUAL_ENV uv run --no-sync ruff check src/ tests/
env -u VIRTUAL_ENV uv run --no-sync ruff format --check src/ tests/
```
(mypy targets `src/jaxstro`, unaffected — run it to be safe.)

**Step 10 — Commit.**
```bash
git add tests/conftest.py tests/README.md pyproject.toml \
        tests/unit tests/integration tests/validation
git commit  # message below
```
```
test(phase-c): 3-tier test architecture (unit/integration/validation)

git mv the 16 flat test modules into tier dirs; conftest auto-applies the
tier marker from each test's path; declare markers + --strict-markers in
pyproject; add a structural guard (no flat modules, tiers populated, marker
applied) and tests/README. Partition: unit 393 / integration 24 / validation 27.
444 baseline preserved (+3 guard tests).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

---

## Task 2 — CI (dormant) + local gate script

**Files:**
- Create: `scripts/check.sh` (the real Phase-C gate)
- Create: `.github/workflows/tests.yml` (`workflow_dispatch`-only)

**Step 1 — Write `scripts/check.sh`.** Runs every CI job's exact commands locally.
```bash
#!/usr/bin/env bash
# Local mirror of the dormant GitHub Actions gate (Actions minutes are exhausted).
# Run from repo root. Any failure aborts (set -e).
set -euo pipefail
RUN="env -u VIRTUAL_ENV uv run --no-sync"

echo "== lock-check =="
env -u VIRTUAL_ENV uv lock --check

echo "== lint: ruff check =="
$RUN ruff check src/ tests/
echo "== lint: ruff format --check =="
$RUN ruff format --check src/ tests/
echo "== lint: mypy =="
$RUN mypy src/jaxstro

echo "== test-matrix (current interpreter; CI does 3.11/3.12/3.13) =="
$RUN pytest -m "not slow" -q

echo "== ml-integration =="
env -u VIRTUAL_ENV uv run --no-sync --extra ml pytest tests/integration -q

echo "== wheel-smoke =="
env -u VIRTUAL_ENV uv build --wheel -o dist/
rm -rf /tmp/jaxstro-clean
env -u VIRTUAL_ENV uv venv /tmp/jaxstro-clean
env -u VIRTUAL_ENV uv pip install --python /tmp/jaxstro-clean/bin/python dist/*.whl
/tmp/jaxstro-clean/bin/python -c "import jaxstro; print(jaxstro.__name__, 'imports clean')"

echo "ALL LOCAL GATES PASSED"
```
Then `chmod +x scripts/check.sh`.

**Step 2 — Run it (the gate).**
```bash
bash scripts/check.sh
```
Expected: ends with `ALL LOCAL GATES PASSED`. Fix any real failure at root cause (do not
weaken). Note: `uv lock --check` must pass — if it reports drift, run `uv lock` and review
the diff before committing (lockfile drift is itself a finding).

**Step 3 — Write `.github/workflows/tests.yml`** — six jobs, dormant trigger. The
`pull_request:` trigger is present-but-commented so it is a one-line Phase-D switch.
```yaml
name: tests

# Actions minutes are exhausted -> manual-only so this NEVER auto-runs (0 minutes).
# Phase D: uncomment `pull_request:` to enable PR gating once usage resets.
on:
  workflow_dispatch:
  # pull_request:

concurrency:
  group: tests-${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

env:
  JAX_ENABLE_X64: "1"
  XLA_PYTHON_CLIENT_PREALLOCATE: "false"
  XLA_PYTHON_CLIENT_ALLOCATOR: platform
  OMP_NUM_THREADS: "1"

jobs:
  lock-check:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          python-version: "3.13"
      - run: uv lock --check

  lint:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          python-version: "3.13"
      - run: uv sync --locked --extra dev
      - run: uv run --no-sync ruff check src/ tests/
      - run: uv run --no-sync ruff format --check src/ tests/
      - run: uv run --no-sync mypy src/jaxstro

  test-matrix:
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12", "3.13"]
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          python-version: ${{ matrix.python-version }}
      - run: uv sync --extra dev --python ${{ matrix.python-version }}
      - run: uv run --no-sync pytest -m "not slow" -q

  ml-integration:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          python-version: "3.13"
      - run: uv sync --locked --extra dev --extra ml
      - run: uv run --no-sync pytest tests/integration -q

  wheel-smoke:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          python-version: "3.13"
      - run: uv build --wheel -o dist/
      - name: Install wheel in a clean venv and import
        run: |
          uv venv /tmp/clean
          uv pip install --python /tmp/clean/bin/python dist/*.whl
          /tmp/clean/bin/python -c "import jaxstro; print(jaxstro.__name__, 'imports clean')"

  tests:
    if: always()
    needs: [lock-check, lint, test-matrix, ml-integration, wheel-smoke]
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - name: All required jobs succeeded
        run: |
          echo "lock-check=${{ needs.lock-check.result }} lint=${{ needs.lint.result }} test-matrix=${{ needs.test-matrix.result }} ml-integration=${{ needs.ml-integration.result }} wheel-smoke=${{ needs.wheel-smoke.result }}"
          if [ "${{ needs.lock-check.result }}" = "success" ] \
             && [ "${{ needs.lint.result }}" = "success" ] \
             && [ "${{ needs.test-matrix.result }}" = "success" ] \
             && [ "${{ needs.ml-integration.result }}" = "success" ] \
             && [ "${{ needs.wheel-smoke.result }}" = "success" ]; then
            echo "all required jobs green"
          else
            echo "::error::a required job did not succeed"
            exit 1
          fi
```

**Step 4 — Validate the YAML parses.**
```bash
env -u VIRTUAL_ENV uv run --no-sync python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/tests.yml')); print('yaml ok')" \
  || echo "pyyaml absent — fall back to careful visual review of the workflow"
```
Expected: `yaml ok` (or the documented fallback). Confirm `on:` has `workflow_dispatch`
and `pull_request:` is commented.

**Step 5 — Commit.**
```bash
git add scripts/check.sh .github/workflows/tests.yml
git commit  # message below
```
```
ci(phase-c): dormant 6-job workflow + scripts/check.sh local gate

GitHub Actions minutes exhausted -> tests.yml is workflow_dispatch-only (the
pull_request trigger present-but-commented, a one-line Phase-D switch) so it
costs zero minutes. scripts/check.sh runs every job's exact commands locally
(lock-check, ruff check + format --check, mypy, pytest matrix slice,
ml-integration, wheel-smoke) and is the real Phase-C gate.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

---

## Task 3 — Docs (mystmd site: scaffold + v1 core)

**Files (create):**
- `docs/myst.yml`
- `docs/index.md` (dual front door)
- `docs/00-getting-started/index.md`
- `docs/10-theory/index.md` (AD-safe-numerics thesis), `.../rootfinding.md`,
  `.../cumulative-trapz.md`
- `docs/30-decisions/index.md` + 11 ported ADRs (`0001-...md` … `0011-...md`)
- `docs/40-api/index.md`
- Stub `index.md` in each of `20-architecture/`, `50-howto/`, `60-validation/`,
  `90-development-log/`, `95-release/`, `99-bibliography/`

> **Note:** `docs/plans/` already exists and holds plan docs — leave it untouched; the
> mystmd `project.exclude` in `myst.yml` must exclude `plans/**` so the site doesn't try to
> render them.

**Prerequisite — mystmd CLI.**
```bash
myst --version || echo "install: npm i -g mystmd   (or: uv tool install mystmd)"
```
If absent, install before the build step. mystmd is a build tool, **not** a package dep.

**Step 1 — Skill the prose.** Invoke `research-workflow:docs-writing-voice` for the voice +
page anatomy; `myst-expert` for syntax; `elements-of-style` for the clarity pass. These are
mandatory per the design doc.

**Step 2 — Write `docs/myst.yml`** with project metadata, the toc, and `exclude: [plans/**]`.
Minimal toc covering the skeleton; reference `myst-expert` for current `myst.yml` schema.

**Step 3 — Write the v1 core pages** (real content):
- `index.md` — dual front door ("Learn the methods" ↔ "Look up the API") + routed paths.
- `00-getting-started/index.md` — install (uv), `enable_high_precision()` first-run, a
  first safe-math + root-find worked example (CGS, anchored numbers).
- `10-theory/index.md` — the 10-principle AD-safe-numerics thesis (design §9.1); each
  principle links to a method page.
- `10-theory/rootfinding.md` — `lax.scan` fixed-iteration vs `while_loop`; `bisect`
  zero-grad caveat; `newton`/`newton_ppf`.
- `10-theory/cumulative-trapz.md` — Newton–Cotes, the dx-outside uniform path, parity.
- `30-decisions/` — port ADRs `0001`–`0011` from `.adr/` (`.adr/` is gitignored/local;
  the docs copy is the published record). Keep their frontmatter; add an index.
- `40-api/index.md` — reference landing enumerating the public modules (units, constants,
  astrometry, coords, numerics, spatial, params, testing, jaxconfig).

**Step 4 — Write honest stubs** for `20-architecture`, `50-howto`, `60-validation`,
`90-development-log`, `95-release`, `99-bibliography`: each an `index.md` with a one-line
orientation and a visible **"Planned — not yet written"** admonition. They must build clean.

**Step 5 — Build gate.**
```bash
cd docs && myst build 2>&1 | tee /tmp/myst-build.log
grep -i "warning" /tmp/myst-build.log && echo "FIX WARNINGS" || echo "0 content warnings"
```
Expected: **0 content warnings**; all cross-refs/citations resolve. Fix every warning (a
broken xref or missing target is a real defect).

**Step 6 — Commit.**
```bash
git add docs/myst.yml docs/index.md docs/00-getting-started docs/10-theory \
        docs/20-architecture docs/30-decisions docs/40-api docs/50-howto \
        docs/60-validation docs/90-development-log docs/95-release docs/99-bibliography
git commit  # message below
```
```
docs(phase-c): mystmd Diataxis site — skeleton + v1 core

Full §9/ADR-0005 skeleton + dual-front-door landing; real content for
getting-started, the AD-safe-numerics theory thesis + rootfinding/cumulative-trapz
method pages, 11 ported ADRs, and the API reference landing. Remaining sections
are honest 'planned' stubs. `myst build` -> 0 content warnings.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

---

## Task 4 — CHANGELOG.md

**Files:** Create `CHANGELOG.md`.

**Step 1 — Gather provenance.** The entries must trace to real commits:
```bash
git log --oneline --no-merges b6d65b3~30..b6d65b3   # Phase B + params arc
```
Cross-check against `.claude-work/PHASE_B_COMPLETE.md` (T0–T8 table) and
`.claude-work/JAXSTRO_PARAMS_COMPLETE.md`.

**Step 2 — Write `CHANGELOG.md`** in keep-a-changelog format, heading `## 0.1.0
(unreleased)`, with grouped subsections:
- **Added** — `jaxstro.params` (Parameterization + Identity/Exp/Softplus/Sigmoid
  bijectors); PhotometricUnits + Jy/AB constants (Oke & Gunn 1983); quadrature factory
  (GL/GH + Hermite); `inverse_cdf_draw`, generic `newton_ppf`; `jaxstro.testing`
  grad-audit engine; constants round-out (ALPHA_FS, E_ESU, R_E, SIGMA_T, R_GAS);
  hatchling/Apache-2.0/py.typed/uv.lock release baseline.
- **Changed** — `cumulative_trapz` uniform path → dx-outside (progenax parity);
  `condition_number` singular sentinel → `+inf`.
- **Fixed** — A_RAD `7.565767e-15 → 7.565733250e-15` (= 4σ/c exactly); Julian-vs-tropical
  year provenance comment; `project_onto(eps=0)` NaN.
Header text: "All notable changes … adheres to Semantic Versioning."

**Step 3 — Verify entries map to commits.**
```bash
grep -n "params\|PhotometricUnits\|cumulative_trapz\|A_RAD\|condition_number" CHANGELOG.md
```
Expected: each headline claim corresponds to a real Phase-B/params commit (spot-check 3–4).

**Step 4 — Commit.**
```bash
git add CHANGELOG.md
git commit -m "docs(phase-c): CHANGELOG (keep-a-changelog) for Phase B + jaxstro.params

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5 — CLAUDE.md hardening + README staleness fix

**Files:** Modify `CLAUDE.md` (package-level), `README.md`.

**Step 1 — Harden `CLAUDE.md`** to progenax depth, adapted to infrastructure. Add/expand:
- **Quick Commands** — the `env -u VIRTUAL_ENV uv run --no-sync pytest` gate (full +
  `--extra ml`), tier selection, `scripts/check.sh`, ruff/mypy.
- **AD-safe-numerics patterns** — the `where`-trap (NaN backprop through dead branch →
  double-`where`), fixed-iteration not `while_loop`, saturation as silent gradient killer.
- **Critical Invariants** — `cumulative_trapz` dx-outside; Gauss-Hermite probabilists'-via-
  physicists (hermgauss + √2) parity; `condition_number` → `+inf`; `bisect` structurally-zero
  grad w.r.t. params (use `newton`/`newton_ppf`); `params` cached-derived-leaf caveat (fit
  the leaf the observable reads).
- **Provenance discipline** — every constant cites CODATA 2018 / IAU 2015 / Oke & Gunn 1983.
- **Definition of Complete**, **Common Issues**, **Debugging Checklist** sections.
Keep the existing brain-spoke handshake + status-update sections.

**Step 2 — Fix `README.md` staleness** to match `pyproject.toml`:
- Python badge + "Requirements": `3.10+` → `3.11+`.
- JAX badge + text: `0.4.28+` → `0.10.1+`.
- "Minimal dependencies" line: note `jax`, `jaxlib`, `equinox`, `jaxtyping` (equinox is now
  a core dep per ADR-0002 — the current "Only JAX and jaxtyping" is stale).
- Status badge: keep `v0.1.0` consistent with pyproject `version = "0.1.0"`.

**Step 3 — Verify the staleness fixes landed.**
```bash
grep -n "3.10\|0.4.28" README.md && echo "STILL STALE — fix" || echo "README matches pyproject"
```
Expected: no stale `3.10` / `0.4.28` strings remain (the python-3.10 badge URL included).

**Step 4 — Commit.**
```bash
git add CLAUDE.md README.md
git commit -m "docs(phase-c): progenax-depth CLAUDE.md + README staleness fix

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Close-out (after T5, before any merge talk)

1. **Final whole-arc review** — independent superpowers:code-reviewer over the full T1–T5
   diff against this plan + the design doc + CLAUDE.md constraints.
2. **Full gate green:** `bash scripts/check.sh` passes end-to-end; `myst build` clean.
3. **Completion doc** `.claude-work/PHASE_C_COMPLETE.md` (what shipped, per-task commits,
   verification evidence, lessons, Phase-D handoff).
4. **Update `STATUS.md`** (`next:` → Phase D gate/merge/push/tag held for Anna; `blocker:`).
5. **`brain "phase C release-hardening complete — …"`** capture (brain is pull-only).
6. **Stop.** Report to Anna. **No push, no merge, no tag** without her separate explicit
   words. Branch `feature/phase-c-release-hardening` is kept until merged AND pushed.

## Out of scope (Phase D / Anna's call)

`myst build` CI job · GitHub Pages deploy · standalone `validation/validate_*.py` CLIs +
plots · push / tag / PyPI · release-staging (namespace vs `jaxstro-core` rename) · sibling
pyproject floor bumps.
