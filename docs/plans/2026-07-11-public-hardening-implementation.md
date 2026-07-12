# jaxstro Public Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Establish a source-verified, AD-safe, provenance-backed public jaxstro with efficient CI, current docs, and a Pages-ready release checklist.

**Architecture:** Work in explicitly approved slices. First restore a fully observable local gate, then harden from verified findings, protect that baseline with a fast PR workflow, hoist generic provenance-card tooling, and only then refresh/publish documentation. Keep runtime dependencies minimal: registry parsing belongs in dev/docs tooling while generic schema/rendering lives in `jaxstro.testing`.

**Tech Stack:** Python 3.11–3.13, JAX, pytest, Ruff, mypy, uv, GitHub Actions, MyST, YAML/PyYAML in dev/docs tooling only, GitHub Pages.

## Execution status — 2026-07-12

- [x] Slice A0 — verified baseline restoration
- [x] Slice A — source-backed audit and hardening
- [x] Slice D0 — efficient public CI
- [x] Slice B — provenance registry hoist
- [x] Slice C — documentation currency, page pedagogy, JaxtroViz, and rendered-DOM gate
- [x] Slice D.1 — local Pages workflow and `/jaxstro` base-path proof
- [x] Slice D.2 — release checklist, citation/contributor metadata, and sdist hardening
- [ ] Slice D.3 — remote publication, deliberately stopped pending separate authorization

---

## Program rules

- Anna approves every slice before any code or workflow change. Stop at each slice checkpoint.
- Preserve the existing untracked kickoff file; stage only files belonging to the approved task.
- Use `@evidence-first-scientific-execution` for meaningful verification commands.
- Use primary sources or rendered PDFs for scientific/convention claims; never cite memory.
- A numerical test failure is evidence, not a reason to relax a tolerance.
- Run `bash scripts/check.sh` before each slice commit whenever the slice touches Python, tests, or CI.
- Do not push, modify GitHub settings, deploy Pages, tag, or upload to PyPI without Anna's explicit remote-action approval.

## Slice A0 — restore the baseline gate

**Approval required:** Anna has approved the architecture, but must explicitly approve A0 before implementation.

### Task A0.1: Capture the failure and define a behavior-preserving test-edit contract

**Files:**
- Modify: `tests/unit/test_spatial.py:10-40,1093-1112`
- Modify: `STATUS.md:1-12`

**Step 1: Re-run the narrow lint failure**

Run: `env -u VIRTUAL_ENV uv run --no-sync ruff check tests/unit/test_spatial.py`

Expected: seven violations: four `E402` errors and three `E702` errors; no semantic test failure is claimed.

**Step 2: Write the failing invariant checks before editing style**

Add or identify targeted tests that cover the nearby exact-neighbor randomized and clustered cases. Do not change their generated coordinates, cutoff, capacity, assertions, or expected pair-set semantics.

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/test_spatial.py -q`

Expected: current behavioral result recorded before style-only edits.

**Step 3: State update boundary**

Prepare a factual `STATUS.md` replacement for the stale quantity branch claim using `git merge-base --is-ancestor 655e756 HEAD` and the baseline result. Do not claim a live suite count until the gate reaches pytest.

### Task A0.2: Make only the minimal Ruff repair

**Files:**
- Modify: `tests/unit/test_spatial.py:10-40,1093-1112`

**Step 1: Fix import order without changing x64 timing**

Keep imports in one module-level block, then call `jax.config.update("jax_enable_x64", True)` before the module constructs any arrays. The repaired shape is:

```python
import jax
import jax.numpy as jnp
import pytest

from jaxstro.spatial import ...
from jaxstro.spatial.morton import MAX_BITS_3D

jax.config.update("jax_enable_x64", True)
```

**Step 2: Split only the three semicolon-separated assignments**

```python
origin = jnp.array([-1.0, -1.0, -1.0])
box = 2.0
cutoff = 0.3
```

Do not alter values or test control flow.

**Step 3: Verify the narrow lint and test result**

Run:

```bash
env -u VIRTUAL_ENV uv run --no-sync ruff check tests/unit/test_spatial.py
env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/test_spatial.py -q
```

Expected: Ruff exits 0; the targeted behavioral result agrees with Task A0.1.

### Task A0.3: Establish the new full baseline and commit

**Files:**
- Modify: `STATUS.md:1-12`

**Step 1: Run the prescribed full gate**

Run: `bash scripts/check.sh`

Expected: lock, formatting, lint, mypy, fast tests, ML integration, and wheel smoke all exit 0. Record exact test counts from command output, not an assumed June count.

**Step 2: Update status from observed evidence**

Replace the stale quantity branch text; state the actual baseline outcome, current next slice, and whether any gate failed. Keep release/naming decisions explicitly open.

**Step 3: Recheck status and commit the coherent A0 slice**

Run:

```bash
git diff --check
bash scripts/check.sh
git status --short
```

Commit only A0 files:

```bash
git add tests/unit/test_spatial.py STATUS.md
git commit -m "chore: restore verified local baseline"
```

**Checkpoint:** Report the complete local-gate evidence and request Anna's approval for Slice A. Do not start the audit automatically.

## Slice A — adversarial audit and hardening

**Approval required:** request after A0 evidence is reported.

### Task A.1: Build the auditable inventory and findings ledger

**Files:**
- Create: `docs/audits/2026-07-11-core-hardening-audit.md`
- Inspect: `src/jaxstro/constants.py`, `src/jaxstro/units.py`, `src/jaxstro/coords.py`, `src/jaxstro/astrometry.py`
- Inspect: `src/jaxstro/numerics/`, `src/jaxstro/testing/grad_audit.py`
- Inspect: `tests/unit/test_constants.py`, `tests/unit/test_units.py`, `tests/unit/test_coords.py`, `tests/unit/test_astrometry.py`, `tests/integration/test_grad_audit.py`

**Step 1: Enumerate public claims and entry points**

Record symbol, source/provenance comment, code location, existing value test, transform/gradient status, and evidence state. Mark missing proof as `open`, never as a defect until verified.

**Step 2: Verify every constant or convention against its primary source**

Use source notes and rendered PDFs/tables. Capture exact locator, unit system, epoch/frame, rounding policy, and comparison result in the ledger.

**Step 3: Record numerical and API-contract probes separately**

For each candidate, name the forward invariant, boundary input, expected gradient contract, and source of truth. Do not conflate eager validation with JIT-traced behavior.

**Step 4: Commit audit evidence only if it is source-backed**

```bash
git add docs/audits/2026-07-11-core-hardening-audit.md
git commit -m "docs(audit): inventory jaxstro core evidence"
```

### Task A.2: Extend jaxstro's existing AD-vs-FD registry

**Files:**
- Modify: `tests/integration/test_grad_audit.py`
- Modify: `tests/validation/test_grad_audit.py` if the public-coverage ratchet belongs there
- Inspect: `src/jaxstro/testing/grad_audit.py`, `src/jaxstro/testing/contracts.py`

**Step 1: Write one failing case per differentiated public contract**

Use `Case(...)` from `jaxstro.testing.grad_audit` with an explicit `expect`, `grad_contract`, tolerance, finite probe, and at least one meaningful edge probe where applicable. A discrete spatial operation must declare a known limitation rather than pretend to be smooth.

**Step 2: Run the focused audit test**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest tests/integration/test_grad_audit.py tests/validation/test_grad_audit.py -q`

Expected: failing only for an identified missing/incorrect contract, never because a case uses an arbitrary unstable finite-difference point.

**Step 3: Fix only a confirmed root cause with a regression test**

For each ledger finding, create the failing value/edge/gradient regression first, apply the smallest implementation change, then verify the focused test and full gate. Add the audit source/locator and claim boundary to the ledger.

**Step 4: Commit each independently verified finding**

```bash
git add <finding-specific-source> <finding-specific-test> docs/audits/2026-07-11-core-hardening-audit.md
git commit -m "fix(<area>): harden verified <contract>"
```

### Task A.3: Add public-repository hygiene checks and close the audit

**Files:**
- Modify: `docs/audits/2026-07-11-core-hardening-audit.md`
- Inspect: `.gitignore`, tracked-file inventory

**Step 1: Scan tracked content for local/internal residue**

Run:

```bash
git grep -n -E '/Users/anna|\.brain-drafts|knowledge/raw|\.pdf$' -- ':!docs/plans/2026-07-11-session-kickoff-public-hardening.md'
git ls-files | rg '(^|/)(dist|_build|\.mypy_cache|__pycache__)(/|$)'
```

Expected: investigate each hit; allow only intentional public documentation references.

**Step 2: Run the full gate and close only verified findings**

Run: `bash scripts/check.sh`

**Step 3: Commit audit closure**

```bash
git add docs/audits/2026-07-11-core-hardening-audit.md
git commit -m "docs(audit): close verified core hardening findings"
```

**Checkpoint:** Report code fixes, source-backed findings, unresolved questions, and gate evidence. Request Anna's approval for D0.

## Slice D0 — efficient public CI

**Approval required:** request after Slice A.

### Task D0.1: Write the workflow contract before changing triggers

**Files:**
- Modify: `.github/workflows/tests.yml`
- Create: `.github/workflows/full-gate.yml`
- Modify: `scripts/check.sh` only if the local mirror must gain a new focused gradient/docs command

**Step 1: Define the PR job set**

Keep `lock-check`, Ruff plus formatter check, mypy, one pinned current-Python fast tier, the A gradient gate, and wheel smoke. Add `cancel-in-progress: true` per ref. Finish with an `if: always()` aggregate job named `tests`.

**Step 2: Define the exhaustive workflow**

Run only on schedule and `workflow_dispatch`: Python 3.11–3.13 fast matrix, ML integration, full validation/slow coverage as applicable, and the docs gate after C. Do not hide failures behind `continue-on-error`.

**Step 3: Use lockfile-keyed uv caching and explicit timeouts**

Use `astral-sh/setup-uv@v6`, the existing JAX memory environment, timeout bounds, and caching only when it does not bypass `uv sync --locked` correctness.

### Task D0.2: Validate workflow syntax and local equivalence

**Files:**
- Modify: `.github/workflows/tests.yml`
- Create: `.github/workflows/full-gate.yml`

**Step 1: Compare commands to local gates**

Run: `bash scripts/check.sh`

Expected: local commands cover or exceed the required PR job contracts.

**Step 2: Lint workflow YAML with locally available tooling; otherwise inspect exact YAML structure**

Confirm trigger separation, permissions, aggregate dependencies, job names, matrix values, and cancellation group.

**Step 3: Commit without enabling remote branch protection**

```bash
git add .github/workflows/tests.yml .github/workflows/full-gate.yml scripts/check.sh
git commit -m "ci: enable efficient public PR gate"
```

**Checkpoint:** Anna decides whether to push and whether to configure GitHub branch protection. Request approval for B only after the local CI contract is verified.

## Slice B — provenance registry hoist

**Approval required:** request after D0.

### Task B.1: Port generic schema/rendering with no core runtime parser dependency

**Files:**
- Create: `src/jaxstro/testing/provenance_cards.py`
- Modify: `src/jaxstro/testing/__init__.py`
- Create: `tests/unit/test_provenance_cards.py`

**Step 1: Write failing generic-schema tests**

Test required fields, allowed status values, deterministic family ordering, malformed card errors, stable Markdown/MyST rendering, and code/validation reference formatting using in-memory mappings. Do not import PyYAML in the library module.

**Step 2: Implement the minimal standard-library API**

Expose typed card validation and render functions that accept already-parsed mappings. Keep filesystem and YAML policy out of the installed module.

**Step 3: Run focused tests and type checking**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/test_provenance_cards.py -q
env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro/testing
```

### Task B.2: Add jaxstro's registry, generator, and enforcement

**Files:**
- Create: `docs/provenance/registry/constants.yaml`
- Create: `docs/provenance/registry/transforms.yaml`
- Create: `docs/provenance/registry/atmospheres.yaml`
- Create: `scripts/build_provenance_registry.py`
- Create: `tests/validation/provenance_cards/test_registry.py`
- Create: generated MyST reference pages at the approved docs location
- Modify: `docs/myst.yml`
- Modify: `pyproject.toml`, `uv.lock` only to add a dev/docs-only YAML parser if absent

**Step 1: Write enforcement tests first**

Tests must require every card field, a source locator, existing code references, existing validation paths, deterministic output, and committed-generated equality.

**Step 2: Add parser-adapter generator**

The script reads registry YAML, passes parsed cards to `jaxstro.testing.provenance_cards`, supports `--emit` and `--check`, and writes no timestamps/randomness.

**Step 3: Add source-backed cards**

Use only source-verified A evidence. Each card identifies scope, unit/frame conventions, sources/locators, code symbols, validation, status, and deviations. Do not card an unverified atmosphere claim.

**Step 4: Regenerate, test, and commit**

```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/build_provenance_registry.py --emit
env -u VIRTUAL_ENV uv run --no-sync python scripts/build_provenance_registry.py --check
env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/test_provenance_cards.py tests/validation/provenance_cards -q
bash scripts/check.sh
git add src/jaxstro/testing scripts/build_provenance_registry.py docs/provenance docs/myst.yml tests pyproject.toml uv.lock
git commit -m "feat(provenance): add jaxstro model-card registry"
```

**Checkpoint:** Report the runtime dependency boundary, each card's evidence state, and regenerated-page proof. Request approval for C.

## Slice C — documentation currency and pedagogy

**Approval required:** request after B. C0 is a discovery/verification pass; every later page is individually approval-gated.

### Task C0.1: Inventory documentation drift and executable examples

**Files:**
- Create: `docs/audits/2026-07-11-docs-currency-audit.md`
- Inspect: `README.md`, `docs/myst.yml`, `docs/00-getting-started/`, `docs/10-theory/`, `docs/20-architecture/`, `docs/40-api/`, `docs/50-howto/`, `docs/60-validation/`
- Create: `tests/integration/test_readme_examples.py` if absent

**Step 1: Enumerate fenced Python snippets and public API claims**

Classify each as executable, illustrative, or intentionally pseudocode. Map it to an installed symbol and expected behavior.

**Step 2: Test README quick-start examples in a subprocess**

Use the public import path and enable x64 before arrays. Assert physically meaningful outputs/units where the README makes a numerical claim.

**Step 3: Build docs and inspect generated evidence**

Run:

```bash
cd docs && myst build --html
rg -n 'warning|error' _build/logs/myst.build.json
```

Then inspect `docs/_build/site/myst.xref.json` for resolved target URLs. For any behavior that depends on rendered markup, inspect the built HTML/DOM rather than the MyST AST.

**Step 4: Report page inventory and request page order approval**

Do not edit pedagogical prose or figures in C0 without Anna's page-specific approval.

### Task C1+: Revise one approved documentation page at a time

**Files:**
- Modify: exactly one approved `docs/**/*.md` page per task
- Modify: `docs/40-api/index.md` only when its approved API card references change
- Create/modify: `laboratory/` figure specs and tests only when the approved page needs a figure

**Step 1: Present a page card to Anna**

Include learner goal, misconception addressed, source/code evidence, exact examples, proposed figure, and proof command. Stop for approval.

**Step 2: Write/extend executable tests before changing a claim**

Add a focused snippet or numerical-invariant test. Use a deterministic seed/config for any plotted data.

**Step 3: Revise the approved page and, if needed, figure spec**

Use native cross-references. Never write raw HTML to force target/rel behavior. Generated PDF/PNG remain ignored; committed WebP is deterministic and has descriptive alt text.

**Step 4: Run page and site gates, then commit the one-page slice**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest <approved-focused-tests> -q
cd docs && myst build --html
cd .. && <approved-docs-gate>
git add <approved-page> <approved-tests> <approved-assets>
git commit -m "docs: refresh <approved-page-topic>"
```

### Task C2: Make documentation verification a reusable gate

**Files:**
- Create: `scripts/check_docs.sh` or `docs/Makefile` gate target
- Modify: `scripts/check.sh`
- Modify: `.github/workflows/full-gate.yml`
- Modify: `STATUS.md`, `CHANGELOG.md`

**Step 1: Write failing checks for broken Markdown links, zero content warnings, and xref assumptions**

Use generated xref data as the URL oracle. A duplicate root-flat slug must fail the check rather than silently become `-1`.

**Step 2: Implement the three-part docs gate**

The gate scans links, runs the MyST HTML build, and validates content-warning/xref invariants. Keep it usable locally and from CI.

**Step 3: Run complete docs and local quality gates, then commit**

```bash
bash scripts/check_docs.sh
bash scripts/check.sh
git add scripts/check_docs.sh scripts/check.sh .github/workflows/full-gate.yml docs STATUS.md CHANGELOG.md tests
git commit -m "docs: add verified documentation gate"
```

**Checkpoint:** Present the final DOM/build proof and request Anna's approval for D.

## Slice D — Pages publication and release checklist

**Approval required:** request after C. Remote publication is a separate explicit approval within this slice.

### Task D.1: Add Pages workflow using jaxstro's actual build output

**Files:**
- Create: `.github/workflows/pages.yml`
- Modify: `scripts/check_docs.sh` or docs gate target if it needs a `BASE_URL` mode

**Step 1: Write workflow contract**

Use `actions/checkout@v4`, `actions/setup-node@v4` with Node 20, the locally validated mystmd version, `BASE_URL=/${{ github.event.repository.name }}`, the full docs gate, `actions/upload-pages-artifact@v3`, and `actions/deploy-pages@v4`. Upload `docs/_build/site`, not progenax's `docs/website/_build/html`.

**Step 2: Verify the local base-path build**

Run the docs gate with `BASE_URL=/jaxstro` and inspect built asset/internal-link URLs in rendered HTML and generated xref data.

**Step 3: Commit local workflow configuration only**

```bash
git add .github/workflows/pages.yml scripts/check_docs.sh
git commit -m "ci(docs): add GitHub Pages deployment workflow"
```

### Task D.2: Write and validate the release checklist

**Files:**
- Create: `docs/95-release/checklist.md`
- Modify: `docs/myst.yml`
- Create/modify: `CITATION.cff`, `CONTRIBUTING.md` only after their content is reviewed

**Step 1: Write checkable release rows**

Cover version/tag, public CI, docs/Pages, citation/Zenodo, sdist contents, license/metadata, contributor guidance, downstream floor bumps, and public-hygiene scan.

**Step 2: Add explicit stop gates**

The checklist must state that no PyPI upload occurs until Anna chooses the permanent package name and explicitly authorizes upload. It must state that progenax's PyPI path remains blocked on that decision.

**Step 3: Build docs and commit**

```bash
cd docs && myst build --html
git add docs/95-release/checklist.md docs/myst.yml CITATION.cff CONTRIBUTING.md
git commit -m "docs(release): add publication checklist"
```

### Task D.3: Remote publishing only on explicit authorization

**Files:** no local code change required unless a deployment failure produces a reviewed fix.

**Step 1: Request explicit authorization**

Ask separately for: push of approved commits, GitHub Pages source configuration, and deploy verification. Do not infer permission from the local workflow commit.

**Step 2: Configure and deploy**

After approval, set Pages source to GitHub Actions, push, monitor the workflow, and validate the public URL in a browser at `https://jaxstro.github.io/jaxstro/`.

**Step 3: Run final hygiene evidence and report**

Repeat tracked-file/local-path scans; report deployed commit, rendered-DOM result, remaining release decisions, and whether PyPI is intentionally blocked.

## End state

The program is complete only when every approved slice has its recorded evidence, the docs site is deployed and verified in rendered form, and the release checklist makes any unresolved naming/upload decision impossible to miss. The first PyPI upload is intentionally outside this plan until Anna chooses the irreversible package name.
