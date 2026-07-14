# Progenax-Jaxstro Adoption Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an evidence-backed, symbol-level Progenax adoption audit that separates supported Jaxstro migrations from future ownership proposals, protects Progenax scientific policy, and inventories stale-test cleanup without changing runtime behavior.

**Architecture:** Jaxstro owns one authoritative tracked report and a focused structural contract test. Progenax remains read-only except for an ignored maintainer-local execution record during the audit and one repository-status update after Anna approves the completed report. Findings accumulate layer by layer, then become three approval-gated recommendation queues; no runtime migration or test deletion occurs in this plan.

**Tech Stack:** Markdown, Python 3.11+, pytest, Git, ripgrep, Jaxstro's generated scientific-contract registry, and Progenax's existing test and documentation evidence.

## Global Constraints

- Treat `docs/superpowers/specs/2026-07-14-progenax-jaxstro-adoption-audit-design.md` as the controlling specification.
- Startrax and Gravax are protected: do not inspect, edit, test, migrate, or use either repository as an audit gate.
- Do not modify Progenax runtime source, tracked tests, tracked product documentation, configuration, dependencies, public API, or generated artifacts.
- The only Progenax file created before report approval is the ignored maintainer-local `docs/plans/2026-07-14-jaxstro-adoption-audit-and-refactor.md` record.
- Update Progenax `STATUS.md` exactly once, after Anna approves the completed audit report.
- Continue to treat `jaxstro.units` as the stable production unit-system contract and `jaxstro.quantity` as an opt-in alpha surface with no authorized Progenax adoption.
- Only `ADOPT_READY` findings may enter a future migration plan.
- Other readiness dispositions are not migration authorization.
- Classify affected Progenax tests as `KEEP`, `REWRITE`, `REPLACE`, or `STALE_CANDIDATE`; do not delete or rewrite tests during this audit.
- A future stale-test deletion requires replacement contract coverage, a mutation or deliberate-break check, focused and affected green gates, no remaining consumers, and Anna's explicit cleanup approval.
- Never remove a failing test merely to make a gate pass.
- Use ASCII punctuation in authored prose and LaTeX for mathematical notation.
- Do not add dependencies, push, publish, release, or deploy.
- Commit Jaxstro audit slices separately. Do not commit the ignored Progenax execution record.

---

## File and responsibility map

### Jaxstro

- Create `docs/audits/2026-07-14-progenax-jaxstro-adoption-audit.md`: authoritative snapshots, coverage, findings, recommendation queues, validation matrix, stale-test dispositions, and decision register.
- Create `tests/integration/test_progenax_adoption_audit.py`: portable structural and completion ratchets that do not require a Progenax checkout at test time.
- Modify `STATUS.md` only after report approval: add one concise `previous:` record without replacing unrelated `next:`, `blocker:`, or `due:` state.

### Progenax

- Create ignored `docs/plans/2026-07-14-jaxstro-adoption-audit-and-refactor.md`: live snapshot, command log, layer progress, referenced finding IDs, review decisions, and deviations.
- Modify `STATUS.md` only after report approval: add one concise `previous:` record without replacing unrelated release work.

### Explicitly unchanged

- `src/progenax/**`
- `src/experimental/**`
- `tests/**`, `scripts/**`, and `docs/website/**` in Progenax
- dependency and lock files in both repositories
- all Startrax and Gravax files

---

### Task 1: Establish immutable snapshots and the audit contract

**Files:**
- Create: `tests/integration/test_progenax_adoption_audit.py`
- Create: `docs/audits/2026-07-14-progenax-jaxstro-adoption-audit.md`
- Create, maintainer-local: `../progenax/docs/plans/2026-07-14-jaxstro-adoption-audit-and-refactor.md`
- Read: `docs/superpowers/specs/2026-07-14-progenax-jaxstro-adoption-audit-design.md`
- Read: `../progenax/CLAUDE.md`
- Read: `../progenax/README.md`
- Read: `../progenax/pyproject.toml`

**Interfaces:**
- Consumes: the approved design and live Git state.
- Produces: the report section contract, a portable structural test, and the Progenax execution ledger used by later tasks.

- [ ] **Step 1: Verify and record live snapshots**

Run:

```bash
git status --short --branch
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git -C ../progenax status --short --branch
git -C ../progenax status --porcelain --untracked-files=no
git -C ../progenax rev-parse --abbrev-ref HEAD
git -C ../progenax rev-parse HEAD
git -C ../progenax check-ignore -v docs/plans/2026-07-14-jaxstro-adoption-audit-and-refactor.md
```

Expected:

- Jaxstro may show the pre-existing untracked `.superpowers/` path; leave it untouched.
- Progenax's tracked-only status prints nothing. If it prints a path, stop before creating either audit document.
- `git check-ignore` identifies Progenax's `docs/plans/` rule.
- Record the literal branches and 40-character commits printed here in both audit documents.

- [ ] **Step 2: Write the failing structural test**

Create `tests/integration/test_progenax_adoption_audit.py`:

```python
"""Structural contracts for the Progenax-Jaxstro adoption audit."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "docs/audits/2026-07-14-progenax-jaxstro-adoption-audit.md"

REQUIRED_HEADINGS = (
    "# The Shared Machinery Boundary",
    "## Purpose and safety contract",
    "## Audited snapshots and exclusions",
    "## Coverage inventory",
    "## Progenax responsibility map",
    "## Existing canonical Jaxstro use",
    "## Existing-API adoption findings",
    "## Future Jaxstro ownership proposals",
    "## Progenax-owned no-move list",
    "## Experimental and Informax boundary",
    "## Units and quantity boundary",
    "## Recommendation queues",
    "## Validation matrix",
    "## Test disposition and stale-test cleanup",
    "## Decision register",
)

READINESS_DISPOSITIONS = (
    "ADOPT_READY",
    "JAXSTRO_HARDEN_FIRST",
    "PROPOSE_FOR_JAXSTRO",
    "KEEP_IN_PROGENAX",
    "DEFER_EXPERIMENTAL",
    "REJECT",
)

TEST_DISPOSITIONS = ("KEEP", "REWRITE", "REPLACE", "STALE_CANDIDATE")


def _report_text() -> str:
    return REPORT.read_text(encoding="utf-8")


def test_progenax_adoption_audit_has_required_structure() -> None:
    text = _report_text()

    for heading in REQUIRED_HEADINGS:
        assert heading in text
    for disposition in READINESS_DISPOSITIONS:
        assert f"`{disposition}`" in text
    for disposition in TEST_DISPOSITIONS:
        assert f"`{disposition}`" in text


def test_progenax_adoption_audit_preserves_protected_boundaries() -> None:
    text = _report_text()

    assert "Startrax" in text
    assert "Gravax" in text
    assert "does not inspect" in text.lower()
    assert "no migration" in text.lower()
    assert "jaxstro.units" in text
    assert "jaxstro.quantity" in text
    assert "opt-in alpha" in text
```

- [ ] **Step 3: Prove the test is red**

Run:

```bash
uv run pytest tests/integration/test_progenax_adoption_audit.py -q
```

Expected: two failures with `FileNotFoundError` for the missing audit report.

- [ ] **Step 4: Create the authoritative report skeleton**

Create the report with frontmatter `title`, `date: 2026-07-14`, and `status: in-progress`. Use every heading in `REQUIRED_HEADINGS` in that exact order.

Under `Audited snapshots and exclusions`, record literal paths, branches, commits, and tracked-worktree states from Step 1. State that Startrax and Gravax are protected and this audit does not inspect, edit, test, migrate, or use them as gates. State that the audit performs no migration.

Under `Coverage inventory`, use:

```markdown
| Layer | Family | Source scope | Evidence scope | Status | Finding IDs |
| --- | --- | --- | --- | --- | --- |
```

Enumerate all source, experimental, script, test, and documentation families from Tasks 2 through 7 with initial status `not-started`.

Define this exact finding block schema:

```markdown
### PXJ-001: Concise finding title

- **Surface:** exact Progenax paths and symbols
- **Responsibility:** scientific or numerical responsibility
- **Current Jaxstro use:** exact import or `none`
- **Proposed owner:** `Jaxstro`, `Progenax`, `Informax`, or `unchanged`
- **Disposition:** one readiness disposition
- **Evidence:** source, tests, documentation, and consumers
- **Transform contract:** `jit`, `vmap`, differentiation, shape, and static-data behavior
- **Units contract:** explicit `G`, explicit units, wrapper default, or dimensionless
- **Test disposition:** one test disposition with exact test paths
- **Risk:** scientific, API, numerical, and dependency risks
- **Required gates:** exact pre-migration and post-migration evidence
- **Decision:** `OBSERVED`
- **Cleanup allowed:** `false`
```

The example defines the schema only; do not retain `PXJ-001` as a fabricated finding. Real IDs begin in Task 2 and remain stable.

Define every readiness and test disposition. State that only `ADOPT_READY` may enter a later migration plan and stale-test removal is prohibited during this audit.

- [ ] **Step 5: Create the ignored Progenax ledger**

Create the maintainer-local record with these headings:

```markdown
# Jaxstro adoption audit and refactor record

## Scope and prohibitions
## Repository snapshots
## Layer progress
## Command log
## Referenced Jaxstro finding IDs
## Test dispositions
## Deviations and stale evidence
## Review decisions
## Closeout evidence
```

Record literal snapshot values. State that the audit modifies no tracked Progenax source, tests, product documentation, configuration, dependencies, or public API. Initialize layers to `not-started` and other empty sections to `No entries.`

- [ ] **Step 6: Verify green structure and ignored Progenax state**

Run:

```bash
uv run pytest tests/integration/test_progenax_adoption_audit.py -q
git -C ../progenax check-ignore -v docs/plans/2026-07-14-jaxstro-adoption-audit-and-refactor.md
git -C ../progenax status --porcelain --untracked-files=no
```

Expected: `2 passed`, the ledger matches the ignore rule, and Progenax prints no tracked changes.

- [ ] **Step 7: Commit the Jaxstro audit contract**

```bash
git add docs/audits/2026-07-14-progenax-jaxstro-adoption-audit.md tests/integration/test_progenax_adoption_audit.py
git commit -m "docs: establish progenax adoption audit contract"
```

Do not add any Progenax file.

---

### Task 2: Inventory current Jaxstro use and shared mechanics

**Files:**
- Modify: `docs/audits/2026-07-14-progenax-jaxstro-adoption-audit.md`
- Modify, maintainer-local: `../progenax/docs/plans/2026-07-14-jaxstro-adoption-audit-and-refactor.md`
- Read: `docs/validation/contracts.json`
- Read: `src/jaxstro/**` and matching Jaxstro tests and API pages
- Read: `../progenax/src/progenax/numerics.py`
- Read: `../progenax/src/progenax/defaults.py`
- Read: `../progenax/src/progenax/__init__.py`
- Read: `../progenax/src/progenax/stellar.py`
- Read: `../progenax/src/progenax/analytical/few_body.py`
- Read: `../progenax/src/progenax/binaries/orbital_state.py`
- Read: `../progenax/src/progenax/cluster/constants.py`
- Read: `../progenax/src/progenax/diagnostics/q_approx.py`
- Read: `../progenax/src/progenax/imf/base.py`
- Read: `../progenax/src/progenax/imf/chabrier.py`
- Read: `../progenax/src/progenax/kinematics/api.py`
- Read: `../progenax/src/progenax/kinematics/plummer_df.py`
- Read: `../progenax/src/progenax/profiles/api.py`

**Interfaces:**
- Consumes: Task 1 snapshots and finding schema.
- Produces: the complete direct-import map, Jaxstro maturity evidence, shared-mechanics findings, and the stable-units/deferred-quantity boundary.

- [ ] **Step 1: Capture every direct Jaxstro import**

Run:

```bash
rg -n "^(from jaxstro|import jaxstro)" ../progenax/src ../progenax/tests ../progenax/scripts ../progenax/docs ../progenax/pyproject.toml
rg -l "(^|[^A-Za-z0-9_])jaxstro([.]|[^A-Za-z0-9_])" ../progenax/src ../progenax/tests ../progenax/scripts ../progenax/docs ../progenax/pyproject.toml | sort
```

Record result counts and commands in the Progenax ledger.

- [ ] **Step 2: Match each imported symbol to current Jaxstro evidence**

For every literal imported symbol, run:

```bash
rg -n "literal_symbol_name" src/jaxstro
rg -n "literal_symbol_name" docs/validation/contracts.json
rg -n "literal_symbol_name" tests
rg -n "literal_symbol_name" docs/50-api docs/20-methods docs/30-representations
```

Replace `literal_symbol_name` with the actual symbol each time. Record `classified`, `public but unclassified`, or `not public` from evidence. A name match alone does not establish compatibility.

- [ ] **Step 3: Enumerate local generic kernels and wrappers**

Run:

```bash
rg -n "^(def|class) |jax[.]|jnp[.]|lax[.]|random[.]|linalg[.]|searchsorted|interp|trapz|quadrature|root|solve|cdf|ppf|validate|precision" \
  ../progenax/src/progenax/numerics.py \
  ../progenax/src/progenax/defaults.py \
  ../progenax/src/progenax/__init__.py \
  ../progenax/src/progenax/stellar.py \
  ../progenax/src/progenax/analytical/few_body.py \
  ../progenax/src/progenax/binaries/orbital_state.py \
  ../progenax/src/progenax/cluster/constants.py \
  ../progenax/src/progenax/diagnostics/q_approx.py \
  ../progenax/src/progenax/imf/base.py \
  ../progenax/src/progenax/imf/chabrier.py \
  ../progenax/src/progenax/kinematics/api.py \
  ../progenax/src/progenax/kinematics/plummer_df.py \
  ../progenax/src/progenax/profiles/api.py
```

For each candidate, inspect implementation, call sites, tests, and closest Jaxstro owner. Record mathematical meaning; units and explicit `G` behavior; dtype, shapes, broadcasting, and boundaries; `jit`, `vmap`, gradients, PRNG, static data, and precision; public status; errors; and tolerances.

- [ ] **Step 4: Write current-use, mechanics, units, and quantity findings**

Create real `PXJ-NNN` blocks with exact paths. Use `JAXSTRO_HARDEN_FIRST` when contract evidence is incomplete and `PROPOSE_FOR_JAXSTRO` when no supported owner exists.

State that Progenax retains stable `jaxstro.units` and its domain default. Record `jaxstro.quantity` only as `PROPOSE_FOR_JAXSTRO` or `DEFER_EXPERIMENTAL`; no alpha-quantity migration is authorized.

Map affected tests to dispositions without changing them. Mark dependency, mechanics, units, and quantity coverage rows `reviewed` and update the ledger.

- [ ] **Step 5: Verify and commit**

Run:

```bash
uv run pytest tests/integration/test_progenax_adoption_audit.py -q
git diff --check -- docs/audits/2026-07-14-progenax-jaxstro-adoption-audit.md
git -C ../progenax status --porcelain --untracked-files=no
```

Expected: tests pass, whitespace check is silent, and Progenax prints nothing.

Commit:

```bash
git add docs/audits/2026-07-14-progenax-jaxstro-adoption-audit.md
git commit -m "docs: audit progenax shared machinery"
```

---

### Task 3: Audit released structure, kinematics, clusters, and dynamics

**Files:**
- Modify: `docs/audits/2026-07-14-progenax-jaxstro-adoption-audit.md`
- Modify, maintainer-local: `../progenax/docs/plans/2026-07-14-jaxstro-adoption-audit-and-refactor.md`
- Read: `../progenax/src/progenax/profiles/**`
- Read: `../progenax/src/progenax/kinematics/**`
- Read: `../progenax/src/progenax/cluster/**`
- Read: `../progenax/src/progenax/diagnostics/**`
- Read: `../progenax/src/progenax/dynamics/**`
- Read: matching Progenax unit, integration, validation, and website files

**Interfaces:**
- Consumes: Task 2 mechanics decisions.
- Produces: a released-science ownership map separating reusable primitives from Progenax models and acceptance policy.

- [ ] **Step 1: Inventory every released symbol and export**

Run:

```bash
rg -n "^(class|def) |__all__" \
  ../progenax/src/progenax/profiles \
  ../progenax/src/progenax/kinematics \
  ../progenax/src/progenax/cluster \
  ../progenax/src/progenax/diagnostics \
  ../progenax/src/progenax/dynamics
```

Record every file as `reviewed` with finding IDs or `reviewed, no finding`. Include `__init__.py` exports.

- [ ] **Step 2: Trace each family across consumers and evidence**

For each literal public or ownership-relevant symbol, run:

```bash
rg -n "literal_symbol_name" ../progenax/src ../progenax/tests ../progenax/scripts ../progenax/docs/website
```

Identify scientific parameters and acceptance semantics. Record generic sub-operations separately rather than proposing movement of an entire model.

- [ ] **Step 3: Write findings and the no-move list**

Create material findings. Use `KEEP_IN_PROGENAX` for domain science even when Jaxstro should own a lower-level operation. Populate the no-move list with exact model families and reasons. Assign test dispositions without editing tests.

- [ ] **Step 4: Verify and commit**

```bash
uv run pytest tests/integration/test_progenax_adoption_audit.py -q
git diff --check -- docs/audits/2026-07-14-progenax-jaxstro-adoption-audit.md
git -C ../progenax status --porcelain --untracked-files=no
git add docs/audits/2026-07-14-progenax-jaxstro-adoption-audit.md
git commit -m "docs: map progenax structural science ownership"
```

Expected: tests pass and Progenax has no tracked changes.

---

### Task 4: Audit released populations, binaries, builders, and analytical models

**Files:**
- Modify: `docs/audits/2026-07-14-progenax-jaxstro-adoption-audit.md`
- Modify, maintainer-local: `../progenax/docs/plans/2026-07-14-jaxstro-adoption-audit-and-refactor.md`
- Read: `../progenax/src/progenax/imf/**`
- Read: `../progenax/src/progenax/binaries/**`
- Read: `../progenax/src/progenax/analytical/**`
- Read: `../progenax/src/progenax/builders.py`
- Read: `../progenax/src/progenax/builders_cluster.py`
- Read: `../progenax/src/progenax/stellar.py`
- Read: `../progenax/src/progenax/tidal.py`
- Read: `../progenax/src/progenax/protocols.py`
- Read: matching Progenax tests, scripts, and website files

**Interfaces:**
- Consumes: Tasks 2 and 3 ownership criteria.
- Produces: complete released-core population and composition coverage.

- [ ] **Step 1: Inventory symbols and public exports**

```bash
rg -n "^(class|def) |__all__" \
  ../progenax/src/progenax/imf \
  ../progenax/src/progenax/binaries \
  ../progenax/src/progenax/analytical \
  ../progenax/src/progenax/builders.py \
  ../progenax/src/progenax/builders_cluster.py \
  ../progenax/src/progenax/stellar.py \
  ../progenax/src/progenax/tidal.py \
  ../progenax/src/progenax/protocols.py
```

Record every file and package-root export in coverage.

- [ ] **Step 2: Classify policy versus machinery**

For every family, distinguish scientific distributions, empirical prescriptions, composition rules, and acceptance policies from generic numerical transforms. Trace each material symbol with:

```bash
rg -n "literal_symbol_name" ../progenax/src ../progenax/tests ../progenax/scripts ../progenax/docs/website
```

Preserve Progenax ownership of distribution functions, IMFs, binary population choices, cluster construction, tidal acceptance policy, and domain semantics unless exact source evidence shows purely generic machinery.

- [ ] **Step 3: Write findings and test dispositions**

Create required finding blocks, update the no-move list, mark coverage complete, and classify affected tests. A future shared primitive without current Jaxstro support remains `PROPOSE_FOR_JAXSTRO`.

- [ ] **Step 4: Verify and commit**

```bash
uv run pytest tests/integration/test_progenax_adoption_audit.py -q
git diff --check -- docs/audits/2026-07-14-progenax-jaxstro-adoption-audit.md
git -C ../progenax status --porcelain --untracked-files=no
git add docs/audits/2026-07-14-progenax-jaxstro-adoption-audit.md
git commit -m "docs: map progenax population science ownership"
```

Expected: tests pass and Progenax remains unchanged.

---

### Task 5: Audit experimental gravoturbulence and inference boundaries

**Files:**
- Modify: `docs/audits/2026-07-14-progenax-jaxstro-adoption-audit.md`
- Modify, maintainer-local: `../progenax/docs/plans/2026-07-14-jaxstro-adoption-audit-and-refactor.md`
- Read: `../progenax/src/experimental/gravoturb_fdf/README.md`
- Read: `../progenax/src/experimental/gravoturb_fdf/cluster.py`
- Read: `../progenax/src/experimental/gravoturb_fdf/masses.py`
- Read: `../progenax/src/experimental/gravoturb_fdf/field/**`
- Read: `../progenax/src/experimental/gravoturb_fdf/diagnostics/**`
- Read: `../progenax/src/experimental/gravoturb_fdf/theory/**`
- Read: `../progenax/src/experimental/gravoturb_fdf/inference/**`
- Read: `../progenax/src/experimental/gravoturb_fdf/validation/**`
- Read: `../progenax/tests/experimental/**`

**Interfaces:**
- Consumes: the stable released-core boundary.
- Produces: a separate appendix distinguishing Jaxstro mechanics, Informax policy, Progenax science, visualization consumers, and local experiments.

- [ ] **Step 1: Inventory every experimental symbol and ecosystem import**

```bash
rg -n "^(class|def) |^(from jaxstro|import jaxstro)|^(from informax|import informax)|^(from jaxstroviz|import jaxstroviz)" \
  ../progenax/src/experimental/gravoturb_fdf
```

Record every file as `reviewed` or `reviewed, no finding`. Keep banked validation scripts separate from current experimental library code.

- [ ] **Step 2: Classify ownership conservatively**

For each material primitive, choose generic Jaxstro machinery, Informax inference policy, Progenax gravoturbulence science, visualization/validation consumer, or local experiment. Default uncertain ownership to `DEFER_EXPERIMENTAL`. Existing Jaxstro quadrature use does not transfer ownership of the surrounding likelihood or science model.

- [ ] **Step 3: Write findings and test dispositions**

Populate the experimental boundary, create exact finding blocks, classify experimental tests and validation scripts, and mark coverage reviewed. Do not propose promotion, packaging, or deletion.

- [ ] **Step 4: Verify and commit**

```bash
uv run pytest tests/integration/test_progenax_adoption_audit.py -q
git diff --check -- docs/audits/2026-07-14-progenax-jaxstro-adoption-audit.md
git -C ../progenax status --porcelain --untracked-files=no
git add docs/audits/2026-07-14-progenax-jaxstro-adoption-audit.md
git commit -m "docs: audit progenax experimental ownership"
```

Expected: tests pass and Progenax remains unchanged.

---

### Task 6: Audit scripts, workflows, and documentation dependencies

**Files:**
- Modify: `docs/audits/2026-07-14-progenax-jaxstro-adoption-audit.md`
- Modify, maintainer-local: `../progenax/docs/plans/2026-07-14-jaxstro-adoption-audit-and-refactor.md`
- Read: `../progenax/scripts/**`
- Read: `../progenax/docs/website/**`
- Read: `../progenax/docs/methods/**`
- Read: `../progenax/docs/dev-methods-guides/**`
- Read: `../progenax/pyproject.toml`
- Read: `../progenax/scripts/check.sh`
- Read: `../progenax/scripts/release_gate.sh`

**Interfaces:**
- Consumes: source ownership findings.
- Produces: workflow-consumer evidence, affected documentation paths, and exact future validation gates.

- [ ] **Step 1: Inventory script-level use and duplicated mechanics**

```bash
rg -n "^(from jaxstro|import jaxstro)|jax[.]|jnp[.]|random[.]|linalg[.]|searchsorted|interp|trapz|quadrature|root|solve|cdf|ppf" ../progenax/scripts
```

Classify results as package consumer, validation oracle, demonstration-only implementation, profiler, generated-artifact owner, or duplicated generic machinery.

- [ ] **Step 2: Trace public claims and examples**

```bash
rg -n "jaxstro|progenax[.]numerics|cumulative_trapz|inverse_cdf_draw|newton_ppf|gauss_hermite|STELLAR|DEFAULT_UNITS" \
  ../progenax/docs/website ../progenax/docs/methods ../progenax/docs/dev-methods-guides
```

Record pages defining public behavior or teaching a compatibility route. Documentation strengthens a contract finding but does not prove runtime equivalence.

- [ ] **Step 3: Record repository-owned validation commands**

Read `scripts/check.sh`, `scripts/release_gate.sh`, and `pyproject.toml`. Record focused, fast, experimental, lint, type, coverage, and release commands in the validation matrix. Do not run the heavy release gate.

- [ ] **Step 4: Complete workflow coverage, verify, and commit**

Create material findings and `reviewed, no finding` entries, then run:

```bash
uv run pytest tests/integration/test_progenax_adoption_audit.py -q
git diff --check -- docs/audits/2026-07-14-progenax-jaxstro-adoption-audit.md
git -C ../progenax status --porcelain --untracked-files=no
git add docs/audits/2026-07-14-progenax-jaxstro-adoption-audit.md
git commit -m "docs: audit progenax workflow dependencies"
```

Expected: tests pass and Progenax remains unchanged.

---

### Task 7: Map findings to tests and identify stale-test candidates

**Files:**
- Modify: `docs/audits/2026-07-14-progenax-jaxstro-adoption-audit.md`
- Modify, maintainer-local: `../progenax/docs/plans/2026-07-14-jaxstro-adoption-audit-and-refactor.md`
- Read: `../progenax/tests/unit/**`
- Read: `../progenax/tests/integration/**`
- Read: `../progenax/tests/validation/**`
- Read: `../progenax/tests/experimental/**`
- Read: `../progenax/tests/demos/**`
- Read: `../progenax/scripts/build_test_dashboard.py`
- Read: all Progenax validation registries and helpers

**Interfaces:**
- Consumes: all findings from Tasks 2 through 6.
- Produces: a complete test-disposition matrix and safe cleanup prerequisites for later refactor plans.

- [ ] **Step 1: Inventory test files and helpers by family**

```bash
find ../progenax/tests -type f -name 'test_*.py' -print | sort
find ../progenax/tests -type f -name '*.py' -print | sort
```

Record counts for unit, integration, validation, experimental, and demo families. Account for fixtures and helpers that do not start with `test_`.

- [ ] **Step 2: Trace each finding to exact tests**

For every surface symbol and private helper named by a finding, run:

```bash
rg -n "literal_symbol_name|literal_private_helper_name" ../progenax/tests ../progenax/scripts
```

Record whether each test protects a scientific invariant, numerical value, units, public import, error behavior, JAX transformation, implementation detail, artifact, or obsolete private route.

- [ ] **Step 3: Apply the stale-test rule**

Assign:

- `KEEP` for surviving scientific, numerical, units, public API, error, or transformation behavior.
- `REWRITE` when behavior remains but the assertion is coupled to a superseded private implementation.
- `REPLACE` when a stronger owner-level contract must coexist until mutation evidence proves it.
- `STALE_CANDIDATE` only when behavior disappears with the superseded private path and no surviving contract depends on it.

For every non-`KEEP` row, record the exact replacement test file, retained contract, mutation or deliberate-break probe, focused command, affected-suite command, and deletion preconditions. If evidence is incomplete, use `KEEP` and record the uncertainty.

- [ ] **Step 4: Complete the cleanup matrix**

Use:

```markdown
| Finding ID | Test path | Current contract | Disposition | Replacement or rewrite gate | Mutation gate | Cleanup allowed |
| --- | --- | --- | --- | --- | --- | --- |
```

Every cleanup cell is `false`. State that no tracked Progenax test changed.

- [ ] **Step 5: Prove no Progenax change and commit**

```bash
git -C ../progenax status --porcelain --untracked-files=no
git -C ../progenax diff --name-only -- tests src scripts docs/website pyproject.toml
uv run pytest tests/integration/test_progenax_adoption_audit.py -q
git diff --check -- docs/audits/2026-07-14-progenax-jaxstro-adoption-audit.md
git add docs/audits/2026-07-14-progenax-jaxstro-adoption-audit.md
git commit -m "docs: classify progenax test migration debt"
```

Expected: both Progenax commands print nothing and Jaxstro tests pass.

---

### Task 8: Synthesize recommendations and ratchet completeness

**Files:**
- Modify: `docs/audits/2026-07-14-progenax-jaxstro-adoption-audit.md`
- Modify: `tests/integration/test_progenax_adoption_audit.py`
- Modify, maintainer-local: `../progenax/docs/plans/2026-07-14-jaxstro-adoption-audit-and-refactor.md`

**Interfaces:**
- Consumes: complete coverage, findings, and test dispositions.
- Produces: a review-ready report, three recommendation queues, validation gates, and automated completeness checks.

- [ ] **Step 1: Build three separate recommendation queues**

Populate:

1. `Recommended next slices`: only `ADOPT_READY`, ordered by scientific/API risk, ecosystem leverage, validation strength, and dependency order.
2. `Jaxstro hardening backlog`: `JAXSTRO_HARDEN_FIRST` with missing evidence.
3. `Research ownership questions`: `PROPOSE_FOR_JAXSTRO` and `DEFER_EXPERIMENTAL`, explicitly non-actionable.

For each recommended slice, list finding IDs, one-repository boundary, pre- and post-migration tests, public API decision, and separate cleanup gate.

- [ ] **Step 2: Complete validation and decision records**

For each `ADOPT_READY` finding, record exact numerical-equivalence, units, gradient, `jit`, `vmap`, public-import, focused-test, affected-test, Jaxstro-contract, consumer-search, and cleanup-review commands.

Record every finding as `OBSERVED` with cleanup `false`. The report does not advance any finding to `APPROVED`.

- [ ] **Step 3: Mark the report review-ready**

Change every coverage row to `reviewed` or `reviewed, no finding`. Change frontmatter to `status: review`. Remove incomplete classifications and permission-like language.

- [ ] **Step 4: Add final completeness tests**

Append to `tests/integration/test_progenax_adoption_audit.py`:

```python
FINDING_FIELDS = (
    "Surface",
    "Responsibility",
    "Current Jaxstro use",
    "Proposed owner",
    "Disposition",
    "Evidence",
    "Transform contract",
    "Units contract",
    "Test disposition",
    "Risk",
    "Required gates",
    "Decision",
    "Cleanup allowed",
)


def _finding_blocks(text: str) -> tuple[tuple[str, str], ...]:
    matches = tuple(
        re.finditer(
            r"^### (?P<finding_id>PXJ-\d{3}):.*?$\n"
            r"(?P<body>.*?)(?=^### PXJ-\d{3}:|^## |\Z)",
            text,
            re.MULTILINE | re.DOTALL,
        )
    )
    return tuple(
        (match.group("finding_id"), match.group("body")) for match in matches
    )


def test_every_audit_finding_has_complete_evidence_fields() -> None:
    text = _report_text()
    findings = _finding_blocks(text)

    assert findings
    assert len({finding_id for finding_id, _ in findings}) == len(findings)
    for finding_id, body in findings:
        for field in FINDING_FIELDS:
            assert f"- **{field}:**" in body, (finding_id, field)
        assert "- **Decision:** `OBSERVED`" in body, finding_id
        assert "- **Cleanup allowed:** `false`" in body, finding_id


def test_review_audit_has_no_incomplete_coverage_or_cleanup() -> None:
    text = _report_text()

    assert "status: review" in text or "status: approved" in text
    assert "| not-started |" not in text
    assert "| in-progress |" not in text
    assert "Cleanup allowed:** `true`" not in text
    assert "| true |" not in text
    assert "reviewed, no finding" in text


def test_only_adopt_ready_findings_enter_recommended_next_slices() -> None:
    text = _report_text()
    section = text.split("### Recommended next slices", 1)[1].split(
        "### Jaxstro hardening backlog", 1
    )[0]

    assert "ADOPT_READY" in section
    for forbidden in (
        "JAXSTRO_HARDEN_FIRST",
        "PROPOSE_FOR_JAXSTRO",
        "KEEP_IN_PROGENAX",
        "DEFER_EXPERIMENTAL",
        "REJECT",
    ):
        assert forbidden not in section
```

- [ ] **Step 5: Run final report gates**

```bash
uv run pytest tests/integration/test_progenax_adoption_audit.py -q
uv run ruff check tests/integration/test_progenax_adoption_audit.py
uv run ruff format --check tests/integration/test_progenax_adoption_audit.py
git diff --check
```

Expected: all audit tests pass, Ruff is clean, and whitespace checks are silent. Correct the report or contract rather than weakening a failed assertion.

- [ ] **Step 6: Complete the Progenax ledger**

Set every layer to `reviewed`. Record all finding IDs, commands, deviations, and no-tracked-change evidence. Set `Review decisions` to `Awaiting Anna's report review.`

- [ ] **Step 7: Commit the review-ready report**

```bash
git add docs/audits/2026-07-14-progenax-jaxstro-adoption-audit.md tests/integration/test_progenax_adoption_audit.py
git commit -m "docs: finalize progenax adoption audit"
```

---

### Task 9: Run audit gates and stop for Anna's review

**Files:**
- Read: `docs/audits/2026-07-14-progenax-jaxstro-adoption-audit.md`
- Read: `tests/integration/test_progenax_adoption_audit.py`
- Read: `docs/superpowers/specs/2026-07-14-progenax-jaxstro-adoption-audit-design.md`
- Read, maintainer-local: `../progenax/docs/plans/2026-07-14-jaxstro-adoption-audit-and-refactor.md`

**Interfaces:**
- Consumes: the review-ready report.
- Produces: independent structural and read-only proof, followed by a mandatory user-review pause.

- [ ] **Step 1: Run focused Jaxstro gates**

```bash
uv run pytest \
  tests/integration/test_progenax_adoption_audit.py \
  tests/integration/test_future_capabilities_roadmap.py \
  tests/integration/test_sota_assessment.py \
  -q
uv run ruff check tests/integration/test_progenax_adoption_audit.py
uv run ruff format --check tests/integration/test_progenax_adoption_audit.py
```

Expected: all selected tests pass and Ruff is clean.

- [ ] **Step 2: Prove Progenax remained read-only**

```bash
git -C ../progenax status --porcelain --untracked-files=no
git -C ../progenax diff --name-only -- src tests scripts docs/website pyproject.toml uv.lock
git -C ../progenax check-ignore -v docs/plans/2026-07-14-jaxstro-adoption-audit-and-refactor.md
```

Expected: the first two commands print nothing and the third identifies the ignore rule.

- [ ] **Step 3: Self-review every claim**

Check every family, finding citation, contract classification, ownership boundary, units rule, transform claim, queue placement, test disposition, mutation gate, and cleanup flag. Verify Startrax and Gravax appear only as protected exclusions.

Correct factual defects, rerun Steps 1 and 2, and commit corrections only when needed:

```bash
git add docs/audits/2026-07-14-progenax-jaxstro-adoption-audit.md tests/integration/test_progenax_adoption_audit.py
git commit -m "docs: correct progenax audit review findings"
```

- [ ] **Step 4: Stop for report approval**

Present the report path, Jaxstro commit, audited Progenax commit, gate results, three recommendation queues, stale-test candidate count, and explicit no-Progenax-change statement. Do not update either `STATUS.md`, advance findings, or start migration planning before Anna responds.

---

## Mandatory approval gate

Task 10 begins only after Anna explicitly approves the completed report. Requested changes return to Task 9 and its gates.

---

### Task 10: Record approval and close status consistently

**Files:**
- Modify: `docs/audits/2026-07-14-progenax-jaxstro-adoption-audit.md`
- Modify: `STATUS.md`
- Modify, maintainer-local: `../progenax/docs/plans/2026-07-14-jaxstro-adoption-audit-and-refactor.md`
- Modify: `../progenax/STATUS.md`

**Interfaces:**
- Consumes: Anna's explicit report approval and green Task 9 evidence.
- Produces: an approved report, completed local ledger, and one final status record in each repository without changing active priorities.

- [ ] **Step 1: Mark report and ledger approved**

Change report frontmatter from `status: review` to `status: approved`. Record Anna's report approval in the decision register without advancing individual findings beyond `OBSERVED`. Migration findings still require separate approval.

Replace `Awaiting Anna's report review.` in the Progenax ledger with the literal approval decision.

- [ ] **Step 2: Add exact completion records**

In Jaxstro `STATUS.md`, add immediately after the current top-level `next:` paragraph:

```markdown
previous: Progenax-Jaxstro adoption audit completed and approved. The authoritative report records current Jaxstro use, adoption-ready candidates, Jaxstro-hardening prerequisites, future ownership proposals, protected Progenax science, experimental boundaries, test dispositions, and cleanup gates. No runtime API, dependency, public documentation, publication, or sibling migration changed; each migration remains separately approval-gated.
```

In Progenax `STATUS.md`, add immediately after the current top-level `next:` paragraph:

```markdown
previous: Progenax-Jaxstro adoption audit completed and approved. The authoritative Jaxstro report records current adoption, future ownership proposals, protected Progenax science, stale-test candidates, and per-slice validation gates. This audit changed no Progenax runtime source, public API, tracked tests, dependencies, product documentation, or scientific behavior; migrations and stale-test cleanup remain separately approval-gated.
```

Preserve unrelated `next:`, `blocker:`, and `due:` text.

- [ ] **Step 3: Verify and commit Jaxstro closeout**

```bash
uv run pytest tests/integration/test_progenax_adoption_audit.py -q
git diff --check -- docs/audits/2026-07-14-progenax-jaxstro-adoption-audit.md STATUS.md
git add docs/audits/2026-07-14-progenax-jaxstro-adoption-audit.md STATUS.md
git commit -m "docs: approve progenax adoption audit"
```

Record the resulting Jaxstro commit in the local Progenax ledger.

- [ ] **Step 4: Verify and commit Progenax status separately**

```bash
git -C ../progenax diff --check -- STATUS.md
git -C ../progenax diff --name-only
```

Expected: the only tracked Progenax path printed is `STATUS.md`.

```bash
git -C ../progenax add STATUS.md
git -C ../progenax commit -m "docs: record jaxstro adoption audit"
```

- [ ] **Step 5: Prove final repository state**

```bash
git status --short --branch
git log -1 --oneline
git -C ../progenax status --short --branch
git -C ../progenax log -1 --oneline
git -C ../progenax check-ignore -v docs/plans/2026-07-14-jaxstro-adoption-audit-and-refactor.md
```

Expected: Jaxstro shows only pre-existing unrelated untracked state, Progenax has no tracked changes, the ledger remains ignored, and each repository has its own closeout commit.

Stop. A future implementation plan may cover the first explicitly approved `ADOPT_READY` slice. That plan must include its own Progenax test rewrite and stale-test cleanup proof; this audit plan authorizes neither.
