# jaxstro public-hardening design

**Status:** Anna-approved 2026-07-11; implementation remains approval-gated per slice.

## Goal

Bring public `jaxstro` to a trustworthy, well-explained, release-ready state:
source-verified core infrastructure, enforced gradient and provenance contracts, current
pedagogical documentation, an efficient public CI gate, and a GitHub Pages deployment.

## Grounded starting point

- `main` is `05baf32`; the historical `codex/quantity-implementation` branch is absent,
  while quantity commit `655e756` is an ancestor of `main`.
- `STATUS.md` is stale: it still calls quantity in-progress and still names Actions-minute
  exhaustion as the reason for dormant CI.
- The prescribed baseline command, `bash scripts/check.sh`, currently stops at Ruff with
  seven style errors in `tests/unit/test_spatial.py`; it has not yet reached the test-count
  stage. This is developer-quality evidence, not a scientific-validation failure.
- jaxstro already ships the shared AD-vs-FD engine in `jaxstro.testing.grad_audit`; the
  hardening work must extend jaxstro's case registry rather than port another copy.
- The MyST project is rooted at `docs/`, with built output in `docs/_build/site`; it is not
  progenax's `docs/website/` layout. Quantity documentation exists but must be executable
  and behavior-checked. Spatial has API coverage but no dedicated learner-facing chapter.
- Standard GitHub-hosted runners are free for public repositories. This makes efficient
  PR CI practical; larger runners remain a possible cost.

## Chosen program order

```text
A0 baseline repair
  -> A audit and hardening
  -> D0 efficient CI
  -> B provenance-registry hoist
  -> C docs reconciliation and page-by-page pedagogy
  -> D Pages deployment and release checklist
       -> stop before PyPI until the package name is chosen
```

Every arrow is an explicit checkpoint. Work does not advance automatically, and Anna
approves every slice before implementation.

## Slice contracts

### A0 — baseline repair

Correct only the seven Ruff violations in `tests/unit/test_spatial.py`; do not change test
meaning or library behavior. Re-run `bash scripts/check.sh` and record its complete live
gate result, including test partition/count and wheel smoke. Reconcile `STATUS.md` from git
evidence in this slice.

### A — adversarial audit and code hardening

Maintain a findings ledger separating source-verified constants/transforms, numerical edge
behavior, AD-vs-FD evidence, and API/docstring mismatches. Verify scientific or convention
claims against primary sources or rendered PDFs, never recollection. Every confirmed defect
receives a regression test before its fix. Expand jaxstro's registry of public-entry-point
gradient cases and boundary probes using the existing shared audit engine. Treat audit
evidence as implementation evidence, not as a substitute for physical validation.

### D0 — efficient public CI

Enable a cancellation-aware, required pull-request gate with lock verification, Ruff, mypy,
a current-Python fast test tier, the focused gradient gate added in A, and wheel smoke. Keep a
scheduled/manual full gate for the Python 3.11–3.13 matrix, ML integration, full validation,
and (after C) the docs gate. Use a single aggregator check so a failed or skipped dependency
cannot appear green to branch protection. Keep `scripts/check.sh` as the local release mirror.

### B — provenance-registry hoist

Move package-independent model-card schema validation and deterministic rendering into
`jaxstro.testing`; retain YAML parsing in repository build tooling so the core runtime
dependency set remains unchanged. Card jaxstro's constants, unit conventions, coordinate and
astrometric transforms, and atmosphere-model boundaries. Commit generated MyST reference pages
and enforce regeneration, required fields, source locators, code references, and validation
links. Progenax migration is a later, out-of-session consumer migration.

### C — documentation currency and pedagogy

Start with a drift inventory and executable verification: current snippets, README examples,
links, MyST content warnings, `myst.xref.json`, and rendered DOM behavior. Update status and
changelog from the evidence. Then use individual HITL approvals per page before revising
narrative, structured API cards, examples, or figures. Build a deterministic `laboratory/`
figure-spec registry and CLI: generated PDF/PNG are ignored, optimized WebP is embedded, and
plotted invariants are tested. Use native MyST cross-references; do not attempt raw-HTML
new-tab links or infer URLs from directory structure.

### D — Pages and release readiness

Adapt progenax's known-good Pages workflow to jaxstro's actual docs root and output path. The
workflow runs the same docs gate before publishing at
`https://jaxstro.github.io/jaxstro/` with the repository-name base URL. The Pages source setting,
pushes, and deployment are explicit remote-action approvals. The release checklist covers
CITATION.cff, Zenodo, sdist inspection, CONTRIBUTING guidance, and dependency floors. It stops
before a PyPI upload until Anna settles the permanent distribution name; progenax's PyPI path
remains downstream-gated on this decision and jaxstro's release. Run public-hygiene scans early
and again before release.

## Global invariants

- No primary-source claims from memory.
- Never weaken tests or numerical tolerances to obtain green results.
- Verify MyST claims in rendered output; use generated xref data for slugs.
- Keep code, test, documentation, and provenance claims distinct in reports.
- Commit coherent, verified units only; merge to local `main` and push only on Anna's go.
