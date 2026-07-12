# jaxstro public-hardening program — session kickoff (2026-07-11)

**Mission:** bring `jaxstro` (the ecosystem-core package; repo `jaxstro/jaxstro`, ALREADY public)
to the same shipped standard progenax just reached: audited + hardened code, everything correct
and well-explained, a current well-designed docs site, and the site LIVE on GitHub Pages.

**Mode of work (non-negotiable):** Anna is the human-in-the-loop. START WITH THE BRAINSTORMING
SKILL — do not write code first. Propose the slice plan below as a starting point, refine it with
her one question at a time, get explicit approval for the plan and then per-slice. Verify claims
against sources/PDFs/code — never from memory. Never weaken tests to pass. Commit frequently;
merge to local main per slice on Anna's go; push when she says push (repo is public — every push
is world-visible immediately).

---

## Where jaxstro stands today (verified 2026-07-11 — re-verify at session start)

- `main` HEAD `05baf32` (spatial exact O(N) neighbour-gather merge). Working tree was clean.
- **STATUS.md is STALE**: says `jaxstro.quantity` is in progress on `codex/quantity-implementation`
  — but quantity is in `src/jaxstro/` and spatial has merged since. Reconcile first (check that
  branch for unmerged work before believing anything).
- Modules: astrometry, atmospheres, constants, coords, geometry, jaxconfig, numerics, params,
  provenance, quantity, spatial, testing (incl. the ratchet harness, ADR-0021/0024), units.
- Phase C (2026-06-17) delivered: 3-tier tests (447 then; run for the live count), dormant CI
  (`tests.yml`, workflow_dispatch only — Actions minutes were exhausted), `scripts/check.sh` as
  the real local gate (lock-check, ruff, mypy, pytest, ml-integration, wheel-smoke), a 25-page
  Diátaxis MyST site (`docs/website/`: getting-started / theory / architecture / 30-decisions
  ADRs / api / howto / validation / dev-log / release) with **honest stubs**, CHANGELOG, CLAUDE.md.
- Phase D was left pending: tag/version decision, release-staging naming (jaxstro vs
  jaxstro-core), CI trigger flip, brain roots registration, sibling floor bumps.
- The docs site PREDATES `quantity` and the new `spatial` primitives → almost certainly stale.

## Proposed slice plan (brainstorm this with Anna before executing)

**Slice A — audit + code hardening.**
Adversarial audit of the shipped core (subagent-driven, findings verified before fixing):
correctness of physics/math constants and transforms against primary sources (constants.py,
units.py, coords.py, astrometry.py — check every value's provenance), numerics stability edges,
gradient integrity (AD-vs-FD over public entry points — port progenax's grad-audit harness
pattern if absent), API-contract honesty (docstrings vs behavior), eager input validation.
TDD every fix. Quality over quantity on tests — delete low-leverage ones rather than accrete.

**Slice B — provenance registry hoist (ADR-0034 prove-then-hoist, the plan of record).**
The machine-readable model-card registry + generated glossary + enforcement suite is proven in
progenax (`docs/provenance/registry/*.yaml` → glossary + 5th registry;
`scripts/build_provenance_registry.py`, `tests/validation/provenance_cards/`). Hoist the TOOLING
into `jaxstro.testing` (or a sibling module) so both packages share it; then card jaxstro's own
constants/transforms (CODATA/IAU values, coordinate conventions, atmosphere models). progenax
migrates to the hoisted tooling as a follow-up, not in this session.

**Slice C — docs currency + design + pedagogy.**
(1) Reconcile the site with the code: quantity + spatial chapters missing; audit every existing
page's API snippets against src (run them); kill or fill the honest stubs; STATUS/CHANGELOG
refresh. (2) Pedagogy pass page-by-page WITH Anna's approval per page (progenax C2 pattern),
publication-quality figures via a modular figure library in `laboratory/` (port progenax's
ICViz architecture: FigureSpec registry + CLI, StarViz seaborn theme, PDF/PNG gitignored,
WebP embedded — figures double as correctness proofs). (3) Landing page + API-page structure
(progenax's structured cards + registry badges pattern). (4) Docs gate: port progenax's
3-part check (broken-.md-link scan + zero-content-warnings build) if jaxstro lacks one.

**Slice D — publish the site + release checklist.**
GitHub Pages deploy — the known-good recipe from progenax (live at
https://jaxstro.github.io/progenax/, workflow `.github/workflows/pages.yml`):
- Actions workflow, NOT branch deploy: checkout@v4, setup-node@v4 (node 20), pin mystmd to the
  locally-validated version, run the FULL docs gate (not bare `myst build`), upload
  `docs/website/_build/html` via upload-pages-artifact@v3 → deploy-pages@v4 with
  `permissions: {contents: read, pages: write, id-token: write}` + `environment: github-pages`.
- `env: BASE_URL: /${{ github.event.repository.name }}` → site at https://jaxstro.github.io/jaxstro/.
- One-time: `gh api repos/jaxstro/jaxstro/pages -X POST -f build_type=workflow` (source =
  GitHub Actions — the #1 silent failure).
- **CI-minutes note:** jaxstro's CI is dormant BECAUSE minutes were exhausted; the repo is now
  public and public repos get free Actions minutes — verify, then flipping triggers on may be
  free. Raise with Anna in brainstorming (it interacts with Phase D's pending CI decision).
- Then write jaxstro's release checklist (mirror progenax's `95-release/checklist.md` Slice-D
  section): CI trigger flip, CITATION.cff + Zenodo at tag, sdist excludes, CONTRIBUTING.md,
  PyPI — noting progenax's PyPI publication is GATED ON JAXSTRO'S, so jaxstro-on-PyPI is the
  ecosystem's critical path (the Phase-D naming decision — jaxstro vs jaxstro-core vs namespace
  — must be settled BEFORE the first upload; a PyPI name is forever).

## Hard-won lessons from the progenax program — apply from minute one

1. **Verify in the rendered DOM, not the built AST.** A MyST plugin emitting raw `html` nodes
   deployed as literal escaped `<a href=...>` text. mystmd 1.10.1 + myst-theme: html nodes are
   escaped at BOTH inline and block level; raw HTML in .md renders but target/rel are STRIPPED.
   → `target="_blank"` internal links are infeasible; use native cross-refs (hover previews).
2. **MyST serves ROOT-FLAT slugs** (`dir/page_name.md` → `/page-name`), deduped
   build-order-dependently (`/binaries` vs `/binaries-1`). Never compute page URLs by directory
   path; let MyST resolve links, and gate any assumption against the built `myst.xref.json`.
3. **No primary-source claims from memory.** Multiple fabricated equation/locator claims were
   caught only by reading the actual PDFs. jaxstro is constants/transforms-heavy: every CODATA/
   IAU/reference value gets a source check, not a recollection.
4. **README code blocks should be executed by a test** (progenax `test_readme_examples.py`
   pattern) — port it if jaxstro lacks one.
5. **Docs "trim" = consolidate drift, never cut for length** (ADR-0032). Pedagogy pass asks
   Anna per page.
6. **Estimate cost before expensive runs; jit/vmap discipline in validation scripts too.**
7. **Post-public hygiene sweep:** grep tracked files for `/Users/anna`, check nothing internal
   (audit reports, brain drafts, PDFs) is tracked — progenax needed a post-flip scrub.

## Session start ritual

1. `/brain-pack jaxstro` for the hub context pack; read `~/brain/AGENTS.md` conventions if doing
   cross-session work.
2. Reconcile STATUS.md vs git (incl. `codex/quantity-implementation` — merged or orphaned?).
3. Run the real gate baseline: `bash scripts/check.sh` — record the live test count.
4. THEN open brainstorming with Anna on the slice plan above.
