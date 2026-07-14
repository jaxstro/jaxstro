---
title: Progenax-Jaxstro adoption audit design
date: 2026-07-14
status: approved
---

# Progenax-Jaxstro adoption audit design

## Purpose

Jaxstro needs a trustworthy account of where Progenax already uses Jaxstro,
where supported Jaxstro APIs should replace local shared machinery, which
Progenax-local primitives may become future Jaxstro owners, and which scientific
behavior must remain in Progenax. The audit must expose high-leverage next
slices without treating architectural similarity as permission to migrate code.

The audit is read-only. It produces an evidence-backed report and checklist; it
does not authorize implementation, deletion, publication, or sibling-package
migration. Each executable migration requires a later, separately approved
implementation plan.

## Safety boundary

- Startrax and Gravax are protected repositories. The audit does not inspect,
  edit, migrate, test, or use them as rollout gates while that hold remains.
- Progenax source, tests, tracked product documentation, configuration, and
  public API are read-only during the audit.
- The only Progenax write made when the audit begins is a dedicated
  maintainer-local audit/refactor record under `docs/plans/`.
- Progenax `STATUS.md` is not an audit diary. It is updated once after the
  approved audit or refactor work is complete, using its existing `next:`,
  `blocker:`, and `due:` format.
- No superseded Progenax path is deleted until its replacement is independently
  verified and Anna explicitly approves cleanup.
- Temporary coexistence may be used inside a later migration slice for
  verification. It must not become a permanent compatibility layer.
- The audit records the exact Jaxstro and Progenax branches, commits, and dirty
  worktree states used for its evidence. A changed snapshot makes affected
  findings stale until rechecked.

## Document ownership

### Authoritative Jaxstro report

The cross-package ownership report and checklist live at:

`docs/audits/2026-07-14-progenax-jaxstro-adoption-audit.md`

This is the single source of truth for findings, ownership dispositions,
readiness, and recommended ordering. It may link to the Jaxstro roadmap after
the findings are reviewed, but the roadmap does not duplicate the checklist.

### Maintainer-local Progenax record

When the audit begins, Progenax receives:

`docs/plans/2026-07-14-jaxstro-adoption-audit-and-refactor.md`

If work begins on a later date, the filename uses that actual start date. This
record contains the Progenax snapshot, local evidence, approved finding IDs,
commands, results, deviations, and cleanup decisions. It links to the Jaxstro
report instead of copying cross-package conclusions. Progenax already treats
`docs/plans/` as maintainer-local working material.

### Status synchronization

Neither repository status file is used for finding-by-finding progress. The
final Progenax `STATUS.md` update records only the completed repository state,
remaining blocker if any, and next real action. A Jaxstro status update is made
only when the audit becomes or completes a repository-level next action; it
must preserve unrelated active work.

## Audit coverage

The audit covers the whole Progenax repository in five ordered layers. Every
source, script, test, and documentation family receives either one or more
findings or an explicit `reviewed, no finding` entry so omissions remain
visible.

### Layer 1: released scientific core

Review released profiles and distribution functions, initial mass functions,
binary populations and orbital mechanics, cluster and population builders,
analytical models, dynamics and tidal machinery, diagnostics, and public
protocols.

The main question is where Progenax-owned scientific policy should call stable
Jaxstro machinery. A generic implementation beneath a model does not transfer
ownership of the model, its parameters, or its scientific acceptance rules.

### Layer 2: internal shared mechanics

Review integration, quadrature, root finding, inverse-CDF sampling,
interpolation, coordinates, spatial utilities, random sampling, array
validation, numerical guards, configuration, units, and precision setup. This
layer is the primary source of low-risk adoption candidates because Jaxstro may
already provide the generic owner.

### Layer 3: experimental research code

Review experimental field representations, covariance and Fisher calculations,
likelihoods, priors, HMC, neural posterior estimation, simulation-based
calibration, inference diagnostics, projection, Gaussianization, and
experimental validation utilities.

Every finding distinguishes among generic Jaxstro numerical machinery,
Informax inference policy, Progenax scientific modeling, and experimental code
that should remain local. Nothing is promoted merely because it is reusable.

### Layer 4: scripts and workflows

Review direct Jaxstro use outside library code, duplicated script utilities,
validation scripts coupled to local implementations, affected generated
artifacts, and import patterns that should remain examples rather than public
APIs.

### Layer 5: tests, documentation, and dependency evidence

Use tests and documentation to identify load-bearing behavior, public imports,
scientific tolerances, derivative expectations, examples that would change,
and contracts that are tested but not documented. Dependency declarations and
runtime configuration are part of this evidence layer.

## Finding model

Each finding records:

| Field | Requirement |
| --- | --- |
| Finding ID | Stable, grep-friendly identifier |
| Progenax surface | Exact files, modules, and symbols |
| Responsibility | Scientific or numerical job performed |
| Current Jaxstro use | Imports, wrappers, or absence of use |
| Proposed owner | Jaxstro, Progenax, Informax, or unchanged |
| Disposition | One readiness band defined below |
| Evidence | Source, tests, documentation, and consumers |
| Transform contract | `jit`, `vmap`, differentiation, shapes, and static data |
| Units contract | Explicit `G`, explicit units, or wrapper default |
| Test disposition | Keep, rewrite, replace, or stale-candidate with evidence |
| Risk | Scientific, API, numerical, and dependency risks |
| Required gates | Exact evidence needed before and after migration |
| Decision state | Observed through cleanup, with approval recorded |
| Cleanup permission | `false` until separately approved |

Findings are organized into six ownership lanes:

1. Existing Jaxstro use that is already canonical.
2. Safe candidates for an existing supported Jaxstro API.
3. Missing shared primitives that may become future Jaxstro owners.
4. Scientific behavior that must remain Progenax-owned.
5. Experimental or cross-package ownership questions.
6. Units and quantity decisions.

## Readiness dispositions

Every finding receives exactly one disposition:

- `ADOPT_READY`: an existing supported Jaxstro API matches the Progenax need.
- `JAXSTRO_HARDEN_FIRST`: an API exists but lacks sufficient contract,
  evidence, or compatible behavior.
- `PROPOSE_FOR_JAXSTRO`: a generic primitive may belong in Jaxstro but does not
  yet have a supported owner there.
- `KEEP_IN_PROGENAX`: the behavior is Progenax scientific policy or has no
  justified shared owner.
- `DEFER_EXPERIMENTAL`: ownership remains uncertain or depends on experimental
  research code.
- `REJECT`: moving the behavior would create an incorrect or harmful boundary.

Only `ADOPT_READY` findings can enter a migration plan. Future ownership
candidates remain explicitly non-actionable even when the report recommends
later design work.

## Recommendation order

The report does not collapse uncertain scientific judgments into a numeric
score. It ranks `ADOPT_READY` findings using visible qualitative judgments:

1. Scientific and API risk.
2. Ecosystem leverage and removed duplication.
3. Strength of existing validation evidence.
4. Dependency order among candidates.

The highest-leverage next slice has strong evidence, an existing supported
Jaxstro owner, low scientific risk, substantial reuse, and no public break.
The conclusion presents three separate queues:

- recommended next slices;
- Jaxstro hardening work required before adoption;
- future research-ownership questions.

Protected scientific ownership and experimental deferrals never enter the
executable queue.

## Units and quantity boundary

Progenax continues to use the stable `jaxstro.units` contract, including its
domain default and explicit `G` or units requirements. The audit may record
future opportunities for `jaxstro.quantity`, but that package remains an
opt-in alpha surface and no Progenax migration to it is authorized.
`jaxstro.quantity` findings use `PROPOSE_FOR_JAXSTRO` or
`DEFER_EXPERIMENTAL`, not `ADOPT_READY`, unless a later design explicitly
changes the maturity and adoption decision.

## Migration lifecycle

An eventual migration follows this state machine:

```text
OBSERVED
  -> CANDIDATE
  -> APPROVED
  -> IMPLEMENTED
  -> VERIFIED
  -> CLEANUP_ALLOWED
```

`PROPOSAL_ONLY`, `DEFERRED`, `PROTECTED`, and `REJECTED` cannot advance without
a new decision from Anna.

### Candidate evidence

Before a finding can be `ADOPT_READY`, the report must establish:

- equivalent scientific meaning;
- compatible units and explicit `G` behavior;
- compatible shapes, dtypes, precision, broadcasting, and boundary behavior;
- compatible `jit`, `vmap`, and differentiation semantics;
- compatible error behavior;
- a supported and classified Jaxstro contract; and
- no accidental crossing between released and experimental ownership.

### Verification before cleanup

After an approved migration but before deletion:

- focused equivalence tests pass;
- derivative checks pass where differentiation is supported;
- JIT and vectorization checks pass;
- public Progenax imports remain valid;
- the relevant Progenax suite passes;
- Jaxstro contract tests pass;
- repository-wide searches find no unintended consumers; and
- scientific tolerances and numerical differences are recorded.

Cleanup additionally requires recorded evidence, no remaining consumers, and
Anna's explicit approval.

### Stale-test cleanup

The audit identifies Progenax tests that are coupled only to a superseded local
implementation, but it does not delete or rewrite them. Each affected test is
classified as `KEEP`, `REWRITE`, `REPLACE`, or `STALE_CANDIDATE` with the
scientific or API behavior it protects.

A later approved refactor slice may delete a `STALE_CANDIDATE` only after:

- replacement tests protect the same scientific, numerical, units, public API,
  and transformation contracts that remain valid;
- a mutation or deliberate break proves the replacement tests fail when the
  retained contract is violated;
- focused and affected Progenax tests pass after the migration;
- repository-wide searches confirm the old implementation and its private test
  hooks have no remaining consumers; and
- Anna explicitly approves the cleanup.

A failing test is never removed merely to make a gate pass. Regression tests
for still-supported public or scientific behavior remain, even when their
implementation changes. Obsolete fixtures, helpers, snapshots, and
implementation-detail assertions are removed in the same approved cleanup
commit as the superseded path so the suite describes the surviving system.

## Stop conditions

The audit or later migration stops without cleanup if:

- units, defaults, or explicit `G` semantics differ;
- numerical differences exceed an established tolerance;
- gradients become unavailable or materially change;
- JIT, vectorization, dtype, shape, error, or boundary behavior changes;
- a public Progenax API would break;
- the audited snapshot or its assumptions become stale;
- a generic-looking primitive contains Progenax scientific policy;
- a future proposal is mistaken for an existing supported API;
- released and experimental ownership are mixed without a separate decision;
- an unreviewed consumer is discovered; or
- a proposed stale-test deletion lacks replacement contract coverage and a
  mutation check; or
- validation evidence is incomplete or ambiguous.

## Report architecture

The Jaxstro report contains:

1. Purpose and safety contract.
2. Audited repository snapshots and exclusions.
3. Coverage inventory.
4. Progenax responsibility and dependency map.
5. Existing canonical Jaxstro use.
6. `ADOPT_READY` and `JAXSTRO_HARDEN_FIRST` findings.
7. Future Jaxstro ownership proposals.
8. Progenax-owned no-move list.
9. Experimental and Informax-boundary appendix.
10. Units and quantity boundary.
11. Prioritized recommendation queues.
12. Validation matrix and migration checklist.
13. Test-disposition and stale-test cleanup checklist.
14. Decision register.

The report is usable as both a narrative assessment and a symbol-level
checklist. Its tables use stable finding IDs so the Progenax record and future
implementation plans can reference decisions without copying them.

## Validation of the audit itself

The audit is complete only when:

- every in-scope family is accounted for;
- every recommendation cites exact source and test evidence;
- every existing Jaxstro owner is checked against the current contract
  registry rather than inferred from a public import alone;
- each `ADOPT_READY` finding has explicit pre-migration and post-migration
  gates;
- every affected Progenax test is classified as `KEEP`, `REWRITE`, `REPLACE`,
  or `STALE_CANDIDATE` without deleting it during the audit;
- current adoption and future ownership proposals are in separate queues;
- released, experimental, and inference-policy ownership are not conflated;
- Startrax and Gravax remain uninspected and unchanged;
- the Progenax working tree has no audit-induced changes beyond the dedicated
  maintainer-local record;
- the report contains no placeholders, unresolved classifications, or implied
  implementation authorization; and
- Anna reviews and approves the report before implementation planning begins.

## Non-goals

This design does not authorize:

- changes to Startrax or Gravax;
- Progenax runtime, tests, public documentation, or dependencies;
- Jaxstro runtime modules or public API changes;
- adoption of `jaxstro.quantity` by Progenax;
- promotion of experimental code;
- permanent compatibility shims;
- deletion of superseded paths;
- publication, push, release, or package deployment; or
- implementation of any finding before a separate approved plan.

## Transition after written-spec approval

After Anna reviews this written specification, the next artifact is a detailed
read-only audit execution plan. That plan creates the empty report structure,
records repository snapshots, inventories Progenax layer by layer, checks each
candidate against Jaxstro contracts, writes the evidence-backed findings, and
stops at report review. Migration planning begins only after that report is
approved.
