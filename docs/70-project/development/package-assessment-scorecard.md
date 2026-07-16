---
title: Jaxstro research-software assessment scorecard
description: Living grades, evidence, limiting deficiencies, and promotion gates for the package.
---

# Jaxstro research-software assessment scorecard

This scorecard assesses Jaxstro as scientific infrastructure and a public
research-software package. The companion [](./sota-assessment.md) ranks
investments; this page records current evidence and what would justify changing
an assessment.

**Assessment date: 2026-07-15; registry reconciliation: 2026-07-15.** Grades
describe repository evidence reviewed on that date and must be re-audited when
supporting contracts change.

## Evaluation criteria

- **A:** coherent, independently evidenced, and ready for reuse within the
  stated boundary.
- **B:** useful and substantially validated, with material gaps in uniformity,
  adoption, discoverability, or evidence depth.
- **C:** implemented but incomplete, unevenly validated, or difficult to use
  without maintainer knowledge.
- **D:** experimental or insufficiently evidenced for ordinary scientific use.
- **F:** absent, contradicted by evidence, or unsafe to claim.

## Current grades

| Dimension | Grade | Evidence | Evidence-backed rationale | Deficiency preventing the next grade | Promotion evidence required |
| --- | --- | --- | --- | --- | --- |
| Research workflow coverage | B+ | [](../../40-workflows/investigations/investigations.md) | Three executable investigations connect public APIs to contracts, evidence, limitations, and warranted claims | Coverage is narrow relative to public callable breadth | More question-driven workflows with independent evidence artifacts |
| Contract coverage | B+ | [](../../50-api/research-infrastructure/contracts.md) | Public modules and consequential callables have export-audited contracts | Many callables remain explicitly unclassified | Prioritized callable coverage with tested transform and failure claims |
| Evidence linkage | A- | [](../../60-validation/evidence-index.md) | Computational evidence, source provenance, and scientific policy remain distinct and freshness checked | Evidence depth is uneven by module | Method-appropriate artifacts for consequential contracts |
| Limitation coverage | B+ | [](../../40-workflows/reproducible-research/evidence-and-claim-boundaries.md) | Workflows and contracts record bounded non-claims | Limitation coverage is not generated for every public callable | A limitation-coverage ratchet tied to contract records |
| Architecture and ownership | A | [](../direction/architecture.md) | The thin foundation and one-way ecosystem boundary are explicit | Consumer evidence for future abstractions is incomplete | Pinned multi-consumer ownership evidence |
| Numerical correctness | A- | [](../../60-validation/validation.md) | Analytic cases, limits, round trips, convergence, adaptive-quadrature failure envelopes, and FD comparisons cover major kernels | Validation and performance artifacts remain uneven | Per-contract evidence coverage and missing-evidence ratchets |
| AD honesty | A | [](../../20-methods/change-constraints-evolution/rootfinding.md) | Smooth, blocked, zero, value-first, validation-only, and certified implicit paths are distinguished | The taxonomy is not uniformly generated per callable | Callable-level AD coverage with evidence links |
| JAX architecture | A- | [](../../20-methods/methods.md) | JIT, VMAP, scan, PyTree, JVP/VJP, and gradient behavior are tested on substantial surfaces | Transform and batching-cost behavior is hard to discover per callable | Generated transform matrix linked to executable evidence |
| Units and dimensional safety | B+ | [](../../30-representations/units-quantities/quantity-system.md) | `UnitSystem` is mature and `quantity` is implemented | Adoption remains deferred while two dimensional surfaces coexist | Downstream parity, performance, serialization, and migration evidence |
| Performance evidence | B+ | [](../../60-validation/evidence-index.md) | Rootfinding and spectra use units-explicit records with freshness checks | Compile, graph, memory, and cost coverage remains uneven | Method-appropriate performance records |
| Downstream usefulness | B+ | [](../direction/science-general-vision.md) | Sibling packages motivate selected shared foundations | Adoption claims are not generated from pinned revisions | Compatibility records and consumer validation |
| Maintenance readiness | B+ | [](../release/checklist.md) | Release gates, deterministic docs, Ruff, MyPy, and wheel smoke exist | Type strictness and manually duplicated truth remain uneven | Contract-driven docs and stricter typing evidence |

## Coverage by research workflow

The workflow registry measures executable connections rather than page volume.
Each investigation names public APIs, contract links, indexed evidence links,
validation targets, limitations, and a warranted claim. This structure does not
automatically validate any downstream scientific model.

The next useful additions are investigations that expose currently weak
contract, evidence, or limitation coverage. A large number of shallow examples
would not justify a higher grade.

## Grade-change policy

A grade changes only when the relevant contract, independent validation,
limitation statement, reproducible artifact, and consumer evidence are present.
Every change records its date, supporting evidence, and newly satisfied
criterion. A newly discovered defect can lower a grade immediately.

## Current hardening sequence

**Scientific contract registry: implemented.** Evidence depth remains uneven,
and an unclassified symbol is not treated as supported.

| Metric identity | Symbol | Value | Units |
| --- | --- | ---: | --- |
| Registered public modules | `N_module,contract` | 17 | modules |
| Callable-level contracts | `N_callable,contract` | 18 | callables |
| Explicitly unclassified public callables | `N_callable,unclassified` | 225 | callables |
| Module-inherited public record types | `N_symbol,inherited` | 157 | symbols |

**Unified evidence infrastructure: implemented.** Computational measurements,
source provenance, and scientific policy remain separate evidence classes.
Method-specific scientific thresholds remain method-owned.

| Metric identity | Symbol | Value | Units |
| --- | --- | ---: | --- |
| Indexed evidence artifacts | `N_artifact,evidence` | 5 | artifacts |
| Distinct evidence classes | `N_class,evidence` | 3 | evidence classes |

**Executable research workflow registry: implemented.** The schema-2 registry
connects investigations to contracts, evidence, validation targets, and
limitations.

| Metric identity | Symbol | Value | Units |
| --- | --- | ---: | --- |
| Executable research investigations | `N_workflow` | 3 | investigations |
| Callable contract links | `N_contract,workflow` | 7 | contract links |
| Indexed artifact links | `N_evidence,workflow` | 2 | indexed evidence links |

The next investment is broader workflow coverage with independent evidence and
explicit limitation records. Callable coverage and downstream adoption remain
separate promotion gates.
