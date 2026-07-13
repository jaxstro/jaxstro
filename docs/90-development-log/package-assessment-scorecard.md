---
title: Jaxstro package assessment scorecard
description: Living grades, evidence, limiting deficiencies, and promotion gates for the package and its curriculum.
---

# Jaxstro package assessment scorecard

This is the living assessment of Jaxstro as scientific infrastructure, a public
package, and curriculum material. The companion
[](./sota-assessment.md) ranks investments; this page records current grades and
what evidence is required to change them.

**Assessment date: 2026-07-12; registry reconciliation: 2026-07-12.** Grades
describe the repository evidence reviewed on that date and must be re-audited
when their supporting contracts change.

## Grading rubric

- **A:** unusually strong, coherent, independently evidenced, and ready to teach
  or reuse within its stated boundary.
- **B:** useful and substantially validated, with material gaps in uniformity,
  adoption, discoverability, or evidence depth.
- **C:** implemented but incomplete, unevenly validated, or difficult to use
  without maintainer knowledge.
- **D:** experimental or insufficiently evidenced for ordinary scientific use.
- **F:** absent, contradicted by evidence, or unsafe to claim.

Pluses and minuses identify position within a band. Grades describe current
evidence, not effort or ambition.

## Current grades

| Dimension | Grade | Evidence | Evidence-backed rationale | Deficiency preventing the next grade | Promotion evidence required |
| --- | --- | --- | --- | --- | --- |
| Internal differentiable-science foundation | A− | [](../60-validation/index.md) | Broad shared primitives, explicit ownership, and extensive executable validation | Evidence depth and downstream verification are not uniform | Generated contracts, uniform evidence classes, and pinned adoption records |
| Public scientific package | B+ | [](../40-api/index.md) | Cohesive surface, minimal core dependencies, typed failures, and a public site | Public reach and external adoption trail the implementation | Stable release, public compatibility evidence, and external use cases |
| Curriculum concept | B+ | [](../00-getting-started/how-to-learn.md) | Predict → compute → audit, objectives, misconceptions, and science-pattern routing form a coherent philosophy | The approved foundations sequence is not yet implemented | Reviewed foundations spine with tested scientific examples |
| Executable curriculum | B | [](../10-theory/index.md) | Tested examples exist and documentation is build-gated | Activities, instructor materials, diagnostics, and rubrics are incomplete | CI-executed investigations and instructor-facing assessment materials |
| Architecture and ownership | A | [](../40-api/contracts.md) | Thin-foundation ADRs, one-way ecosystem boundaries, and generated module contracts are explicit | Evidence depth remains uneven at callable level | Expand callable classification without weakening the thin-foundation boundary |
| Numerical correctness | A− | [](../60-validation/index.md) | Analytic cases, limits, round trips, convergence, and FD comparisons cover major kernels | Validation and performance artifacts remain uneven by module | Per-contract evidence coverage and missing-evidence ratchets |
| AD honesty | A | [](../10-theory/rootfinding.md) | Smooth, blocked, zero, value-first, validation-only, and certified implicit paths are distinguished | The taxonomy is distributed across prose and tests | Generated callable-level AD contract matrix |
| JAX architecture | A− | [](../10-theory/index.md) | JIT, VMAP, scan, PyTree, JVP/VJP, and gradient behavior are tested on substantial surfaces | Transform and batching-cost behavior is hard to discover per callable | Generated transform matrix linked to executable evidence |
| Failure semantics | A | [](../10-theory/rootfinding.md) | Root, atmosphere, spatial, and spectral APIs expose typed or structured failure evidence | Some numerical helpers still return less-auditable scalar state | Callable-level boundary and failure records with coverage ratchets |
| Test architecture | A− | [](../60-validation/index.md) | Unit, integration, and validation tiers contain analytic and independent checks | No public module-by-module branch-coverage ratchet is maintained | Coverage policy tied to semantic contracts rather than raw volume alone |
| Units and dimensional safety | B+ | [](../20-architecture/quantity-system.md) | `UnitSystem` is mature and `quantity` is substantially implemented | Two live dimensional surfaces remain while adoption is deferred | Downstream parity, performance, serialization, and migration evidence |
| Provenance | A | [](../60-validation/evidence-index.md) | Source cards, runtime manifests, full-card content digests, and a class-preserving evidence index are freshness checked | Downstream reproduction policies and adoption records are not yet uniform | Pinned downstream evidence manifests and compatibility records |
| API cohesion | B+ | [](../40-api/contracts.md) | Public modules and selected consequential callables now have generated, export-audited contracts | Many public callables remain explicitly unclassified | Prioritized callable-level coverage and downstream query evidence |
| Documentation correctness | A− | [](../60-validation/index.md) | Examples, routes, links, figures, and content contracts are tested | Active guides and some command narratives can lag implementation | Current `CLAUDE.md` plus generated claims and freshness gates |
| Accessibility | B+ | [](../10-theory/rootfinding.md) | Alt text and redundant visual encodings are enforced for new figures | Structural checks do not yet constitute learner-centered accessibility evidence | Keyboard, contrast, comprehension, and learner review gates |
| Performance evidence | B+ | [](../60-validation/evidence-index.md) | Rootfinding and spectra use one units-explicit metric/comparison envelope with deterministic freshness checks | Compile, graph-size, memory, and cost coverage remains uneven by module | Method-appropriate performance records for consequential public contracts |
| Downstream usefulness | B+ | [](../20-architecture/science-general-vision.md) | Active sibling packages motivate and consume selected shared foundations | Adoption claims are not generated from pinned downstream revisions | Symbol-to-project compatibility records and consumer validation |
| External reach | B | [](../index.md) | The package is science-general, permissively licensed, dependency-light, and publicly documented | Stable distribution and independent adoption are limited | Release evidence, external examples, and comparative positioning |
| Maintenance readiness | B+ | [](../95-release/checklist.md) | Release gates, deterministic docs, Ruff, MyPy, and wheel smoke exist | Type strictness and manually duplicated truth remain uneven | Contract-driven docs, stricter typing plan, and unified evidence infrastructure |

## Coverage by scientific area

### Numerics — A−

Rootfinding, interpolation, integration, quadrature, distributions, splines,
grids, optimization, ODEs, operators, and linear algebra have meaningful
mathematical tests. Rootfinding is the current exemplar for separating a robust
value, execution telemetry, finite-map sensitivity, and a certified implicit
derivative. Other modules do not yet expose equally rich contracts and artifacts.

### Coordinates and geometry — B+

Coordinate transformations, singularity documentation, and gradient checks are
useful and science-facing. Frame conventions and geometric degeneracies need
stronger visual and contract-level presentation.

### Spatial methods — B+

Approximate candidate generation and exact pair acceptance are correctly
separated, with explicit capacity and overflow semantics. Scaling, memory, and
adversarial-configuration evidence remain limited.

### Spectra and atmospheres — A−

Source semantics, structured outcomes, interpolation-policy evidence, and host
versus JAX boundaries are unusually explicit. Some validation depends on optional
local artifacts, so reproducibility boundaries must remain visible.

### Units and quantities — B+

The canonical unit-system surface is mature and the quantity layer is deeply
tested. Ecosystem adoption remains an evidence question, not an automatic
migration.

### Parameters and inference bridge — B

The selective bridge is appropriately narrow and avoids becoming an inference
framework. More examples are needed around identifiability, constraint geometry,
and cached derived leaves.

### Testing and provenance — A

Gradient audits, numerical ratchets, source cards, and deterministic reports are
connected by one scientific contract registry and a class-preserving evidence
index. Evidence depth remains uneven across public callables, and source evidence
still must not be mistaken for computational validation.

## Grade-change policy

A grade does not improve merely because a feature or documentation claim lands.
Promotion requires the relevant scientific contract, independent validation,
limitation statement, reproducible artifact where metrics matter, and downstream
adoption evidence where reuse is the justification.

Every change to a grade must record the date, supporting evidence, and the
criterion newly satisfied. A regression in evidence or a newly discovered defect
can lower a grade immediately.

## Current hardening sequence

**Scientific contract registry: implemented.** The registry is deliberately not
evidence-complete. Evidence depth remains uneven, and an unclassified symbol is
not treated as supported.

| Metric identity | Symbol | Value | Units |
| --- | --- | ---: | --- |
| Registered public modules | `N_module,contract` | 16 | modules |
| Callable-level contracts | `N_callable,contract` | 15 | callables |
| Explicitly unclassified public callables | `N_callable,unclassified` | 208 | callables |
| Module-inherited public record types | `N_symbol,inherited` | 125 | symbols |

The completed foundation and next investment are:

**Unified evidence infrastructure: implemented.** Computational measurements,
source provenance, and scientific policy remain separate evidence classes in one
freshness-checked index. Method-specific scientific thresholds remain method-owned;
the shared envelope validates identity, units, comparison truth, environment policy,
and serialization without inventing cross-method acceptance rules.

| Metric identity | Symbol | Value | Units |
| --- | --- | ---: | --- |
| Indexed evidence artifacts | `N_artifact,evidence` | 5 | artifacts |
| Distinct evidence classes | `N_class,evidence` | 3 | evidence classes |

The single next investment is **Phase B:** build executable foundations and
research-student investigations from the contracts and indexed evidence.
