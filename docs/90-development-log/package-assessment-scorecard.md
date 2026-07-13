---
title: Jaxstro package assessment scorecard
description: Living grades, evidence, limiting deficiencies, and promotion gates for the package and its curriculum.
---

# Jaxstro package assessment scorecard

This is the living assessment of Jaxstro as scientific infrastructure, a public
package, and curriculum material. The companion
[](./sota-assessment.md) ranks investments; this page records current grades and
what evidence is required to change them.

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

| Dimension | Grade | Evidence-backed rationale | Deficiency preventing the next grade | Promotion evidence required |
| --- | --- | --- | --- | --- |
| Internal differentiable-science foundation | A− | Broad shared primitives, explicit ownership, and extensive executable validation | Evidence depth and downstream verification are not uniform | Generated contracts, uniform evidence classes, and pinned adoption records |
| Public scientific package | B+ | Cohesive surface, minimal core dependencies, typed failures, and a public site | Public reach and external adoption trail the implementation | Stable release, public compatibility evidence, and external use cases |
| Curriculum concept | A− | Predict → compute → audit, objectives, misconceptions, and science-pattern routing form a coherent philosophy | The conceptual foundations are not yet a complete course-ready sequence | Approved foundations spine with reviewed scientific examples |
| Executable curriculum | B | Tested examples exist and documentation is build-gated | Activities, instructor materials, diagnostics, and rubrics are incomplete | CI-executed investigations and instructor-facing assessment materials |
| Architecture and ownership | A | Thin-foundation ADRs and one-way ecosystem boundaries are explicit | Some active summaries lag newer modules | Scientific contract registry and generated ownership inventory |
| Numerical correctness | A− | Analytic cases, limits, round trips, convergence, and FD comparisons cover major kernels | Validation and performance artifacts remain uneven by module | Per-contract evidence coverage and missing-evidence ratchets |
| AD honesty | A | Smooth, blocked, zero, value-first, validation-only, and certified implicit paths are distinguished | The taxonomy is distributed across prose and tests | Generated callable-level AD contract matrix |
| JAX architecture | A− | JIT, VMAP, scan, PyTree, JVP/VJP, and gradient behavior are tested on substantial surfaces | Transform and batching-cost behavior is hard to discover per callable | Generated transform matrix linked to executable evidence |
| Failure semantics | A | Root, atmosphere, spatial, and spectral APIs expose typed or structured failure evidence | Some numerical helpers still return less-auditable scalar state | Callable-level boundary and failure records with coverage ratchets |
| Test architecture | A− | Unit, integration, and validation tiers contain analytic and independent checks | No public module-by-module branch-coverage ratchet is maintained | Coverage policy tied to semantic contracts rather than raw volume alone |
| Units and dimensional safety | B+ | `UnitSystem` is mature and `quantity` is substantially implemented | Two live dimensional surfaces remain while adoption is deferred | Downstream parity, performance, serialization, and migration evidence |
| Provenance | A− | Source cards, runtime manifests, deterministic rendering, and freshness checks exist | Artifact schemas and environment policies are duplicated | Unified evidence envelope without weakened scientific thresholds |
| API cohesion | B+ | Public modules, exports, types, and evidence routes are documented | The manual API catalog is growing and can drift | Generated contract-backed reference tables and export checks |
| Documentation correctness | A− | Examples, routes, links, figures, and content contracts are tested | Active guides and some command narratives can lag implementation | Current `CLAUDE.md` plus generated claims and freshness gates |
| Accessibility | B+ | Alt text and redundant visual encodings are enforced for new figures | Structural checks do not yet constitute learner-centered accessibility evidence | Keyboard, contrast, comprehension, and learner review gates |
| Performance evidence | B | Rootfinding and spectra have reproducible cost evidence | Compile, warm runtime, graph size, memory, and evaluation metrics are not standardized | Shared metric envelope and method-appropriate performance records |
| Downstream usefulness | A− | Gravax, Progenax, Fluxax, and Startrax motivate and consume shared foundations | Adoption claims are not generated from pinned downstream revisions | Symbol-to-project compatibility records and consumer validation |
| External reach | B | The package is science-general, permissively licensed, dependency-light, and publicly documented | Stable distribution and independent adoption are limited | Release evidence, external examples, and comparative positioning |
| Maintenance readiness | B+ | Release gates, deterministic docs, Ruff, MyPy, and wheel smoke exist | Type strictness and manually duplicated truth remain uneven | Contract-driven docs, stricter typing plan, and unified evidence infrastructure |

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

### Testing and provenance — A−

Gradient audits, numerical ratchets, source cards, and deterministic reports are
distinctive strengths. They remain several strong tools rather than one uniform
Scientific contract registry and evidence system.

## Grade-change policy

A grade does not improve merely because a feature or documentation claim lands.
Promotion requires the relevant scientific contract, independent validation,
limitation statement, reproducible artifact where metrics matter, and downstream
adoption evidence where reuse is the justification.

Every change to a grade must record the date, supporting evidence, and the
criterion newly satisfied. A regression in evidence or a newly discovered defect
can lower a grade immediately.

## Current hardening sequence

1. Build the Scientific contract registry.
2. Unify evidence envelopes and freshness infrastructure.
3. Build executable foundations and research-student investigations from those
   contracts and evidence.
