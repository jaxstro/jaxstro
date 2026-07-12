---
title: Package and documentation SOTA assessment
description: Evidence-backed strengths, gaps, and ranked investments for Jaxstro.
---

# Package and documentation SOTA assessment

Here “state of the art” means unusually strong alignment among scientific scope,
numerical contracts, JAX behavior, independent evidence, provenance, pedagogy,
and maintenance cost—not the largest feature list. Maturity labels are
**ratified**, **validated**, **implemented**, **experimental**, and **planned**;
no numeric score is assigned without an observable rubric.

## Delivered strengths

| Dimension | Maturity | Evidence-backed assessment |
| --- | --- | --- |
| Scientific breadth | validated | Units, constants, coordinates, geometry, numerics, spatial operations, spectra, atmospheres, parameter bridges, and provenance cover recurring lower-level needs. |
| Ownership discipline | ratified | ADRs and architecture keep generic mechanics here and physical interpretation, retry policy, and scientific state downstream. |
| Numerical robustness | validated | Rootfinding, interpolation, quadrature, linear algebra, distributions, meshes, and special functions expose boundary and failure contracts in [](../60-validation/index.md). |
| Conditioning | validated | Denominator, slope-floor, covariance, positive-definite, and limiting-kernel checks are explicit. |
| AD honesty | ratified | The theory index classifies smooth, zero, blocked, surrogate, validation-only, value-first, and certified implicit paths. |
| JAX transform coverage | validated | Selected `jit`, `vmap`, `lax.map`, `grad`, JVP/VJP, scan, and PyTree behavior is tested with named exceptions. |
| Dimensional safety | implemented | `jaxstro.units` is canonical; `jaxstro.quantity` implements exact dimensions and transforms without claiming ecosystem adoption. |
| API cohesion | validated | Exports, reference docs, typed results, and focused tests give downstream packages stable contracts. |
| Serialization | implemented | Scalar quantities have explicit round trips and array rejection; provenance records render deterministically; root telemetry uses fixed-shape NamedTuples suitable for transport, but no public root round-trip/replay serializer is yet validated. |
| Performance and compilation evidence | implemented | Rootfinding and spectra have evidence, but compile/runtime/memory coverage is not uniform. |
| Evidence freshness | validated | Root artifacts, provenance pages, and figures have emit/check or deterministic freshness gates. |
| Provenance | validated | Runtime manifests and source-backed cards separate computational reproduction from scientific-source claims. |
| Curriculum quality | implemented | Predict → compute → audit, science-pattern routing, objectives, activities, and evidence-bearing figures now form a learning spine. |
| Accessibility | implemented | The docs gate checks alt text; new figures also use marker and line-style encodings beyond color. |
| Discoverability | implemented | Diátaxis, three entry doors, science patterns, modules, API reference, and validation serve distinct questions. |
| Downstream reuse | implemented | Local parity fixtures and known sibling use motivate shared primitives, but pinned adoption and compatibility evidence is not yet generated or validated. |

## High-confidence gaps

- Evidence depth is uneven across modules; several rely on unit and FD tests
  without standardized compile, runtime, graph, or memory reports.
- JAX transform coverage is distributed across prose and tests rather than a
  generated public contract matrix.
- Dimensional safety has two live surfaces until quantity adoption is ratified
  or rejected.
- Evidence manifests duplicate schema, environment, unit, and freshness logic.
- Curriculum examples are not yet a reusable executable exercise corpus.
- Visual coverage is strong for selected modules but sparse for distribution
  limits, AD contracts, validation logic, quantities, and provenance.
- Downstream reuse lacks a generated symbol-to-project compatibility map.

## Now

### 1. Unify evidence artifact infrastructure

**Impact.** Shared schema, unit, environment, revision-policy, and comparison
helpers remove drift across root, spectra, atmosphere, and future artifacts.

**Audience and ownership.** Maintainers and research students; Jaxstro owns the
generic toolkit while each method owns its metrics.

**Risk.** Forcing unlike scientific evidence into one rigid schema.

**Evidence gate.** Migrate existing artifacts without weakening any freshness,
unit, or scientific threshold.

### 2. Generate a JAX transform-contract matrix

**Impact.** Users can see supported transforms, static arguments, batching cost,
and smoothness caveats per public API.

**Audience and ownership.** Scientific developers; structured contracts live
beside exports and resolve to tests and prose.

**Risk.** Check boxes could hide conditional domains.

**Evidence gate.** Every claimed cell links to a test and limitation; unverified
cells remain explicitly unclaimed.

### 3. Ratify or reject `jaxstro.quantity` adoption

**Impact.** Resolve dual-surface maintenance and either unlock ecosystem-wide
dimensional safety or establish a stable non-adoption boundary.

**Audience and ownership.** All sibling packages; each owns scientific parity
while Jaxstro owns quantity mechanics.

**Risk.** Premature migration could destabilize mature science.

**Evidence gate.** Downstream parity, serialization, performance, ergonomics,
and migration-cost reports precede the ADR.

### 4. Make curriculum activities executable

**Impact.** Students can run the exact fixtures used by the site and receive
invariant-focused feedback.

**Audience and ownership.** Computational-science courses; instructors own
sequencing and grading.

**Risk.** Copied notebooks can become a stale second test suite.

**Evidence gate.** Exercises import public APIs or tested helpers, run in CI,
and include expected reasoning and accessibility metadata.

### 5. Generate downstream adoption evidence

**Impact.** Reuse claims become observable, and abstractions are judged by real
cross-project demand.

**Audience and ownership.** Ecosystem maintainers; downstream projects retain
scientific acceptance.

**Risk.** A manual matrix would immediately age.

**Evidence gate.** Generate records from pinned imports, compatibility tests,
and last-verified revisions.

## Next

### 1. Visualize removable singularities

**Impact.** Teach why correct forward branches can have wrong parameter derivatives.

**Evidence gate.** Curves and limits derive from public APIs, analytic limits,
and central FD; color is not the only encoding.

### 2. Visualize the AD contract taxonomy

**Impact.** Make smooth, blocked, value-first, validation-only, and certified
implicit claims distinguishable at a glance.

**Evidence gate.** First deliver the planned transform-contract or maturity
registry; then require every visual class to map to that registry and an
executable example.

### 3. Add validation-triangle figures

**Impact.** Separate analytic, AD, FD, convergence, and provenance support.

**Evidence gate.** Figures are generated from artifacts and fail their declared tolerances when stale.

### 4. Standardize compile, JAXPR, runtime, and memory reports

**Impact.** Transform-heavy APIs can be assessed on developer cost as well as values.

**Evidence gate.** Reports separate compile, warm runtime, evaluations, graph primitives, and peak memory.

### 5. Define public API maturity metadata

**Impact.** Reference pages can distinguish ratified, validated, implemented, experimental, and planned surfaces.

**Evidence gate.** One registry is checked against exports, docs, and required validation anchors.

### 6. Build provenance and ownership-flow figures

**Impact.** Trace source artifact → preparation → JAX kernel → validation → claim.

**Evidence gate.** Every node resolves to a real record, module, or evidence page.

### 7. Add science-facing exercise sequences

**Impact.** Turn roots, interpolation, integration, distributions, spatial methods, and spectra into research-onboarding units.

**Evidence gate.** Each sequence has objectives, prerequisites, tested code, misconception checks, and a bounded claim rubric.

## Later

### 1. Broaden certified implicit primitives after adoption evidence

**Impact.** Vector roots or fixed points could support differentiable equilibria.

**Evidence gate.** Two consumers share the contract and validate uniqueness, conditioning, residual error, and linear solves first.

### 2. Evaluate adaptive integration ownership

**Impact.** Error-controlled workflows could extend fixed-step coverage.

**Evidence gate.** Demonstrate cross-project demand for a narrowly defined
primitive, compare against established specialized solvers as the default
owner, and show that the proposed boundary avoids duplicating their stack.

### 3. Evaluate sparse iterative operators

**Impact.** Larger inverse problems may need matrix-free solves.

**Evidence gate.** Demonstrate cross-project demand, conditioning, convergence telemetry, AD semantics, and dependency fit.

### 4. Evaluate higher-dimensional mesh primitives

**Impact.** Conservative methods could serve more simulation domains.

**Evidence gate.** Define topology, conservation, capacity, JAX shapes, ownership, and two real consumers before implementation.

## Evidence required

Accept a future SOTA claim only when the relevant layer includes:

1. a mathematical or semantic contract;
2. a public API with typed boundary and failure behavior;
3. supported JAX transforms and cost semantics;
4. independent validation rather than self-consistency alone;
5. reproducible artifacts where metrics matter;
6. accessible teaching that states what the evidence does not prove;
7. downstream adoption evidence when reuse justifies ownership.

The highest-impact direction is not adding the most algorithms. It is making
the existing breadth uniformly inspectable from equation to execution to claim,
while admitting new primitives only when reuse and evidence justify them.
