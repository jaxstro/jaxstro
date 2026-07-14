---
title: Package and documentation SOTA assessment
description: Evidence-backed strengths, gaps, and ranked investments for Jaxstro.
---

# Package and documentation SOTA assessment

The [](./package-assessment-scorecard.md) is the living companion to this ranked
investment assessment. Here, state of the art means unusually strong alignment
among scientific scope, numerical contracts, JAX behavior, independent
evidence, provenance, discoverability, and maintenance cost, not the largest
feature list.

## Delivered strengths

| Dimension | Maturity | Evidence-backed assessment |
| --- | --- | --- |
| Scientific breadth | validated | Units, constants, coordinates, geometry, numerics, spatial operations, spectra, atmospheres, parameter bridges, and provenance cover recurring foundation needs. |
| Ownership discipline | ratified | Architecture and decisions keep generic mechanics here while physical interpretation and scientific acceptance remain downstream. |
| Numerical robustness | validated | Rootfinding, interpolation, quadrature, linear algebra, distributions, meshes, and special functions expose boundary and failure contracts in [](../../60-validation/validation.md). |
| Conditioning | validated | Denominator, slope, covariance, positive-definite, and removable-limit checks are explicit. |
| AD honesty | ratified | Smooth, zero, blocked, surrogate, validation-only, value-first, and certified implicit paths remain distinct. |
| JAX transform coverage | validated | Selected `jit`, `vmap`, `lax.map`, `grad`, JVP/VJP, scan, and PyTree behavior is tested with named exceptions. |
| Dimensional safety | implemented | `jaxstro.units` is canonical; `jaxstro.quantity` is implemented without claiming ecosystem adoption. |
| API cohesion | validated | Grouped owner pages, export checks, typed results, and focused tests expose stable contracts. |
| Serialization | implemented | Quantity and provenance records have explicit round trips, but no public root-result replay serializer is validated. |
| Performance and compilation evidence | implemented | Rootfinding and spectra use units-explicit evidence records; compile, graph, runtime, and memory coverage is not uniform. |
| Evidence freshness | validated | Computational artifacts, scientific policy, and source-backed provenance remain distinct and freshness checked. |
| Provenance | validated | Runtime manifests and source-backed cards answer different questions while sharing deterministic identity and digest mechanics. |
| Research workflow coverage | implemented | Three executable investigations connect public APIs to contracts, indexed evidence, limitations, and warranted claims. |
| Limitation coverage | implemented | Workflow and contract records preserve explicit non-claims, but callable-level coverage remains uneven. |
| Accessibility | implemented | The docs gate checks alt text and rendered-route integrity; broader user evidence remains future work. |
| Discoverability | implemented | Semantic start, methods, representations, workflows, API, validation, and project routes answer distinct questions. |
| Downstream reuse | implemented | Known sibling use motivates shared primitives, but pinned adoption and compatibility evidence is not yet generated. |

## High-confidence gaps

- Evidence depth is uneven across modules.
- Transform support and batching cost remain distributed across prose and tests.
- Quantity adoption awaits downstream parity, serialization, performance,
  ergonomics, and migration-cost evidence.
- Research workflow coverage is narrow relative to the public callable surface.
- Downstream reuse lacks a generated symbol-to-project compatibility map.

## Runtime priority

Future runtime ownership follows this exact order:

1. `jaxstro.ml`
2. `jaxstro.numerics.qmc`
3. `jaxstro.uncertainty`
4. `jaxstro.signal`
5. consumer-driven ecosystem adapters
6. fields only after two consumers

Lineax owns general iterative linear solving, Optimistix owns general nonlinear
systems, Quadax owns adaptive quadrature, and Diffrax owns adaptive differential
equations. Jaxstro may add a narrow consumer-driven adapter only when it adds a
scientific contract or evidence layer absent from the delegated owner.

## Now

### 1. Expand contract and evidence coverage

**Impact.** Researchers can discover supported transforms, limitations, and
evidence per consequential public callable.

**Evidence gate.** Every claimed cell links to a test and limitation; unverified
cells remain explicitly unclaimed.

### 2. Expand executable research workflows

**Impact.** More public APIs are exercised from question through computation,
audit, and warranted claim.

**Evidence gate.** Coverage is generated from the schema-2 workflow registry
and distinguishes contract links, indexed evidence, and limitation links.

### 3. Decide quantity adoption

**Impact.** Resolve dual-surface maintenance without destabilizing downstream
science.

**Evidence gate.** Pinned downstream parity, serialization, performance,
ergonomics, and migration-cost results precede the decision.

### 4. Generate downstream adoption evidence

**Impact.** Reuse and adapter claims become observable.

**Evidence gate.** Records derive from pinned imports, compatibility tests, and
last-verified revisions.

## Next

### 1. Build the minimal `jaxstro.ml` evidence-complete slice

**Impact.** Shared preprocessing, data plans, and fixed-step training become
auditable without duplicating Equinox or Optax.

**Evidence gate.** Deterministic restart, leakage, mask, key, and manifest checks
pass on a bounded reference workflow.

### 2. Add `jaxstro.numerics.qmc`

**Impact.** Population and nested-expectation workflows gain reproducible
low-discrepancy construction.

**Evidence gate.** Reference sequences, scrambles, discrepancy, and convergence
checks pass with explicit key ownership.

### 3. Add `jaxstro.uncertainty`

**Impact.** Scientific maps gain reusable propagation mechanics without
claiming posterior inference.

**Evidence gate.** Analytic linear cases and documented nonlinear failure
boundaries support each method.

### 4. Add `jaxstro.signal`

**Impact.** Sample-axis and spectrum conventions become explicit and reusable.

**Evidence gate.** Normalization, units, windows, phase, and delay identities
are independently checked.

## Later

### 1. Add consumer-driven ecosystem adapters

**Impact.** A real consumer can gain units, telemetry, provenance, or evidence
without forking the delegated solver.

**Evidence gate.** The consumer documents the missing contract and verifies the
adapter against Lineax, Optimistix, Quadax, or Diffrax behavior.

### 2. Revisit multidimensional fields

**Impact.** Shared topology and conservation primitives could serve multiple
simulation domains.

**Evidence gate.** Two consumers first agree on topology, staggering, boundary,
conservation, capacity, and JAX shape contracts.

## Evidence required

Accept a future SOTA claim only when the relevant layer includes a contract, a
typed boundary, supported transforms and cost semantics, independent
validation, reproducible artifacts where metrics matter, explicit limitations,
and downstream adoption evidence when reuse justifies ownership.
