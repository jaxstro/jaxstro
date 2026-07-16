---
title: Future modules and capabilities roadmap
description: Evidence-gated priorities for future Jaxstro ownership and ecosystem adapters.
---

# Future modules and capabilities roadmap

This roadmap separates implemented foundations, proposed Jaxstro owners, and
machinery delegated to established ecosystem libraries. A priority is not an
importable API. Every unchecked item remains a proposal until its ownership,
consumer need, runtime contract, and independent evidence are accepted.

The execution pattern is **predict -> compute -> audit -> state the warranted claim**.
Informax retains posterior inference, simulation-based inference,
identifiability, and experimental-design policy; domain packages retain physical
models and scientific acceptance.

## Existing methods and features

Jaxstro already owns domain-general units, constants, coordinates, geometry,
fixed-shape numerical primitives, spatial mechanics, spectra representation,
parameter bridges, provenance, and validation tooling. The current public
surface is indexed at [](../../50-api/api.md), with evidence boundaries at
[](../../60-validation/validation.md).

Implemented numerical families include interpolation, complete one-dimensional
fixed quadrature, replay-differentiable adaptive one-dimensional quadrature
with an alpha quantity boundary,
fixed-step differential equations, scalar roots, distributions, random-state
mechanics, small dense linear algebra, operators, structured one-dimensional
meshes, and differentiability audits. These foundations do not authorize a
duplicate general solver stack.

## What Jaxstro should become

Jaxstro should remain the reusable, dependency-light substrate for auditable
JAX-native scientific computation. It may add a coherent module when the
contract is domain-general, the execution boundary is explicit, and evidence
shows that central ownership reduces real duplication.

## Recommended additions

### Active program: `jaxstro.quad`

Jaxstro owns the approved quadrature capability program.
Quadax remains a validation and benchmark comparator, not a runtime dependency
or the public owner of Jaxstro's integration contract.

- [x] Establish the canonical namespace, domains, measures, rules, and result
  vocabulary.
- [x] Add sampled-data canonicalization and the complete fixed-rule family.
- [x] Add adaptive Gauss-Kronrod, Clenshaw-Curtis, tanh-sinh, and Romberg
  controllers with typed failure evidence.
- [x] Add replay derivatives, moving-bound evidence, and the alpha quantity
  boundary for all five adaptive methods.
- [ ] Complete the matched external comparison, migration guidance, and the
  Phase A release gate.

The priority order is fixed. Later items do not leapfrog earlier items merely
because a prototype exists.

### Priority 1: `jaxstro.ml`

Own a narrow scientific-ML substrate around libraries that already own models
and optimizers.

- [ ] Add host-fit, JAX-apply preprocessing PyTrees with named feature axes.
- [ ] Add deterministic split and fixed-shape batch plans with explicit keys.
- [ ] Add optimizer-compatible fixed-step training state and auditable traces.
- [ ] Record dataset, preprocessing, seed, structure, and checkpoint manifests.
- [ ] Validate masked execution, restart parity, and data-leakage checks.

Equinox owns callable PyTrees and neural modules. Optax owns optimizer
transformations. Jaxstro does not need a model zoo, optimizer zoo, or second
provenance system.

### Priority 2: `jaxstro.numerics.qmc`

Own sequence construction and evidence for fixed-shape quasi-Monte Carlo
workflows.

- [ ] Add reference-checked Sobol construction and replicated scrambles.
- [ ] Add Latin-hypercube construction with explicit key ownership.
- [ ] Add discrepancy diagnostics and replicated convergence evidence.
- [ ] Record sequence, scramble, dimension, sample count, and seed provenance.

### Priority 3: `jaxstro.uncertainty`

Own mathematical propagation through an already specified scientific map.

- [ ] Add linearized covariance pushforward with conditioning diagnostics.
- [ ] Add sigma-point propagation with explicit weighting conventions.
- [ ] Add keyed ensemble propagation with deterministic shape and key policy.
- [ ] Validate analytic linear cases and document nonlinear failure boundaries.

This module would propagate uncertainty; it would not construct posteriors or
duplicate NumPyro, BlackJAX, or Informax.

### Priority 4: `jaxstro.signal`

Own scientific conventions around sampled axes and spectral interpretation.

- [ ] Define cadence, window, one-sided/two-sided, amplitude, and power contracts.
- [ ] Add window functions with equivalent-noise-bandwidth evidence.
- [ ] Add spectral estimation, phase, and delay helpers driven by consumers.
- [ ] Validate normalization, units, boundary policy, and limiting identities.

JAX owns FFT and convolution mechanics. Jaxstro adds a function only when it
adds a scientific contract rather than another spelling of `jax.numpy`.

### Priority 5: consumer-driven ecosystem adapters

General solvers remain delegated. A small adapter is justified only when a
consumer needs units, shape policy, telemetry, provenance, or evidence absent
from the underlying owner.

- [ ] Use Lineax and JAX for iterative linear systems; add no Jaxstro solver clone.
- [ ] Use Optimistix for nonlinear systems, fixed points, and minimization.
- [ ] Use Quadax as an independent adaptive-quadrature validation and benchmark
  comparator without adding it as a runtime dependency.
- [ ] Use Diffrax for adaptive ODE, SDE, and CDE solving.
- [ ] Require a concrete consumer and an adapter-specific evidence gap before
  proposing any wrapper.
- [ ] Keep adapter status, dependency cost, and upstream behavior explicit.

### Priority 6: fields only after two consumers

Multidimensional fields remain a deferred abstraction until at least two
consumers establish a shared topology and conservation boundary.

- [ ] Identify two independent consumers and their common field contract.
- [ ] Ratify topology, staggering, boundary, conservation, and shape semantics.
- [ ] Validate reference identities and conservation before adding a module.
- [ ] Keep domain discretization and acceptance policy with the consumer.

## What not to add

- No homegrown neural-network framework; use Equinox.
- No general optimizer implementation; use Optax or a specialized owner.
- No general MCMC, VI, NPE, SBI, or posterior orchestration; use Informax and
  established inference libraries.
- No Jaxstro implementation of iterative linear solvers, general nonlinear
  systems, or adaptive differential equations. The separately approved
  `jaxstro.quad` program owns adaptive quadrature.
- No multidimensional field framework before two consumers agree on the shared
  abstraction.
- No method added only because another general-purpose library exposes it.

## Build checklist

- [ ] Generate callable-level transform and maturity coverage.
- [ ] Complete the remaining `jaxstro.quad` matched-comparison, migration, and
  release gates; replay, quantity normalization, derivations, and deterministic
  first-order evidence are complete.
- [ ] Decide quantity adoption from downstream parity and migration evidence.
- [ ] Deliver the smallest evidence-complete `jaxstro.ml` vertical slice.
- [ ] Deliver reference-sequence and convergence gates for `jaxstro.numerics.qmc`.
- [ ] Deliver analytic and nonlinear-boundary evidence for `jaxstro.uncertainty`.
- [ ] Deliver convention and normalization evidence for `jaxstro.signal`.
- [ ] Admit ecosystem adapters only from documented consumer gaps.
- [ ] Revisit fields only after two consumers satisfy the shared-contract gate.
- [ ] Generate downstream compatibility records from pinned revisions.
- [ ] Update contracts, evidence, limitations, scorecard, and roadmap together.

## Definition of done for a roadmap item

A checkbox moves to complete only when the capability has:

1. a mathematical or semantic contract;
2. an accepted ownership boundary and named non-owners;
3. a public API with typed failure behavior, when runtime code is appropriate;
4. explicit JAX transforms, static arguments, and cost semantics;
5. independent validation rather than self-consistency alone;
6. deterministic artifacts where measured results matter;
7. documented limitations and a bounded warranted claim;
8. consumer evidence when shared ownership or an adapter is the justification;
9. fresh contract, evidence, scorecard, and roadmap records.

Companion evidence lives in the [](../../50-api/research-infrastructure/contracts.md),
[](../../60-validation/evidence-index.md), [](./package-assessment-scorecard.md),
and [](./sota-assessment.md).
