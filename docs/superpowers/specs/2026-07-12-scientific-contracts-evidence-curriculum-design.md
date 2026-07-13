---
title: Scientific contracts, evidence, and curriculum design
description: >-
  Approved architecture for a tiered public contract registry, unified evidence
  infrastructure, living package scorecard, and executable foundations curriculum.
---

# Scientific contracts, evidence, and curriculum design

## Purpose

Jaxstro will become an evidence-bearing scientific foundation whose public claims
are inspectable from mathematical meaning through JAX execution and validation.
The work proceeds in dependency order:

1. **A — Scientific contract registry:** define what each public surface claims.
2. **C — Unified evidence infrastructure:** represent how those claims are tested.
3. **B — Executable curriculum:** teach from the same contracts and evidence.

The objective is not to maximize the algorithm count. It is to align scientific
scope, numerical behavior, JAX transforms, automatic-differentiation semantics,
evidence, provenance, pedagogy, and maintenance cost.

## Audiences

The primary audiences are research students, computational-science learners,
astronomy students with uneven preparation, instructors, scientific-software
maintainers, and downstream Jaxstro ecosystem packages.

The documentation does not assume that completing a prerequisite guarantees
current preparedness. It provides optional on-ramps without labeling learners as
deficient or lowering the scientific standard.

## Architectural principles

- Runtime primitives remain domain-agnostic and dependency-light.
- Astronomy supplies the primary teaching language without moving domain physics
  into Jaxstro runtime code.
- Contract metadata never decorates or changes runtime functions.
- A passing transform or finite result does not imply an unregistered scientific
  claim.
- Unsupported, conditional, validation-only, and unverified states remain visible.
- Evidence classes remain distinct; implementation tests do not substitute for
  physical validation or source provenance.
- Generated products replace manually copied claims where practical.
- Missing evidence fails closed rather than being interpreted as support.

## A. Scientific contract registry

### Ownership model

Use a hybrid registry. Central schemas, vocabulary, collection, and validation
live in a dependency-light `jaxstro.contracts` subsystem. Small module-local
manifests hold records near their implementation owners.

Every public module receives a module-level contract. Callable-level contracts
are required for APIs that are numerical, transform-bearing, iterative, stateful,
boundary-sensitive, failure-bearing, differentiated, performance-sensitive, or
shared across projects. Simple constants, aliases, immutable record types, and
obvious accessors may inherit their module contract.

### Core records

The central subsystem defines stable records equivalent to:

- `ModuleContract`
- `CallableContract`
- `TransformContract`
- `BoundaryContract`
- `EvidenceReference`
- `PerformanceContract`
- `MaturityLevel`

A module contract records a stable ID, import path, ownership and non-ownership,
intended uses, runtime/preprocessing boundary, dimensional policy, maturity,
callable manifest, and module evidence.

A callable contract records a stable ID and import path, semantic purpose,
domain and shape assumptions, output and failure semantics, boundary behavior,
JAX transforms, AD classification, batching and cost notes, precision policy,
maturity, evidence references, and limitations.

### Vocabulary

Validated string-backed enums or literals cover maturity, support state, AD
semantics, evidence kind, execution boundary, and failure mode. Transform states
are:

- supported;
- conditionally supported;
- intentionally unsupported;
- validation-only;
- unverified.

Human explanation supplements these states but cannot replace them.

### Collection and validation

The collector:

- assembles module-local manifests;
- validates unique IDs and known vocabulary;
- resolves public import paths;
- validates evidence references;
- reports missing callable records as unclassified;
- emits deterministic normalized JSON;
- generates documentation matrices and coverage reports.

It does not execute benchmarks, load scientific datasets, access the network,
import downstream packages, or participate in JAX tracing.

### Versioning

The inventory includes schema version, package version, source revision when
available, deterministic ordering, and an explicit timestamp policy. It never
contains environment-dependent absolute paths. Contract IDs survive documentation
reorganization. Breaking semantic changes require a contract-version change or
explicit supersession record.

### Initial callable coverage

The first complete callable exemplars are:

- safeguarded and certified implicit rootfinding;
- finite power-law distributions;
- interpolation;
- the registry and evidence tooling itself.

All public modules receive module-level records in the same release. Callable
coverage expands through explicit ratchets rather than fabricated completeness.

### Generated products

The registry generates or checks:

- the public transform-contract matrix;
- AD-semantics and batching-cost tables;
- API maturity and ownership tables;
- evidence and limitation links;
- missing-contract reports;
- a machine-readable package inventory;
- later downstream compatibility and curriculum routing.

## Living package assessment

Create `docs/90-development-log/package-assessment-scorecard.md` as the durable,
updateable version of the package assessment. It records a rubric, current grade,
dated rationale, supporting evidence, the deficiency preventing the next grade,
and explicit promotion criteria for every assessed dimension.

Inventory and evidence status are generated or checked against the registry.
Grades remain editorial judgments. A grade cannot improve merely because a new
feature or claim appears; promotion requires the appropriate contract, independent
validation, artifact, limitation statement, and adoption evidence.

The initial scorecard covers architecture, numerical correctness, AD honesty,
JAX architecture, failure semantics, tests, units, provenance, API cohesion,
documentation, pedagogy, accessibility, performance, downstream usefulness,
external reach, and maintenance readiness.

## Active agent guidance

Rewrite `CLAUDE.md` to describe current architecture, commands, ownership rules,
AD boundaries, and load-bearing invariants. Remove obsolete phase history,
outdated derivative recommendations, duplicated narratives, and inventories that
omit current quantity, spectra, atmosphere, provenance, testing, or implicit-root
surfaces. Historical progress belongs in development records and git history.

## C. Unified evidence infrastructure

### Shared envelope

Every generated evidence artifact receives a common envelope containing:

- schema and artifact versions;
- artifact identity;
- package version and source revision;
- generation command;
- environment snapshot policy;
- precision and deterministic configuration;
- metric records;
- comparisons or acceptance results;
- declared limitations.

Method-specific payloads remain owned by their producers. The common layer
standardizes identity, units, provenance, freshness, and verdict representation,
not the scientific metrics themselves.

### Metrics

Each numerical metric records identity, symbol, value, units, comparison rule,
threshold or reference when applicable, status, and explanation. Dimensionless
quantities explicitly say `dimensionless`; counts state their counting unit.
Hardware-dependent wall time remains evidence rather than a universal threshold
unless the artifact defines an appropriate environment policy.

### Lifecycle

The toolkit supports deterministic emit, strict check, schema validation,
revision and environment comparison, unit validation, stable JSON ordering,
Markdown rendering, artifact freshness, and rejection of missing or extra fields.
Documentation reads measured values from artifacts instead of copying them.

### Evidence classes

The infrastructure preserves distinctions among analytic evidence,
implementation tests, AD-versus-FD checks, convergence evidence, performance
evidence, scientific-source provenance, and downstream compatibility evidence.

### Migration

Migrate without changing scientific thresholds:

1. value-first and implicit-root artifacts;
2. spectra performance and conservation artifacts;
3. source-backed provenance cards;
4. atmosphere artifacts with optional external-data policies.

Contract records link to stable evidence IDs rather than filenames. Missing IDs,
stale schemas, stale generated tables, or absent required evidence classes fail
validation.

## B. Foundations and executable curriculum

### Pedagogical spine

Every substantial activity follows:

```text
predict → compute → audit → state the warranted claim
```

Prediction names units, signs, limits, invariants, conditioning, expected
failures, and derivative meaning. Computation retains method state and evidence.
Auditing uses analytic identities, limiting cases, independent numerical methods,
finite differences, convergence, or provenance. The final claim must be no
stronger than that evidence.

### Foundations navigation

Add a `Foundations: the ideas we will not assume` section:

1. Choose your path.
2. Functions, units, and scales.
3. What is a model? Representations, assumptions, and predictions.
4. Parameters, state, and dimensionality.
5. Linear algebra as the language of change.
6. What a derivative means: change, sensitivity, and scientific evidence.
7. Probability and distributions.
8. Models, inference, and information.
9. Sensitivity, conditioning, and identifiability.
10. From mathematical relations to differentiable programs.

The readiness route is short and ungraded. It directs learners to concepts
without assigning readiness labels.

### Layered mathematical depth

Foundation pages begin with scalar and physical intuition, develop an accessible
linear-map view, and include optional deeper treatments. The derivative page
connects rates, local linear maps, Jacobians, JVPs, VJPs, gradients, tangent and
cotangent language, likelihood scores, Hessians, Fisher information, implicit
sensitivities, and the derivative of the executed program.

The probability material distinguishes probability mass from density, support,
normalization, expectation, covariance, conditioning, transformations, sampling,
and aleatoric from epistemic uncertainty. The inference material connects
parameters, physical models, observables, measurement models, likelihoods,
priors, posteriors, prediction, nuisance parameters, identifiability, and model
checking.

### Models and information

A model is a deliberately incomplete representation connecting assumptions and
inputs to predictions about selected aspects of a system. Pages distinguish
conceptual, mathematical, computational, statistical, generative, and surrogate
models. They explain parameters, state, latent variables, observables,
hyperparameters, and nuisance parameters.

Models are treated as question-dependent information compression. The curriculum
distinguishes scientific information, retained data, Shannon information, and
information about parameters. It states that discarded information cannot be
recovered by a better optimizer, sufficiency is model-relative, parameter count
does not equal scientific information, and a precise posterior can accompany a
misspecified model.

### Dimensionality

The curriculum distinguishes spatial dimension, physical dimension, array rank
and shape, data-space dimension, parameter-space dimension, state-space
dimension, intrinsic dimension, effective dimension, and model dimension. It
emphasizes that parameter space can be high-dimensional even for three-dimensional
physical space, while many nominal parameters may yield few identifiable
combinations.

### Linear algebra

The standalone linear-algebra foundation covers vectors as perturbations, linear
maps, matrices as coordinate representations, basis, dot products, norms,
projection, Jacobians, covariance geometry, eigenvectors, singular vectors, null
spaces, condition numbers, quadratic forms, and Hessians. Later pages link back
to this vocabulary rather than redefining it.

### Science examples

Astronomy is the recurring teaching language:

- Newtonian gravity for units, inverse-square sensitivity, singularities,
  vectors, ODEs, and dynamics boundaries;
- the Stefan–Boltzmann law for derivatives, Jacobians, JVPs, VJPs, uncertainty,
  likelihoods, and identifiability;
- parallax and flux-distance relations for inversion and conditioning;
- finite stellar-mass power laws for distributions and transformations;
- stellar and atmosphere tables for interpolation;
- spectra for high-dimensional data and structured low-dimensional models;
- stellar clusters for spatial search and state-space examples.

Each example states what Jaxstro computes, what interpretation is supplied for
teaching, what the downstream package owns, and what evidence is needed before a
scientific claim. Gravax, Progenax, Fluxax, Startrax, Stellax, and other packages
are referenced only with verified current status; no downstream imports enter
Jaxstro examples.

### Executable learning units

Each unit contains a research question, prerequisites, objectives, prediction,
bounded computation, audit checklist, misconception check, claim-writing prompt,
expected evidence classes, instructor notes, and accessibility metadata.

Canonical exercises are lightweight Python modules and MyST pages using public
APIs and deterministic fixtures. Optional notebooks may be generated but cannot
become a second source of truth. Automated feedback names scientific contract
failures rather than reporting only a wrong scalar.

Initial investigations cover root values and sensitivities, removable
singularities in finite power laws, and interpolation boundary policies. The
site generates prerequisites, related APIs, transform behavior, limitations,
validation links, and curriculum gaps from contract records.

## Delivery phases

### Phase A checkpoints

- **A0:** assessment scorecard, navigation, and `CLAUDE.md` currency.
- **A1:** schemas, collector, module-level coverage, and normalized inventory.
- **A2:** callable exemplars and coverage ratchets.
- **A3:** generated contract, transform, maturity, evidence, and limitation pages.

### Phase C checkpoints

- **C1:** common artifact envelope and strict tooling.
- **C2:** rootfinding artifact migration.
- **C3:** selected spectra, provenance, and atmosphere migrations.

### Phase B checkpoints

- **B1:** foundations framework and readiness routing.
- **B2:** complete conceptual foundation spine.
- **B3:** executable rootfinding, distribution, and interpolation investigations.
- **B4:** instructor guidance, rubrics, optional exports, and curriculum coverage.

Use targeted independent reviews only at checkpoint boundaries.

## Failure behavior

The registry and evidence layer reject unknown vocabulary, duplicate IDs,
unresolvable public paths, missing evidence, stale artifacts, invalid units,
undocumented conditional support, and false completeness. Optional evidence is
unavailable rather than silently passing. Documentation generation does not
load datasets, execute benchmarks, access the network, or import downstream
projects.

## Verification strategy

Write focused failing tests before each implementation slice. Required gates
cover schema rejection, deterministic serialization, public-path resolution,
optional-dependency isolation, evidence resolution, generated-page freshness,
coverage ratchets, documentation navigation, analytic astronomy examples,
JAX/AD claims, executable learning fixtures, Ruff, MyPy, and the strict docs
build.

Do not weaken existing scientific thresholds while migrating evidence. Report
measured numerical results in tables with metric identity, symbol, value, and
units.

## Scope boundaries

The registry reports current truth; it does not certify every callable in the
first release. The evidence layer standardizes proof envelopes rather than
forcing one metric schema on different methods. The curriculum supplies the
foundations needed to reason with Jaxstro but does not replace complete courses
in calculus, linear algebra, probability, statistics, or astrophysics. Runtime
Jaxstro remains free of downstream domain logic.
