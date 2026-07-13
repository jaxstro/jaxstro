---
title: Future modules and capabilities roadmap
description: >-
  A living inventory and implementation checklist for growing Jaxstro into the
  ecosystem's primary numerical-methods and scientific-ML foundation without
  absorbing Informax or duplicating specialized JAX libraries.
---

# Future modules and capabilities roadmap

This is the living build note for Jaxstro's future numerical-methods,
scientific-ML, and supporting infrastructure. It preserves the package-wide
assessment and architectural advice that motivated the roadmap, then turns the
recommended build order into checklists.

The roadmap is not a promise to implement every named method. A new capability
still has to satisfy the thin-foundation admission rule: a generic mathematical
contract, a reusable public API, explicit JAX and AD semantics, independent
validation, and evidence that Jaxstro is the right owner.

## Summary

- Jaxstro is already a strong shared numerical-science foundation: units,
  quantities, coordinates, robust numerics, spatial algorithms, spectra,
  atmospheres, parameter transforms, scientific contracts, validation, and
  provenance.
- Its numerical breadth is substantial; its distinguishing strength is not
  merely algorithm count, but explicit JAX, AD, boundary, failure, and evidence
  contracts.
- It is not yet a general scientific-ML toolkit. It has the necessary
  foundations—Equinox parameterization, autodiff products, operators, losses,
  RNG, and provenance—but lacks a coherent data, training, and checkpoint layer.
- The target is to make Jaxstro the primary reusable numerical and scientific-ML
  substrate while keeping inference, information geometry, calibration, SBI,
  and experimental design in Informax.
- The highest-value additions are a narrow `jaxstro.ml` layer, matrix-free
  iterative solvers, vector nonlinear and implicit solves, quasi-Monte Carlo,
  adaptive quadrature, and signal-processing primitives.
- Jaxstro should integrate with Equinox, Optax, Diffrax, NumPyro, and specialized
  solver libraries where appropriate—not attempt to replace all of them.

## Current package scale

These are exported names, not algorithm counts; exports also include types,
constants, and status identifiers. The values record the inventory audited on
2026-07-12 and must be refreshed when the public API or generated registries
change.

| Metric identity | Symbol | Value | Units |
| --- | --- | ---: | --- |
| Numerical public exports | `N_export,numerics` | 181 | exports |
| Quantity public exports | `N_export,quantity` | 61 | exports |
| Atmosphere public exports | `N_export,atmosphere` | 52 | exports |
| Testing public exports | `N_export,testing` | 46 | exports |
| Spectral public exports | `N_export,spectra` | 22 | exports |
| Evidence public exports | `N_export,evidence` | 18 | exports |
| Scientific-contract public exports | `N_export,contract` | 16 | exports |
| Spatial public exports | `N_export,spatial` | 12 | exports |
| Parameterization public exports | `N_export,params` | 6 | exports |
| Registered public modules | `N_module,contract` | 16 | modules |
| Callable-level scientific contracts | `N_callable,contract` | 15 | callables |
| Explicitly unclassified public callables | `N_callable,unclassified` | 208 | callables |
| Module-inherited public record types | `N_type,inherited` | 125 | types |
| Indexed evidence artifacts | `N_artifact,evidence` | 5 | artifacts |
| Evidence classes | `N_class,evidence` | 3 | classes |

The important interpretation is that Jaxstro has broad implementation coverage,
but callable-level contract classification has not yet caught up with that
breadth.

## Existing methods and features

Checked items describe implemented capability families. Their detailed maturity
and evidence remain governed by the [](./package-assessment-scorecard.md),
[](../40-api/contracts.md), and [](../60-validation/index.md).

### Numerical accuracy and defensive primitives

- [x] Neumaier compensated scalar, vector, array, and dot-product summation.
- [x] Safe logarithm, exponential, division, log-sum-exp, relative error, and
  convergence helpers.
- [x] Finiteness, positivity, range, and monotonicity validation.
- [x] Concrete-input validation that does not silently fabricate traced
  assertions.
- [x] Explicit float64 and highest-matmul-precision configuration.

This is valuable infrastructure because numerical stability is handled once
rather than rediscovered inside every downstream physics package.

### Automatic differentiation

- [x] JVP and VJP.
- [x] Jacobian-vector and vector-Jacobian products.
- [x] Hessian-vector products.
- [x] Gauss–Newton products.
- [x] Generic empirical Fisher-style products.
- [x] Independent finite-difference gradients and Jacobians.
- [x] Directional-derivative comparison.
- [x] Explicit gradient contracts: smooth, known-zero, blocked, surrogate, and
  validation-only.

This is one of Jaxstro's strongest areas. It distinguishes “JAX can differentiate
this expression” from “this derivative means what the scientific claim says it
means.”

### Scalar rootfinding and inversion

- [x] Bracket expansion.
- [x] Bisection and batched bisection.
- [x] Newton and Newton-with-gradient execution.
- [x] Monotone inverse interpolation and PPF inversion.
- [x] Safeguarded inverse-quadratic, secant, and midpoint rootfinding.
- [x] Low-level bracket initialization, proposal, update, and checkpointable
  advancement.
- [x] Fixed-shape root telemetry with typed termination status.
- [x] Explicit invalid-trial exclusion through `valid=False`.
- [x] Physical per-lane evaluation skipping through the mapped solver.
- [x] Separately gated implicit-function derivatives.
- [x] Fail-closed certificates for convergence, residual, bracket width,
  uniqueness assumptions, finite slope, and conditioning.

This supports equilibria, event locations, inverse constitutive relations,
timestep controllers, and differentiable implicit scientific models.

### Interpolation and approximation

- [x] Linear one-dimensional interpolation.
- [x] Cubic Hermite interpolation.
- [x] PCHIP-style monotone cubic interpolation.
- [x] Natural cubic splines.
- [x] Tabulated-function PyTrees.
- [x] Static-rank regular-grid interpolation.
- [x] Bilinear and trilinear interpolation.
- [x] Clamp, fill, and eager-rejection boundary policies.
- [x] B-spline bases and design matrices.
- [x] Direct and de Boor spline evaluation.
- [x] Derivatives, antiderivatives, and definite integrals.
- [x] Roughness penalties.
- [x] Fixed-knot least-squares fitting.
- [x] Quantile-based adaptive knot placement.
- [x] Tensor-product design matrices.

This is already a strong foundation for stellar tracks, atmosphere grids,
tabulated equations of state, calibration surfaces, and surrogate
representations.

### Integration and quadrature

- [x] Trapezoidal and cumulative trapezoidal integration.
- [x] Simpson and cumulative Simpson integration.
- [x] Gauss–Legendre quadrature.
- [x] Probabilists' Gauss–Hermite quadrature.
- [x] Gauss–Laguerre quadrature.
- [x] Clenshaw–Curtis quadrature.
- [x] Hermite basis and expansion coefficients.

Fixed-node construction and differentiable integrand evaluation have
deliberately separate contracts.

### Linear algebra and operators

- [x] Norms, projection, and condition-number diagnostics.
- [x] Weighted least squares.
- [x] QR and SVD solves.
- [x] Covariance and correlation matrices.
- [x] Positive-definiteness tests and diagonal jitter selection.
- [x] Dense, diagonal, scaled, sum, product, transpose, and block-diagonal
  operators.
- [x] `matvec`, reverse multiplication, composition, and dense realization.

This is good for small dense scientific problems and matrix-free composition. It
is not yet sufficient for large sparse or Krylov problems.

### Optimization helpers

- [x] Squared, Huber, and pseudo-Huber losses.
- [x] Weighted objective summaries.
- [x] Fixed-iteration Armijo backtracking.
- [x] Gradient and relative-step norms.
- [x] Convergence summaries and typed line-search results.

The boundary is appropriately narrow: these are trustworthy optimizer
ingredients, not a homegrown replacement for Optax or specialized optimization
libraries.

### Differential equations

- [x] Euler.
- [x] Midpoint/RK2.
- [x] RK4.
- [x] Fixed-step dispatch using `lax.scan`.
- [x] Velocity-Verlet integration.
- [x] Typed result records.

This is useful for transparent fixed-step algorithms and validation fixtures. It
is not an adaptive ODE/SDE solver stack.

### Probability, sampling, and random methods

- [x] Normal log-density, CDF, and PPF.
- [x] Lognormal log-density, CDF, and PPF.
- [x] Truncated-normal log-density, CDF, and PPF.
- [x] Finite power-law normalization, log-density, CDF, and PPF.
- [x] Smooth removable-singularity behavior through the logarithmic power-law
  limit.
- [x] Differentiable inverse-CDF sampling.
- [x] Stratified uniforms.
- [x] Explicit PRNG key streams and fold-in streams.
- [x] Seed manifests.
- [x] Systematic, stratified, and residual resampling.

These support population synthesis and Monte Carlo mechanics without turning
Jaxstro into a probabilistic-programming framework.

### Grids, conservative methods, and meshes

- [x] Logarithmic grids.
- [x] Geometric bin edges and centers.
- [x] Conservative rebinning of integrated values.
- [x] Structured one-dimensional meshes.
- [x] Cell and face geometry.
- [x] Neighbor stencils.
- [x] One-dimensional finite-volume divergence.
- [x] Cell-to-face averaging.
- [x] Conservative remapping.

This is a credible first finite-volume layer, but it remains intentionally
one-dimensional.

### Special functions

- [x] Planck functions in wavelength and frequency form.
- [x] Stable logarithmic Planck kernels.
- [x] Log-weight normalization.
- [x] Legendre, Chebyshev, Laguerre, and Hermite polynomial bases.

### Geometry, astrometry, and coordinates

- [x] Vector normalization and angular separation.
- [x] Rotation matrices.
- [x] Axis-angle quaternions.
- [x] Quaternion multiplication and rotation.
- [x] Rigid transforms, inversion, and composition.
- [x] Cartesian and spherical transformations.
- [x] Galactic and equatorial transformations.
- [x] Sky-tangent projection.
- [x] Parallax and proper motions.
- [x] Zenith and parallactic geometry.
- [x] Explicit singular-domain derivative behavior.

### Spatial algorithms

- [x] Morton encoding and decoding.
- [x] Deterministic hashing.
- [x] Particle-to-bin and linear-cell assignment.
- [x] Capacity-limited and exact bin filling.
- [x] Neighbor-stencil candidate gathering.
- [x] Approximate nearest-neighbor candidate generation.
- [x] Exact fixed-radius pair acceptance.
- [x] Explicit overflow, capacity, recall, symmetry, and cutoff contracts.

### Units and quantities

Two related surfaces currently coexist:

- [x] `jaxstro.units`: mature named unit systems, CGS default, astrophysical scale
  systems, photometric units, and explicit gravitational constants.
- [x] `jaxstro.quantity`: dimensional algebra, units, conversion, parsing,
  registries, serialization, unit bases, constants, dimension-aware math, and
  opt-in spectral, temperature–energy, and mass–energy equivalencies.

The quantity layer is implemented and extensively tested, but ecosystem-wide
adoption remains an evidence-gated decision.

### Spectra and atmospheres

- [x] Typed spectral coordinates, sampling semantics, flux semantics,
  provenance, and statuses.
- [x] Frequency and wavelength conversion.
- [x] $F_\lambda/F_\nu$ transformation.
- [x] Surface flux to luminosity and observer flux.
- [x] Point and conservative-bin resampling.
- [x] Prepared rectilinear and simplex interpolation stencils.
- [x] Exact-product atmosphere adapters for NewEra, BOSZ, Sonora, and TLUSTY.
- [x] Catalog discovery, topology selection, acquisition planning, artifact
  coverage, and fail-closed policy selection.
- [x] Host-side preparation followed by fixed-shape JAX evaluation.

This is unusually mature infrastructure for sharing atmosphere and spectral
mechanics without absorbing Fluxax's photometric interpretation.

### Parameters and constrained spaces

- [x] Selective Equinox PyTree-to-vector parameterization.
- [x] Static selection of free and fixed leaves.
- [x] Identity, exponential, softplus, and bounded sigmoid bijectors.
- [x] Analytic log-Jacobian terms.
- [x] Reconstruction of structured models from unconstrained vectors.

This is a parameter-coordinate bridge, not an inference engine.

### Scientific contracts, evidence, and provenance

- [x] Module and callable contracts.
- [x] Maturity, support, execution-boundary, transform, AD, failure, and
  evidence classifications.
- [x] Runtime export auditing.
- [x] Typed computational-evidence artifacts.
- [x] Metric and comparison records with units.
- [x] Environment records and content digests.
- [x] Deterministic JSON and Markdown.
- [x] Artifact hashing and method manifests.
- [x] Source-backed provenance cards.
- [x] Freshness and assertion-bearing evidence gates.

This **predict → compute → audit → state the warranted claim** architecture is
Jaxstro's most distinctive advantage over a miscellaneous utility library.

## What Jaxstro should become

The target identity is:

> Jaxstro is the reusable JAX-native substrate for scientifically trustworthy
> numerical computation and scientific machine learning: mathematical
> primitives, transforms, execution telemetry, dimensional semantics, and
> evidence—not domain policy or inference conclusions.

That is broader than “astrophysics helpers,” but still disciplined.

The ownership split should remain:

| Concern | Owner |
| --- | --- |
| Generic numerical algorithms and typed diagnostics | Jaxstro |
| Generic differentiable scientific-ML mechanics | Jaxstro |
| Units, dimensions, coordinates, operators, reproducibility | Jaxstro |
| Neural-network framework and optimizer implementation | Equinox and Optax |
| Adaptive differential-equation stack | Diffrax or another specialized owner |
| General probabilistic programming | NumPyro/BlackJAX |
| Posterior inference, SBI, calibration and refusal | Informax |
| Fisher geometry, identifiability and OED | Informax |
| Learned proposals or summaries whose meaning is inferential | Informax |
| Domain physics and scientific acceptance | Gravax, Progenax, Fluxax, Startrax, Stellax, and Nebulax |

Informax can consume Jaxstro's linear operators, differentiable solvers,
training mechanics, parameter transforms, and provenance without surrendering
ownership of what inference or experimental design means.

## Recommended additions

Unchecked items are proposals. Completing a checkbox requires the public API,
tests, theory, validation, contract-registry entry, and relevant evidence—not
only an implementation commit.

### Priority 1: a narrow `jaxstro.ml` scientific-ML substrate

This is the most direct missing layer if Jaxstro is to become the main ML
foundation.

- [ ] Add immutable `Standardizer`, `RobustScaler`, and `WhiteningTransform`
  PyTrees.
- [ ] Separate host fitting from JAX application with masks and named feature
  axes.
- [ ] Add deterministic train, validation, and test splits plus keyed
  fixed-shape batching.
- [ ] Define optimizer-agnostic `TrainState`, `TrainingTrace`, and
  `fit_fixed_steps` contracts.
- [ ] Add a `lax.scan` compiled training path with explicit executed masks and
  checkpoints.
- [ ] Add masked and weighted regression losses.
- [ ] Add scale-aware and relative-error losses.
- [ ] Add Gaussian NLL and heteroscedastic-regression mechanics.
- [ ] Add gradient clipping and finite-update rejection with telemetry.
- [ ] Record dataset, normalization, seed, model-structure, and checkpoint
  manifests.
- [ ] Add reproducibility and data-leakage audits.
- [ ] Add Jacobian, divergence, curl, Laplacian, and residual construction for
  scientific ML.
- [ ] Evaluate optional neural-tangent and Gauss–Newton operator diagnostics.

It should not include a model zoo. Equinox already owns neural modules; Optax
owns optimizers. Jaxstro should own the repeatable, audited scientific execution
around them.

### Priority 2: matrix-free iterative linear solvers

- [ ] Add conjugate gradient for symmetric positive-definite systems.
- [ ] Add MINRES for symmetric indefinite systems after a real consumer exists.
- [ ] Add GMRES for general systems after a real consumer exists.
- [ ] Define reusable preconditioner protocols.
- [ ] Return fixed-shape iteration traces.
- [ ] Record residual, conditioning, breakdown, and stagnation statuses.
- [ ] Gate derivatives through `jax.lax.custom_linear_solve` only where the
  mathematical and numerical assumptions are validated.
- [ ] Add dense and analytic parity cases plus independent finite-difference
  gates.

This would unlock larger inverse problems, implicit layers, stellar-structure
systems, PDE discretizations, and scalable Informax geometry without requiring
every project to invent solver telemetry.

### Priority 3: vector nonlinear systems and fixed points

Build this only around concrete consumers, but it is a natural extension of the
scalar-root work.

- [ ] Identify and validate at least one downstream vector-root or fixed-point
  consumer before generalizing the API.
- [ ] Add Newton–Krylov or damped Newton for $F(\mathbf{x})=0$.
- [ ] Evaluate safeguarded fixed-point acceleration such as Anderson
  acceleration.
- [ ] Add trust-region or line-search globalization.
- [ ] Return typed primal and linear-solve certificates.
- [ ] Add strictly gated implicit derivatives.
- [ ] Surface uniqueness, conditioning, residual, and branch assumptions as
  evidence.

Potential science includes hydrostatic stellar structure, equilibrium chemistry,
coupled feedback balances, differentiable steady states, and constrained orbital
solutions.

### Priority 4: quasi-Monte Carlo and variance reduction

- [ ] Add scrambled Sobol sequences.
- [ ] Evaluate Halton sequences only with their limitations documented.
- [ ] Add Latin-hypercube sampling.
- [ ] Add antithetic sampling.
- [ ] Add control-variate primitives.
- [ ] Add effective sample size and weight-degeneracy diagnostics.
- [ ] Add replicated-scramble uncertainty estimates.
- [ ] Record seed and sequence provenance.

This would benefit Progenax population integration, Fluxax synthetic surveys,
simulation ensembles, and Informax nested expectation calculations.

### Priority 5: adaptive quadrature

- [ ] Add Gauss–Kronrod pairs.
- [ ] Add error-estimated adaptive interval subdivision.
- [ ] Define fixed-capacity or explicitly staged execution suitable for JAX.
- [ ] Return typed exhaustion and nonfinite-integrand outcomes.
- [ ] Separate value and derivative contracts.
- [ ] Support vector-valued integrands and batched domains.

Avoid building a competing adaptive ODE ecosystem. Diffrax is the better owner
for general event-driven ODE and SDE solving.

### Priority 6: signal and time-series primitives

- [ ] Define FFT normalization conventions.
- [ ] Add convolution and correlation with explicit boundary policy.
- [ ] Add window functions and equivalent-noise bandwidth.
- [ ] Add power and cross-spectral density.
- [ ] Add phase and time-delay estimation.
- [ ] Evaluate irregular-sampling primitives only after a concrete science use
  case exists.
- [ ] Add unit-aware frequency and time-axis handling.

These would support variability, spectra, detector response, synthetic
observations, and time-domain astronomy.

### Priority 7: generic uncertainty propagation

Keep inferential interpretation in Informax, but Jaxstro can provide mechanics
for:

- [ ] First-order delta-method propagation.
- [ ] Covariance pushforward through Jacobians.
- [ ] Unscented and sigma-point transforms.
- [ ] Ensemble propagation with deterministic keying.
- [ ] Sensitivity decomposition.
- [ ] Conditioning and rank diagnostics.

Jaxstro would return mathematical propagation results; Informax would decide
whether they justify an inferential claim.

### Priority 8: multidimensional field methods

Only add these once Stellax, Nebulax, or another consumer supplies concrete
contracts.

- [ ] Identify at least two consumers and ratify their shared topology and
  conservation boundary.
- [ ] Add structured two- and three-dimensional mesh geometry.
- [ ] Add gradient, divergence, curl, and Laplacian stencils.
- [ ] Add conservative flux divergence.
- [ ] Define explicit boundary-condition objects.
- [ ] Add prolongation and restriction.
- [ ] Add matrix-free elliptic operators.

Do not grow into a full CFD framework without multiple consumers and strong
conservation evidence.

## What not to add

Avoid turning Jaxstro into a collection of inferior replacements:

- No homegrown neural-network framework; use Equinox.
- No full optimizer zoo; use Optax or specialized optimization packages.
- No general MCMC, VI, NPE, SBI, or posterior orchestration; that belongs in
  Informax.
- No Fisher-OED criteria, calibration gates, claim/refusal policy, or learned
  experimental-design proposals; those belong in Informax.
- No full adaptive ODE/SDE stack; use Diffrax.
- No generic duplicate of every NumPyro or Distrax distribution.
- No domain-specific timestep, particle, photometry, stellar-evolution, or
  atmosphere-acceptance policy.
- No algorithm merely because SciPy has it. Add methods when at least two
  projects need the same mathematical contract or when one foundational need is
  exceptionally general.

## Build checklist

This is the recommended order. The capability-level checklists above define the
work inside each rung.

- [ ] Generate the callable-level JAX transform and maturity matrix.
- [ ] Decide the ecosystem future of `jaxstro.quantity` from downstream parity,
  performance, serialization, ergonomics, and migration evidence.
- [ ] Add the minimal `jaxstro.ml` vertical slice: normalization, deterministic
  batching, fixed-step training, typed trace, and checkpoint provenance.
- [ ] Add conjugate gradient plus the linear-solver protocol and evidence model.
- [ ] Validate one real vector-root or fixed-point consumer before generalizing
  the solver.
- [ ] Add scrambled Sobol sampling and replicated uncertainty evidence.
- [ ] Add Gauss–Kronrod adaptive quadrature.
- [ ] Add signal-processing primitives driven by Fluxax or time-domain use.
- [ ] Expand meshes only when multiple field-based packages can define the
  shared boundary.
- [ ] Generate downstream symbol-to-project compatibility records throughout.

That sequence makes Jaxstro genuinely useful as the daily numerical and ML
foundation without sacrificing the thin-foundation discipline that currently
makes it trustworthy.

## Definition of done for a roadmap item

A checkbox moves to complete only when the relevant layer includes:

1. a mathematical or semantic contract;
2. a public API with typed boundary and failure behavior;
3. supported JAX transforms and cost semantics;
4. independent validation rather than self-consistency alone;
5. reproducible artifacts where metrics matter;
6. accessible teaching that states what the evidence does not prove;
7. downstream adoption evidence when reuse justifies ownership; and
8. an updated contract registry, validation index, scorecard, and roadmap.

The highest-impact direction is not adding the most algorithms. It is making the
existing breadth uniformly inspectable from equation to execution to claim,
while admitting new primitives only when reuse and evidence justify them.

## Evidence and companion pages

- [](../40-api/index.md) — current public modules and execution boundaries.
- [](../40-api/contracts.md) — generated ownership, maturity, transform, AD,
  failure, and evidence contracts.
- [](../60-validation/index.md) — independent validation anchors.
- [](./package-assessment-scorecard.md) — living grades and promotion gates.
- [](./sota-assessment.md) — ranked infrastructure and pedagogy investments.
- [](./numerical-methods-roadmap.md) — completed first-generation numerical
  expansion checklist.
- [](../20-architecture/science-general-vision.md) — science-general ecosystem
  boundary.
