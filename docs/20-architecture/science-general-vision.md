---
title: Astro-first, science-general vision
description: >-
  The product vision for jaxstro as evidence-first JAX infrastructure that is
  proven in astronomy but useful across differentiable scientific software.
---

jaxstro is named from astronomy, but its most durable value is more general:
it is a small, evidence-first foundation for scientific software that needs JAX,
automatic differentiation, explicit units and domains, and reproducible
validation.

Astronomy remains the flagship use case. It is a demanding one: constants must be
sourced, units must be explicit, coordinate transforms have real conventions,
spectra are tabulated on uneven grids, and downstream packages need gradients
that are not merely available but trustworthy. A primitive that works under those
constraints is usually useful outside astronomy too.

## Delivered vision

jaxstro provides **JAX-native scientific building blocks with proof-carrying
behavior**:

- **Numerical kernels** that are small, composable, and documented in terms of
  their gradient and boundary contracts.
- **Physical foundations** — constants, unit systems, coordinate transforms, and
  vector geometry — that are explicit rather than hidden in downstream code.
- **Parameter and data bridges** that make scientific models easier to connect to
  optimizers, samplers, and validation tools without becoming an optimizer or a
  sampler itself.
- **Evidence artifacts** that connect API claims to tests, finite-difference
  checks, generated reports, source hashes, and known limitations.

Within astronomy, Gravax, Progenax, Fluxax, and Startrax build on the shared
foundation. Startrax is active; Stellax remains planned. Outside astronomy, the
same primitives may be useful in differentiable physics, geoscience,
instrumentation, simulation calibration, and scientific machine learning when
their documented unit, shape, and transform contracts fit the application.

The [foundation ownership figure](./index.md#fig-jaxstro-foundation) shows the
one-way package boundary. It is an ownership map, not a claim that every function
is differentiable: kernels compose with `jit`, `vmap`, or `grad` only where each
method documents that transform.

## Product boundary

jaxstro should stay small enough that users can audit it. It should not become a
replacement for SciPy, NumPyro, Diffrax, Equinox, Optax, or domain-specific
simulation libraries. Instead, it should provide the shared primitives that make
those systems easier to use safely in scientific code.

The boundary is:

- Put **generic scientific primitives** in jaxstro.
- Put **domain interpretation** in downstream packages.
- Put **large solvers, samplers, and model-specific workflows** in specialized
  libraries or downstream packages.
- Put **evidence and provenance hooks** close to the primitive whose behavior
  they support.

## Delivered foundation map

The modules formerly described here as candidates are installed today. This
table records their deliberately bounded first release rather than presenting
delivered work as a future roadmap.

```{list-table} Delivered science-general modules
:header-rows: 1
:label: tbl-science-general-modules

* - Module
  - Delivered scope
  - Deliberate boundary
* - `jaxstro.numerics.optimization`
  - Robust losses, objective and convergence summaries, gradient/step norms, and
    fixed-iteration Armijo backtracking.
  - Objective helpers, not an optimizer stack or replacement for Optax. See
    [](../20-methods/change-constraints-evolution/optimization.md).
* - `jaxstro.numerics.ode`
  - Euler, midpoint/RK2, RK4, fixed-step scan integration, and velocity-Verlet.
  - Fixed-step methods only; adaptive solver policy stays outside the
    foundation. See [](../20-methods/change-constraints-evolution/ode.md).
* - `jaxstro.numerics.operators`
  - A PyTree operator protocol with dense, diagonal, scaled, sum, product,
    transpose, and block-diagonal implementations.
  - Small matrix-free composition helpers, not a sparse storage or iterative
    solver framework. See [](../20-methods/linear-structure/operators.md).
* - `jaxstro.numerics.distributions`
  - Log-density, CDF, and inverse-CDF kernels for normal, lognormal, power-law,
    and truncated-normal families.
  - Explicit-support kernels, not distributions-as-objects or probabilistic
    programming. See [](../20-methods/probability-sampling/distributions.md).
* - `jaxstro.geometry`
  - Vector normalization, angular distance, axis-angle rotation, quaternions,
    and rigid-transform composition/inversion.
  - Generic geometry without domain coordinate interpretation. See
    [](../30-representations/geometry-coordinates/geometry.md).
* - `jaxstro.numerics.autodiff`
  - JVP, VJP, HVP, Gauss–Newton, and empirical-Fisher products.
  - Thin products over JAX primitives, not a differentiation framework or a
    promise that arbitrary caller functions are smooth. See
    [](../20-methods/change-constraints-evolution/autodiff.md).
* - `jaxstro.provenance`
  - Artifact hashes, environment snapshots, method manifests, and deterministic
    JSON/Markdown rendering.
  - Runtime reproducibility records; scientific source claims belong to the
    separate provenance-card registry. See [](./provenance.md).
* - `jaxstro.numerics.random`
  - Explicit key streams, seed manifests, and systematic, stratified, and
    residual resampling.
  - Key flow stays visible; the module does not own a simulation runtime. See
    [](../20-methods/probability-sampling/random.md).
* - `jaxstro.numerics.meshes`
  - Structured 1D edges, cell/face geometry, neighbors, divergence, face
    averaging, and conservative remapping.
  - One-dimensional structured meshes only. See [](../20-methods/discrete-space/meshes.md).
```

## Units and quantity status

`jaxstro.units` remains the current ecosystem contract. `jaxstro.quantity` is implemented
and available for evaluation, but ecosystem adoption and any replacement cutover remain deferred.
The quantity architecture therefore documents a real implementation without
instructing sibling packages to migrate. See
[](../30-representations/units-quantities/quantity-system.md).

## Admission criteria for future work

A possible addition belongs in jaxstro only when evidence supports all four
conditions:

1. **It is domain-general.** The primitive remains useful without importing
   stars, galaxies, instruments, surveys, or a particular simulator.
2. **Its ownership lowers duplication.** At least two consumers need the same
   lower-level contract, or one consumer supplies unusually strong evidence that
   the abstraction is genuinely foundational.
3. **Its transform boundary is explicit.** Supported JAX transforms, static
   arguments, valid domains, and discrete preprocessing are named and tested.
4. **Its evidence can live beside it.** Tests, validation anchors, provenance,
   and known limitations can be maintained without pulling domain workflows into
   the core.

Passing these checks admits a proposal for design and evidence work; it is not a
promise that the feature will be accepted. The thin-foundation posture remains
the controlling decision.
