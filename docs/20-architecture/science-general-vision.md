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

## Full vision

jaxstro should become the package you reach for when you need **JAX-native
scientific building blocks with proof-carrying behavior**:

- **Numerical kernels** that are small, composable, and documented in terms of
  their gradient and boundary contracts.
- **Physical foundations** — constants, unit systems, coordinate transforms, and
  vector geometry — that are explicit rather than hidden in downstream code.
- **Parameter and data bridges** that make scientific models easier to connect to
  optimizers, samplers, and validation tools without becoming an optimizer or a
  sampler itself.
- **Evidence artifacts** that connect API claims to tests, finite-difference
  checks, generated reports, source hashes, and known limitations.

That makes jaxstro useful inside astronomy as the shared base for Gravax,
Progenax, Fluxax, Startrax, Stellax, and atmosphere tooling. It also makes it
useful outside astronomy for teams building differentiable models in physics,
geoscience, engineering, remote sensing, instrumentation, simulation calibration,
and scientific machine learning.

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

## Future core modules

The following modules are natural extensions of the current foundation. They are
listed as a product checklist, not a promise that every item should land in one
pull request.

```{list-table} Candidate science-general modules
:header-rows: 1
:label: tbl-science-general-modules

* - Module
  - Why it belongs
  - First-slice scope
* - `jaxstro.numerics.optimization`
  - Many scientific models need differentiable objective helpers before they
    need a full optimizer stack.
  - Robust losses, objective summaries, line-search utilities, convergence
    diagnostics, and evidence tests; no replacement for Optax.
* - `jaxstro.numerics.ode`
  - Fixed-step integration is a common scientific primitive and a good fit for
    explicit gradient contracts.
  - Euler, RK2/RK4, leapfrog or velocity-Verlet, fixed-step scan APIs, and
    energy/convergence validation; no adaptive solver in the first slice.
* - `jaxstro.numerics.operators`
  - Scientific code often needs matrix-free linear algebra without committing to
    a large sparse framework.
  - A small LinearOperator PyTree protocol with dense, diagonal, scaled, sum,
    product, and block operators plus `matvec`/`rmatvec` tests.
* - `jaxstro.numerics.distributions`
  - Inference-adjacent packages need stable probability kernels, but full
    probabilistic programming belongs elsewhere.
  - Logpdf, CDF, inverse-CDF, truncated distributions, stable normalization, and
    sampling hooks for common generic families.
* - `jaxstro.geometry`
  - Coordinates already exist; reusable vector geometry and rigid transforms are
    the next generic layer.
  - Rotations, quaternions, angular distances, projection helpers, and transform
    composition with clear conventions.
* - `jaxstro.numerics.autodiff`
  - Downstream packages repeatedly need gradient diagnostics and curvature
    products that are not model-specific.
  - JVP/VJP/HVP helpers, Gauss-Newton products, Fisher-style products, and
    finite-difference cross-checks.
* - `jaxstro.provenance`
  - The package already treats evidence as part of the API; provenance deserves a
    shared runtime representation.
  - Artifact hashes, environment snapshots, method manifests, and deterministic
    JSON/Markdown rendering for validation reports.
* - `jaxstro.numerics.random`
  - Scientific simulations need reproducible random streams and resampling
    methods without hiding PRNG-key flow.
  - Key-stream helpers, stratified/systematic/residual resampling, seed
    manifests, and shape-stable APIs.
* - `jaxstro.numerics.meshes`
  - Many scientific workflows discretize domains before they simulate,
    interpolate, or conserve quantities.
  - Structured mesh helpers, cell/face geometry, finite-volume stencils, and
    conservative remapping in one dimension before higher-rank extensions.
```

## Deferred deliberately

`jaxstro.units.quantity` is the most important missing user-facing abstraction,
but it should not be implemented casually. A quantity layer changes how units
flow through almost every public API, so it needs separate design work around
tracing behavior, PyTree semantics, runtime overhead, error messages, and
interoperability with the existing `UnitSystem` policy.

Until that design is settled, jaxstro should continue to expose explicit unit
systems and documented unit conventions rather than introduce a partial quantity
object.
