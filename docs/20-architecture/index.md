---
title: Architecture
description: >-
  The shape of the software — JAX-native functional design, the astro-first but
  science-general boundary, the units policy, and the one-way dependency rule
  that keeps the foundation thin.
---

This section will tell the narrative "why" of the software's *shape* — the
JAX-native functional and PyTree design, the explicit-units policy, the one-way
dependency rule, and the thin-foundation posture — as connected prose that cites
the decisions behind each. Where the [theory section](../10-theory/index.md)
covers the math of the methods, this section covers the structure of the package.

## Foundation boundary

`jaxstro` is the ecosystem foundation: constants, unit systems, coordinate
transforms, AD-safe numerics, spatial utilities, parameter-vector bridges, and
testing harnesses. It does not own simulations, survey rendering, stellar
evolution, or inference workflows. Downstream packages depend on `jaxstro`; the
dependency arrow does not point back into package-specific code. This keeps
`import jaxstro` small and makes foundation changes auditable before they reach
Fluxax, Progenax, Gravax, Startrax, or later packages.

The keystone decision is [](../30-decisions/0001-thin-foundation-posture.md).
The dependency and packaging decisions are recorded in
[](../30-decisions/0002-adopt-equinox-foundation.md),
[](../30-decisions/0003-standalone-uv-hatchling-project.md), and
[](../30-decisions/0010-ecosystem-config-architecture.md).

## Astro-first, science-general

The package should be marketable as **evidence-first JAX infrastructure for
differentiable science**. Astronomy supplies the pressure tests: physical units,
awkward coordinate transforms, tabulated spectra, stiff numerical ranges, and
gradients that must be trusted by downstream inference. The abstractions that
survive those tests are useful well beyond astronomy.

New foundation modules should pass four checks before they belong here:

1. **Generic across domains.** The primitive is useful without knowing about
   stars, galaxies, filters, surveys, or a specific simulator.
2. **JAX-native by construction.** The public runtime path composes with `jit`,
   `vmap`, and `grad` where differentiation is part of the contract.
3. **Explicit about boundaries.** Units, valid domains, clamping, saturation,
   static arguments, and non-differentiable preprocessing are named rather than
   hidden.
4. **Backed by evidence.** The module has focused unit tests, transform tests,
   finite-difference or analytic validation where relevant, and documentation
   that explains failure modes.

The broader product vision and future-module map are in
[](./science-general-vision.md).

## Quantity architecture

`jaxstro.quantity` is the planned unit-aware value layer: concrete unit objects,
dimension-safe arithmetic, exact parser/serialization, role-aware astrophysical
bases, versioned constants, and explicit equivalencies. It is additive to the
existing `jaxstro.units` API so downstream packages can migrate deliberately.
The approved architecture is in [](./quantity-system.md), and it extends the
decision recorded in
[](../30-decisions/0006-build-own-quantity-not-unxt.md).

## Numerical shape

Public numerical helpers are JAX-native and are designed for `jit`, `vmap`, and
`grad`. Iterative primitives use fixed-shape control flow; AD-sensitive branches
sanitize dangerous operations before selecting values; tests check both forward
values and gradient behavior. The detailed mathematical contracts live in
[](../10-theory/index.md).

## Units policy

`jaxstro` defaults to CGS through `DEFAULT_UNITS`, because it is the
domain-agnostic base layer. Domain packages choose their own package-level
defaults. Core APIs either accept explicit units or explicit physical constants;
convenience wrappers may resolve `units=None` to the package default.
[](../30-decisions/0007-cgs-as-default-units.md) records this policy.

## Data-layer boundary

Large third-party scientific data is not vendored into the package. Foundation
data adapters may expose local discovery, provenance, processed-artifact
validation, and catalog-first runtime selection, but raw products remain in user
cache locations or explicitly gitignored local mirrors. The first large example
is `jaxstro.atmospheres`: it can process and index local NewEra, BOSZ, Sonora,
and TLUSTY atmosphere spectra, while filters, photometry, bolometric
corrections, survey rendering, and physical interpretation remain downstream.

## Test layers

The suite is organized by risk:

| Layer | Purpose | Examples |
| --- | --- | --- |
| Unit | Local functional contracts and edge cases | constants, units, spatial binning, atmosphere file parsing |
| Integration | Cross-module and transform compatibility | grad-audit API, parity checks, package import contracts |
| Validation | Numerical truth checks | FD-vs-AD audits, convergence and derivative checks |

Every release-facing claim should point either to the test that enforces it or to
a decision record that explains why the boundary exists.

## Spectra data architecture

The local atmosphere capability map is in
[](./atmosphere-capabilities.md). It explains which libraries are processed,
which have runtime backends, and why TLUSTY uses ragged frequency-grid subgroups.

The runtime boundary is documented in [](./spectra-data-architecture.md). It
defines the shared `AtmosphereParams -> SpectrumResult` interface, the split
between catalog-first host-side selection and JAX-side prepared interpolation,
and the dataset ownership rule that keeps filters, photometry, bolometric
corrections, and survey semantics downstream until a genuinely shared lower-level
abstraction exists.
