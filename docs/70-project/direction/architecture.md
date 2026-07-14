---
title: Architecture
description: >-
  The shape of the software - JAX-native functional design, the astro-first but
  science-general boundary, the units policy, and the one-way dependency rule
  that keeps the foundation thin.
---

This section explains *why* the software has its current shape: JAX-native
functional and PyTree design, explicit unit boundaries, one-way package
dependencies, evidence ownership, and a deliberately thin foundation. Where the
[theory section](../../20-methods/methods.md) covers mathematical method contracts,
this section covers package structure and responsibility.

## Foundation boundary

`jaxstro` is the ecosystem foundation. Its directly importable modules are
`units`, `quantity`, `constants`, `astrometry`, `coords`, `geometry`,
`numerics`, `spatial`, `params`, `atmospheres`, `provenance`, `testing`, and
`jaxconfig`. It does not own simulations, survey rendering, stellar evolution,
or inference workflows. Domain packages depend on `jaxstro`; `jaxstro` never
imports package-specific code back from them. That one-way rule keeps foundation
changes auditable before they reach Gravax, Progenax, Fluxax, Startrax, or later
packages.

:::{figure} ./figures/jaxstro-foundation.webp
:name: fig-jaxstro-foundation
:alt: One-way package dependency diagram with downstream astronomy packages depending on the jaxstro foundation

The package boundary is an ownership diagram, not a runtime data-flow graph.
Arrows point from a consumer to its dependency. The highlighted host-side band
marks discrete selection and indexing work that is intentionally outside a
differentiable array kernel.
:::

The keystone decision is [](../decisions/0001-thin-foundation-posture.md).
The dependency and packaging decisions are recorded in
[](../decisions/0002-adopt-equinox-foundation.md),
[](../decisions/0003-standalone-uv-hatchling-project.md), and
[](../decisions/0010-ecosystem-config-architecture.md).

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

`jaxstro.units` remains the current canonical ecosystem contract. The
`jaxstro.quantity` layer is implemented: it provides concrete units,
dimension-safe arithmetic, exact parser/serialization, role-aware bases,
versioned constants, and explicit equivalencies. However, ecosystem adoption and any replacement cutover remain deferred.
The representation contract is documented in
[](../../30-representations/units-quantities/quantity-system.md), but that page is an
evaluation surface rather than a migration instruction. It extends the decision recorded in
[](../decisions/0006-build-own-quantity-not-unxt.md).

## Numerical shape

Array kernels are JAX-native and use `jit`, `vmap`, or `grad` only where each
public method's contract supports that transform. Fixed-shape numerical
iterations and sanitized branches protect traced execution, while tests check
forward values and gradients on the smooth domains named by each method. This
does not make discrete indexing, sorting, catalog lookup, or every branch
differentiable. The detailed mathematical contracts live in
[](../../20-methods/methods.md), and quantitative checks live in
[](../../60-validation/validation.md).

## Units policy

`jaxstro` defaults to CGS through `DEFAULT_UNITS`, because it is the
domain-agnostic base layer. Domain packages choose their own package-level
defaults. Core APIs either accept explicit units or explicit physical constants;
convenience wrappers may resolve `units=None` to the package default.
[](../decisions/0007-cgs-as-default-units.md) records this policy.

## Provenance ownership

Two provenance surfaces answer different questions. The
[source-backed provenance cards](../../50-api/research-infrastructure/source-provenance/source-provenance.md) connect constants,
transforms, and capability boundaries to references, code symbols, validation
paths, and evidence states. They describe why a public scientific claim is
trustworthy.

The [runtime artifact manifests](../../50-api/research-infrastructure/provenance.md) instead record what a particular
computation consumed: configuration, environment, seeds, hashes, and method
identifiers. They make a run reproducible. A card does not replace a manifest,
and a manifest does not establish the scientific source behind a constant or
transform.

## Differentiable and discrete boundaries

`spatial` indexing and candidate construction, together with atmosphere catalog
selection, are host-side, discrete preprocessing. Their selected arrays can feed
JAX-native kernels, but the selection decisions themselves are not advertised as
differentiable. This division keeps runtime shapes and ownership explicit rather
than hiding Python decisions inside traced code.

## Data-layer boundary

Large third-party scientific data is not vendored into the package. Foundation
data adapters may expose local discovery, provenance, processed-artifact
validation, and catalog-first runtime selection, but raw products remain in user
cache locations or explicitly gitignored local mirrors. The first large example
is `jaxstro.atmospheres`: it can process and index local NewEra, BOSZ, Sonora,
and TLUSTY atmosphere spectra, while filters, photometry, bolometric
corrections, survey rendering, and physical interpretation remain downstream.
Atmosphere support remains in progress: the capability map distinguishes staged
data, processed artifacts, and implemented runtime backends rather than treating
every catalog entry as available interpolation support.

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
[](../../30-representations/spectra-atmospheres/atmosphere-capabilities.md). It explains
which libraries are processed,
which have runtime backends, and why TLUSTY uses ragged frequency-grid subgroups.

The runtime boundary is documented in
[](../../30-representations/spectra-atmospheres/spectra-data-architecture.md). It
defines the host-side `AtmosphereQuery -> PreparationResult` path, the generic
`jaxstro.spectra` owner, and fixed-topology JAX evaluation. It also explains why
surface versus observer flux, `F_lambda` versus `F_nu`, and point versus bin
sampling must remain explicit. Filters, photometry, instruments, images, and
survey semantics remain downstream in Fluxax.
