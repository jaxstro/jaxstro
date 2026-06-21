---
title: Architecture
description: >-
  The shape of the software — JAX-native functional design, the units policy, and
  the one-way dependency rule that keeps the foundation thin.
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
data adapters may expose local discovery and metadata indexing, but raw products
remain in user cache locations or explicitly gitignored local mirrors. The first
example is `jaxstro.atmospheres`: it can index local PHOENIX/NewEra files, while
spectral interpolation, filter projection, survey rendering, and photometric
semantics remain later backend or downstream-package responsibilities.

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

The first large-data runtime boundary is documented in
[](./spectra-data-architecture.md). It defines the shared
`AtmosphereParams -> SpectrumResult` interface, the split between host-side
processed-artifact loading and JAX-side prepared interpolation, and the dataset
ownership rule that keeps filters, photometry, bolometric corrections, and survey
semantics downstream until a genuinely shared lower-level abstraction exists.
