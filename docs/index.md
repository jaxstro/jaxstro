---
title: jaxstro
subtitle: Evidence-first JAX infrastructure for differentiable science
description: >-
  jaxstro is an astro-first, science-general foundation: units, constants,
  coordinates, AD-safe numerics, parameter bridges, spatial utilities, and
  validation tools for differentiable scientific software.
---

jaxstro is the evidence-first foundation under the differentiable astrophysics
ecosystem and a compact scientific JAX standard library. Everything else in the
ecosystem — gravax, progenax, fluxax, and the planned startrax and stellax —
depends on jaxstro, so jaxstro depends on almost nothing.

Astronomy is the proving ground. The reusable product is broader: **generic,
differentiable, dependency-light scientific infrastructure** with explicit units,
clear boundary behavior, and validation hooks that make numerical claims
auditable. jaxstro owns the primitives that should be shared across scientific
packages: physical constants in CGS, unit systems, coordinate transforms,
AD-safe interpolation and integration, local basis functions, small dense linear
algebra, sampling helpers, parameter-vector bridges, and trust reports. It
deliberately refuses to absorb solver libraries; those belong one layer up (see
[](#two-doors), [](./20-architecture/science-general-vision.md), and
[](./30-decisions/0001-thin-foundation-posture.md)).

This site is the package's single source of truth. It is written for two readers
at once: a **new graduate student** meeting differentiable scientific computing for
the first time, and **future-you** trying to remember *why* a function rounds the
way it does. The theory pages teach the methods; the reference pages let you look
up the call signature; the decision log records every choice and its trade-offs.

(two-doors)=
## Three doors in

::::{grid} 1 1 3 3

:::{card} Learn the methods
:link: ./10-theory/index.md

Start here if you want to understand *how to write numerics that differentiate
cleanly*. The theory section opens with a ten-principle thesis on AD-safe
scientific computing, then fans out into worked method pages — root-finding,
Newton–Cotes integration, quadrature, and more.
:::

:::{card} Audit the evidence
:link: ./60-validation/index.md

Start here if you want to know what makes a numerical claim trustworthy. The
validation section connects public APIs to finite-difference audits, method
evidence anchors, coverage reports, and deterministic trust summaries.
:::

:::{card} Look up the API
:link: ./40-api/index.md

Start here if you already know what you want and need the signature. The API
reference enumerates every public module — `units`, `constants`, `coords`,
`numerics`, `spatial`, `params`, `testing`, `jaxconfig` — and links each symbol
back to the theory it implements.
:::

::::

## Astro-first, not astro-only

jaxstro is useful outside astronomy wherever a project needs JAX-native
scientific primitives whose gradient behavior is part of the contract. If you are
building differentiable models in physics, geoscience, instrumentation,
engineering, statistics, or simulation-adjacent inference, the same foundation
rules apply: name units and domains, avoid hidden Python-side state, keep shapes
static where JAX needs them, validate automatic differentiation against
independent checks, and separate generic numerics from domain interpretation.

The long-term vision is not to become a replacement for SciPy, NumPyro, Diffrax,
or domain simulators. It is to be a small, rigorous base layer that makes those
larger systems easier to trust when an astronomy package, or any differentiable
science package, needs shared constants, transformations, numerical kernels, and
evidence.

## Routed paths

- **New here?** Read [](./00-getting-started/index.md) first — it installs the
  package, turns on float64, and walks one worked example end to end. Then follow
  the bridge into [](./10-theory/index.md).
- **Evaluating the broader package vision?** Read
  [](./20-architecture/science-general-vision.md) for the module boundary, the
  non-astronomy value proposition, and the checklist for future core modules.
- **Planning quantity-aware APIs?** Read
  [](./20-architecture/quantity-system.md) for the planned `jaxstro.quantity`
  design: concrete units, dimensional arithmetic, parser/serialization,
  role-aware bases, constants, equivalencies, and migration from `jaxstro.units`.
- **Porting code from a sibling package?** The [decision log](./30-decisions/index.md)
  explains the hoists and reconciliations (`cumulative_trapz`, Newton-PPF, the
  quadrature factory) that changed call sites.
- **Working with atmosphere spectra?** Start with
  [](./20-architecture/atmosphere-capabilities.md) for the local dataset matrix,
  processed-artifact status, and the boundary between jaxstro spectra and
  downstream photometry.
- **Auditing a number?** Constants carry provenance to CODATA 2018 / IAU 2015
  in [](./40-api/index.md); the [validation](./60-validation/index.md) section is
  where claims meet their tests.

## What jaxstro is *not*

It is not a simulation package, not an inference framework, not a generic SciPy
clone, and not an all-purpose solver stack. It holds no domain physics — no
stellar tracks, no N-body integrator, no IMF, no filters or photometry semantics.
Those live in the packages above it. jaxstro stays at the bottom of the
dependency graph and stays light, because everything depends on it.
