---
title: jaxstro
subtitle: The differentiable foundation of the astrophysics ecosystem
description: >-
  jaxstro is the lightest node in the differentiable-astrophysics dependency graph:
  units, constants, coordinates, and AD-safe numerics that every package above it reuses.
---

Everything else in the ecosystem — gravax, progenax, fluxax, and the planned
startrax and stellax — depends on jaxstro, so jaxstro depends on almost nothing.
It owns the **generic, differentiable, dependency-light** pieces: physical
constants in CGS, unit systems, coordinate transforms, and numerics that survive
`jax.grad`. It deliberately refuses to absorb solver libraries; those belong one
layer up (see [](#two-doors) and [](./30-decisions/0001-thin-foundation-posture.md)).

This site is the package's single source of truth. It is written for two readers
at once: a **new graduate student** meeting differentiable scientific computing for
the first time, and **future-you** trying to remember *why* a function rounds the
way it does. The theory pages teach the methods; the reference pages let you look
up the call signature; the decision log records every choice and its trade-offs.

(two-doors)=
## Two doors in

::::{grid} 1 1 2 2

:::{card} Learn the methods
:link: ./10-theory/index.md

Start here if you want to understand *how to write numerics that differentiate
cleanly*. The theory section opens with a ten-principle thesis on AD-safe
scientific computing, then fans out into worked method pages — root-finding,
Newton–Cotes integration, quadrature, and more.
:::

:::{card} Look up the API
:link: ./40-api/index.md

Start here if you already know what you want and need the signature. The API
reference enumerates every public module — `units`, `constants`, `coords`,
`numerics`, `spatial`, `params`, `testing`, `jaxconfig` — and links each symbol
back to the theory it implements.
:::

::::

## Routed paths

- **New here?** Read [](./00-getting-started/index.md) first — it installs the
  package, turns on float64, and walks one worked example end to end. Then follow
  the bridge into [](./10-theory/index.md).
- **Porting code from a sibling package?** The [decision log](./30-decisions/index.md)
  explains the hoists and reconciliations (`cumulative_trapz`, Newton-PPF, the
  quadrature factory) that changed call sites.
- **Auditing a number?** Constants carry provenance to CODATA 2018 / IAU 2015
  in [](./40-api/index.md); the [validation](./60-validation/index.md) section is
  where claims meet their tests.

## What jaxstro is *not*

It is not a simulation package and not a solver. It holds no domain physics — no
stellar tracks, no N-body integrator, no IMF. Those live in the packages above it.
jaxstro stays at the bottom of the dependency graph and stays light, because
everything depends on it.
