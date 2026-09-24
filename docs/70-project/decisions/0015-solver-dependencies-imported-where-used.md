---
title: "ADR 0015 - Solver dependencies are required but imported only where used"
description: >-
  diffrax, optimistix, and sympy are required runtime dependencies; only the
  modules that use them may import them, so the units, constants, and coordinate
  surfaces load none of them.
id: 0015
date: 2026-09-24
status: accepted
supersedes: 0001
decided_by: user
last_read: 0
---

# 0015 - Solver dependencies are required but imported only where used

## Context

[](./0001-thin-foundation-posture.md) refused solver libraries so that a units-only
consumer would not inherit diffrax and optimistix. Two later changes added three
runtime dependencies without revising it:

- diffrax and optimistix, for the adaptive ODE solve and event root of the Lane-Emden
  solver, shared by hydrax and progenax (`e0f8da2`, 2026-07-24);
- sympy, which renders the LaTeX that every equation-registry record emits
  (`e7fe05e`, 2026-08-29).

Measured on 2026-09-24 with a fresh interpreter per import:

| Import | Loads diffrax and optimistix | Loads sympy |
| --- | --- | --- |
| `jaxstro`, `jaxstro.units`, `jaxstro.constants` | no | no |
| `jaxstro.coords`, `jaxstro.numerics`, `jaxstro.quad` | yes | no |
| `jaxstro.registry` | no | no |

The coordinate and quadrature imports load the solvers because
`jaxstro/numerics/__init__.py` imports `numerics.lane_emden` eagerly. The 2026-09-23
architecture review measured the `jaxstro.numerics` import at 570 ms cumulative, of
which `lane_emden` was 210 ms (one cold `-X importtime` run).

## Decision

diffrax, optimistix, and sympy are required runtime dependencies of jaxstro. Only the
modules that use them may import them: `jaxstro.numerics.lane_emden` for diffrax and
optimistix, and `jaxstro.registry` for sympy, which it already imports lazily.
Importing `jaxstro.units`, `jaxstro.constants`, `jaxstro.coords`, `jaxstro.numerics`,
or `jaxstro.quad` must load none of the three, and a test enforces this.

This supersedes 0001's refusal of solver libraries. It keeps 0001's rule that a
consumer of the lightweight surfaces does not pay for the solvers, and applies it at
import time instead of at install time.

## Rationale

- **One shared implementation.** Lane-Emden and Bonnor-Ebert profiles are used by
  hydrax and progenax. Owning the solver here removes two copies; writing it without
  diffrax and optimistix would mean maintaining an adaptive ODE solver and an event
  root finder inside the foundation.
- **Registry output.** Every registry record emits LaTeX, including algorithmic ones,
  so sympy is needed wherever the registry is used, not as an add-on.
- **Import cost is what consumers feel.** A units or coordinates consumer pays the
  solver import only if it is triggered. Restricting imports to the using modules
  removes that cost without splitting the package or adding extras that hydrax and
  progenax would have to declare.
- **Accepted cost.** Installing jaxstro now resolves three more packages, and their
  releases can require a re-lock. 0001's version-decoupling concern is accepted for
  these three packages only; any further solver dependency needs a new ADR.

## Consequences

- `jaxstro.numerics` stops importing `lane_emden` at package import. Callers import
  `jaxstro.numerics.lane_emden` directly, or `numerics` exposes it lazily; the change
  is made in the same slice as the enforcement test.
- The enforcement test imports each lightweight module in a fresh interpreter and
  asserts that `diffrax`, `optimistix`, and `sympy` are absent from `sys.modules`.
