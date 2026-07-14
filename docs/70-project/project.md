---
title: Project
description: Direction, architecture, decisions, development evidence, and release policy for Jaxstro.
---

# Project

Jaxstro is the dependency-light foundation for reusable, JAX-native scientific
software. It owns domain-general constants, units, coordinates, numerical
primitives, scientific representations, provenance, and validation tooling. It
does not own domain simulations, scientific acceptance policy, or general
solver stacks already maintained elsewhere.

Use this section to inspect why that boundary exists and how it changes:

::::{grid} 1 1 2 2

:::{card} What does Jaxstro own?
:link: ./direction/architecture.md

Inspect the current package map and one-way ecosystem dependency rule.
:::

:::{card} Which future work belongs here?
:link: ./direction/science-general-vision.md

Apply the science-general admission criteria before adding a new foundation
capability.
:::

:::{card} What evidence and gaps guide development?
:link: ./development/development.md

Follow assessed gaps, definitions of done, and capability roadmaps.
:::

:::{card} Why was this architecture chosen?
:link: ./decisions/decisions.md

Read the decision index; individual records retain their established routes.
:::

:::{card} What qualifies a release?
:link: ./release/release.md

Separate local qualification from explicitly authorized publication actions.
:::

:::{card} Which sources are load-bearing?
:link: ./bibliography/bibliography.md

Trace constants and methods to the scientific sources that constrain them.
:::

::::

:::{note}
Project records preserve the ownership and evidence decisions behind every
other route; they do not substitute for runtime or scientific validation.
:::

| Material | Status | Authority |
| --- | --- | --- |
| Architecture and decisions | Current | Package ownership and accepted design rationale |
| Development roadmaps | Planned | Admission and evidence requirements, not schedules |
| Release records | Current policy | Local qualification remains distinct from publication authorization |
| Bibliography | Current source record | Scientific provenance, not independent validation |

Project claims follow the same evidence boundary as runtime claims: an
implemented feature is not automatically validated, and a local verification
result is not authorization for a remote release action.
