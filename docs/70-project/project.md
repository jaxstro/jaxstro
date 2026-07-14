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

- [](./direction/architecture.md) explains current package ownership and the
  one-way ecosystem dependency rule.
- [](./direction/science-general-vision.md) states the product direction and
  admission criteria for future work.
- [](./development/development.md) links current evidence, assessed gaps, and
  definition-of-done roadmaps.
- [](./decisions/decisions.md) indexes the architecture decisions. Individual
  records remain available at their established public routes.
- [](./release/release.md) and [](./release/checklist.md) separate local
  qualification from authorized publication actions.
- [](./bibliography/bibliography.md) records the load-bearing scientific
  sources behind constants and methods.

Project claims follow the same evidence boundary as runtime claims: an
implemented feature is not automatically validated, and a local verification
result is not authorization for a remote release action.
