---
title: Scientific representations
description: >-
  Current Jaxstro representations for units, coordinates, spectra, atmosphere
  artifacts, parameters, and auditable scientific state.
---

# Scientific representations

Scientific representations sit between mathematical methods and research workflows.
They specify what an array means before a numerical method acts on it and before a
workflow interprets the result. A representation fixes conventions such as units,
coordinates, sampling semantics, static metadata, and the boundary of a scientific
claim.

Choosing an array dtype and shape is not the same as choosing a scientific representation.
Two arrays can have identical values and shapes while representing different physical
dimensions, coordinate frames, spectral densities, or parameter constraints. Jaxstro
makes those distinctions explicit where the runtime supports them.

::::{grid} 1 1 2 2

:::{card} Units and quantities
:link: ./units-quantities/constants-and-conventions.md

Choose physical constants, named unit systems, exact dimensional quantities, and
explicit equivalencies without hiding conversions inside numerical kernels.
:::

:::{card} Geometry and coordinates
:link: ./geometry-coordinates/coordinate-transformations.md

Represent vectors, rigid transforms, sky frames, parallaxes, and proper motions with
their orientation, angular, unit, and singular-domain conventions visible.
:::

:::{card} Spectra and atmospheres
:link: ./spectra-atmospheres/spectra-data-architecture.md

Keep spectral coordinate, sampling, density semantic, source provenance, local
artifact identity, and evidence-gated interpolation policy attached to the data.
:::

:::{card} Parameters and scientific state
:link: ./parameters-state/parameters-and-transforms.md

Separate structured physical state from unconstrained optimization coordinates, and
record what an executed method consumed without confusing runtime provenance with
scientific validation.
:::

:::{card} Uncertainty propagation
:link: ./uncertainty/what-uncertainty-represents.md

**Planned Jaxstro capability.** Distinguish probability, uncertain state, noise,
numerical error, and model discrepancy before comparing linearized, sigma-point,
and ensemble propagation. No runtime module or schedule is promised.
:::

:::{card} Fields and discretized domains
:link: ./fields/fields-and-domains.md

**Deferred abstraction.** Separate field values, domains, coordinates, topology,
operators, units, and boundaries. A common runtime awaits evidence from two real
consumers; these guides promise no module or schedule.
:::
::::

## How to read this section

Each implemented capability page begins with a seven-row representation contract.
Read that table before the examples: it names the mathematical object, convention,
runtime owner, shape and unit policy, JAX transform boundary, evidence, and
downstream interpretation boundary. Planned and deferred guides instead begin with
an exact status and keep their derivation, failure modes, evidence gate, and claim
boundary visible.

Status labels separate implemented Jaxstro capabilities from planned propagation
work and deferred field abstractions. Numerical algorithms live under
[](../20-methods/methods.md), research workflows remain at their current published
locations until their dedicated migration, and validation anchors live in
[](../60-validation/validation.md).
