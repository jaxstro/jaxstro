---
title: jaxstro
subtitle: Evidence-first JAX infrastructure for differentiable science
description: >-
  An astro-first, science-general foundation for explicit representations,
  differentiable computation, independent audits, and bounded claims.
---

jaxstro is an **astro-first, science-general, evidence-first** foundation for
scientific software in JAX. It owns the reusable constants, units,
representations, numerical primitives, provenance, and validation tools that
belong beneath domain research packages. It does not own domain simulations,
scientific acceptance policy, or general solver stacks maintained elsewhere.

The site follows one scientific chain:

```text
representation -> computation -> audit -> evidence -> claim
```

Start with [](./00-start-here/start-here.md) if Jaxstro is new to you. Read
[](./00-start-here/why-jax.md) before committing a research program to JAX: it
separates the real benefits of transformations and accelerators from the costs
of static shapes, compilation, explicit state, and derivative boundaries.

## Enter through your research question

::::{grid} 1 1 2 2

:::{card} Which ideas need reconnecting?
:link: ./10-foundations/foundations.md

Rebuild the mathematical bridge among units, models, derivatives,
conditioning, probability, inference, and executable programs.
:::

:::{card} Which numerical method fits?
:link: ./20-methods/methods.md

Choose a method by the question it answers, then inspect its assumptions,
finite algorithm, JAX behavior, audit, and claim boundary.
:::

:::{card} How should the research computation proceed?
:link: ./40-workflows/workflows.md

Connect a scientific representation to an explicit computation plan,
execution record, independent audit, evidence, and bounded claim.
:::

:::{card} What is the current Python owner?
:link: ./50-api/api.md

Look up the importable owner, signature, transform behavior, failure contract,
and evidence without mistaking proposed capabilities for current APIs.
:::

:::{card} What evidence supports the claim?
:link: ./60-validation/validation.md

Trace a numerical or representation claim to its comparison policy, measured
quantity, executable anchor, and stated limit.
:::

::::

`jaxstro.units` is the current canonical ecosystem contract.
`jaxstro.quantity` is implemented for evaluation, but ecosystem adoption and any replacement cutover are deferred.
The broader ownership rationale and
future admission criteria live in
[](./70-project/direction/science-general-vision.md).
