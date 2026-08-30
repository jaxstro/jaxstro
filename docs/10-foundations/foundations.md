---
title: "Foundations: the ideas we will not assume"
description: Connected, optional background for evidence-first computational research.
---

# Foundations: the ideas we will not assume

Use this page when a method, program, or audit depends on an idea that needs a
clearer mathematical or scientific connection.

Optional does not mean unimportant. These pages provide connected concepts,
not prerequisites to pass. They keep the scientific standard high while making
the reasoning behind units, models, derivatives, probability, inference, and
JAX programs available when it becomes useful.

Researchers often encounter calculus, statistics, linear algebra, programming,
physical modeling, and inference separately. Prior exposure does not guarantee
that those concepts are active and connected for a new research problem.
Reconnecting them is substantive scientific work, not remediation.

You may proceed linearly through the recommended route below or enter from any
method page.
Return when an audit exposes a conceptual gap, then rejoin the method or
workflow that raised the question. Foundations connects mathematical objects,
executable programs, and warranted scientific claims.

::::{grid} 1 1 2 2

:::{card} What mathematical object is changing?
:link: ./mathematical-objects/functions-units-scales.md

Connect functions, units, scales, vectors, derivatives, probability, and
uncertainty before choosing a numerical representation.
:::

:::{card} What does the executable model mean?
:link: ./models-and-computation/what-is-a-model.md

Connect physical prediction, measurement, inference, sensitivity,
conditioning, identifiability, and the finite JAX program.
:::

::::

:::{note}
Foundations supplies concepts to the same prediction-to-claim chain used by
Methods, Workflows, API, and Validation; it is not a separate prerequisite
track.
:::

| Family | Status | Primary role |
| --- | --- | --- |
| Mathematical objects | Current guidance | Name domains, scales, structures, and uncertainty |
| Models and computation | Current guidance | Separate scientific relations from executable programs |

## Recommended route through the foundations

For a first pass, use this sequence. It follows one question from a
unit-carrying relation to a bounded computational claim; it is a recommended
route, not a prerequisite chain.

| Step | Question answered | Page |
| --- | --- | --- |
| 1 | What map, units, and scale are being represented? | [](./mathematical-objects/functions-units-scales.md) |
| 2 | Which directions of change are visible or hidden? | [](./mathematical-objects/linear-algebra-language-of-change.md) |
| 3 | Which local change is meant by a derivative? | [](./mathematical-objects/what-is-a-derivative.md) |
| 4 | What variation and support does the uncertainty model admit? | [](./mathematical-objects/probability-and-distributions.md) |
| 5 | Which kind of model is answering the question? | [](./models-and-computation/what-is-a-model.md) |
| 6 | How do parameters become recorded data and a conditional conclusion? | [](./models-and-computation/models-inference-information.md) |
| 7 | Is a sensitive or weakly constrained direction numerical, physical, or both? | [](./models-and-computation/sensitivity-conditioning-identifiability.md) |
| 8 | Which map and derivative does the finite JAX program actually expose? | [](./models-and-computation/from-relations-to-differentiable-programs.md) |

## Mathematical objects

Begin with the objects that scientific programs represent and transform:

1. [](./mathematical-objects/functions-units-scales.md) treats a function as a
   unit-carrying map with a domain, scale, and limiting behavior.
2. [](./mathematical-objects/linear-algebra-language-of-change.md) connects
   vectors and matrices to perturbations, maps, geometry, and conditioning.
3. [](./mathematical-objects/what-is-a-derivative.md) develops derivatives as
   local rates, linear maps, sensitivities, and evidence-bearing claims.
4. [](./mathematical-objects/probability-and-distributions.md) introduces
   support, density, normalization, expectation, covariance, and uncertainty.

## Models and computation

Then connect those objects to scientific reasoning and executable methods:

1. [](./models-and-computation/what-is-a-model.md) separates conceptual,
   mathematical, computational, statistical, generative, and surrogate models.
2. [](./models-and-computation/models-inference-information.md) follows the
   chain from a physical prediction through a measurement model to inference.
3. [](./models-and-computation/sensitivity-conditioning-identifiability.md)
   distinguishes response, numerical stability, and learnable parameter
   combinations.
4. [](./models-and-computation/from-relations-to-differentiable-programs.md)
   separates an ideal relation from the finite JAX program that executes it.

## Why the documentation repeats one scientific cycle

Every substantial route uses the same sequence:

```text
predict -> compute -> audit -> state the warranted claim
```

Prediction records units, signs, scales, limiting cases, invariants,
conditioning, expected failures, and the intended derivative before output can
encourage post-hoc storytelling. Computation records the program that actually
ran, including branches, finite precision, tolerances, traces, and provenance.
Audit asks an independent question through an analytic identity, limit,
finite-difference check, convergence study, conservation law, or source check.

The warranted claim is the scientific product. "The code ran" is weaker than
"the value satisfies the stated numerical contract," which is weaker than a
claim that a physical model adequately describes an observation.

## Connected routes, not duplicate explanations

Foundations asks what an idea means and how it connects to scientific
reasoning. The [](../20-methods/methods.md) chapters explain how numerical methods
work and where their algorithmic boundaries lie. The [](../50-api/api.md)
records exact public surfaces. The [](../60-validation/validation.md) links claims
to executable evidence.

These routes are complementary. Use [](../00-start-here/choose-your-path.md) to
locate a concept by the problem you are facing, or
[](../00-start-here/ways-to-use-these-docs.md) to choose a route through the
site. After an audit, begin the cycle again with a better prediction, method,
or claim.
