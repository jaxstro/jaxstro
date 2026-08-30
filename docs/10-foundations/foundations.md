---
title: "Foundations: the ideas we will not assume"
description: Connected, optional background for evidence-first computational research.
---

# Foundations: the ideas we will not assume

An inference can fail before an algorithm does: a residual can mix units, a
derivative can belong to the wrong map, or a likelihood can omit the calibration
that correlates two measurements. Foundations keeps those connections visible.

These are optional connected routes, not prerequisites to complete in order.

Researchers often encounter calculus, statistics, linear algebra, programming,
physical modeling, and inference separately. The trouble appears when one
measurement must carry all of them at once: a source produces an observable, an
instrument calibrates and records it, an inference assigns responsibility for a
discrepancy, and a program reports a derivative. Reconnecting those steps is
part of the scientific work.

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

## Running case: a two-channel measurement

A source is measured in two channels. The channels share a calibration
uncertainty, and two source parameters can produce nearly the same change in
the recorded data. This small problem is enough to expose the decisions that
usually become hidden inside a scientific program: units, covariance, rank,
priors, model discrepancy, and derivative meaning.

:::{figure} ./figures/two-channel-measurement-overview.png
:name: fig-two-channel-measurement-overview
:alt: A luminous source is observed through two instrument channels, combined into a calibrated two-component data vector, and mapped into an elongated parameter-space uncertainty region.

Two channels see one source through different measurement paths. A shared
calibration can correlate the recorded values, while a long uncertainty region
marks a parameter combination the data weakly distinguish.
:::

Run the companion calculation from the repository root:

```bash
env -u VIRTUAL_ENV uv run --no-sync python \
  examples/onboarding/two_channel_measurement.py --calibration-sigma 0.2 --separation 0.01
```

Change `--calibration-sigma` to alter the shared covariance and `--separation`
to bring the two Jacobian columns together. The calculation prints the
covariance, singular values, and the narrower claim each configuration supports.

| Family | Status | Primary role |
| --- | --- | --- |
| Mathematical objects | Current guidance | Name domains, scales, structures, and uncertainty |
| Models and computation | Current guidance | Separate scientific relations from executable programs |

## Recommended route through the foundations

For a first pass, follow the measurement from its physical relation to the
claim made about it. This is a route through one argument, not a prerequisite
chain.

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
