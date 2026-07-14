---
title: What is a model?
description: Models as question-dependent representations and information compression.
---

# What is a model?

Use this page when a scientific model's purpose, assumptions, or discarded
information needs to be made explicit.

A model is a deliberately incomplete representation that connects assumptions
and inputs to predictions about selected aspects of a system. Its usefulness is
question-dependent: omitting microscopic detail may be exactly right for one
observable and fatal for another.

## Several meanings of model

- A **conceptual model** names entities and causal or structural relationships.
- A **mathematical model** expresses those relationships with equations.
- A **computational model** turns the equations and approximations into an
  executable procedure.
- A **statistical model** assigns probabilities to possible data or latent
  quantities conditional on assumptions and parameters.
- A **generative model** describes a process that could produce simulated
  observations.
- A **surrogate model** approximates a more expensive mapping within a declared
  domain.

One project can contain all six. Confusing them makes it easy to validate code
and accidentally claim that nature itself was validated.

## The parts of a scientific model

**Parameters** are values used to specify a model instance. **State** describes
the configuration that evolves or is solved for. **Latent variables** are not
directly observed. **Observables** are connected to measurements. A
**hyperparameter** controls a distribution or family of parameters. A
**nuisance parameter** affects the data model but is not the scientific target.
These roles depend on the question, not only on the variable's name.

In the Stefan-Boltzmann relation, radius and temperature can be inputs used to
predict luminosity. In an inference problem, luminosity and temperature might
be observed while radius becomes a parameter. The equation is unchanged; the
direction of scientific reasoning is not.

## Models are information compression

A useful model keeps information relevant to a question and discards detail it
declares irrelevant. This is **information compression**, not necessarily file
compression. A stellar spectrum contains many sampled flux values; a model may
represent much of its structured variation with a smaller set of physical
parameters. The discarded information cannot be recovered by a better optimizer.

Sufficiency is model-relative. A compressed statistic can retain all parameter
information under one likelihood and lose crucial evidence under another.
Parameter count is not scientific information, and a precise answer can still
come from a misspecified model.

## Dimensionality is not only spatial

- **Spatial dimension** counts coordinates needed to locate a point.
- **Physical dimension** describes units such as mass, length, and time.
- **Array rank and shape** describe a representation in memory.
- **Data-space dimension** counts independently represented observations.
- **Parameter-space dimension** counts coordinates needed to specify parameters.
- **State-space dimension** counts coordinates needed to specify system state.
- **Model dimension** counts nominal degrees of freedom in a chosen family.
- **Intrinsic dimension** describes the dimension of the structure supporting
  the data or solutions.
- **Effective dimension** describes how many combinations matter at a chosen
  scale or under a chosen experiment.

A system living in three-dimensional physical space can have enormous state and
parameter spaces. Conversely, many nominal parameters can collapse into a few
identifiable combinations.

## Predict

Name the model type, target observable, assumptions, parameters, state, latent
quantities, and information intentionally discarded. Predict at least one case
where the model should fail.

## Compute

Evaluate the mapping with explicit units and provenance. Keep preparation,
runtime kernels, and measurement-model operations visible rather than hiding
them behind one undifferentiated function.

## Audit

Separate implementation checks, numerical checks, predictive checks, residual
structure, and source evidence. Ask whether a discrepancy diagnoses code,
numerics, data, or model inadequacy.

## State the warranted claim

State which mapping was tested, on which domain, against which evidence. Do not
turn parameter recovery under simulated data into a universal identifiability
claim or turn a small residual into proof that the model is physically complete.

## Misconception check

> A model is not "the truth with noise added." It is a purposeful representation.
> More parameters do not guarantee more scientific information, and fewer data
> columns do not automatically mean less relevant information.

Continue to [](../mathematical-objects/linear-algebra-language-of-change.md) and
[](./models-inference-information.md).
