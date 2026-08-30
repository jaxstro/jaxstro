---
title: What is a model?
description: Models as question-dependent representations and information compression.
---

# What is a model?

Use this page when a scientific model's purpose, assumptions, or discarded
information needs to be made explicit.

A source, an instrument, and a recorded datum do not belong to one undifferentiated
model. Each representation keeps some structure and leaves other structure out.
The question is whether the retained structure can answer the observable at
hand.

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

One minimal scientific chain makes the distinct model roles visible:

```{math}
z = f(\theta, s) + \delta,
\qquad
d = h(z, \eta) + \varepsilon.
```

Here $f$ is a physical model from parameters $\theta$ and state $s$ to an ideal
observable $z$; $\delta$ represents declared model discrepancy; $h$ is a
measurement model with calibration or selection parameters $\eta$; and
$\varepsilon$ represents a stated error model. Replacing $f$ with a fast
approximation changes the computational or surrogate model; it does not settle
whether $\delta$ is negligible for the observable.

## Models are information compression

A useful model keeps information relevant to a question and discards detail it
declares irrelevant. This is **information compression**, not necessarily file
compression. A stellar spectrum contains many sampled flux values; a model may
represent much of its structured variation with a smaller set of physical
parameters. The discarded information cannot be recovered by a better optimizer.

Sufficiency is model-relative. A compressed statistic can retain all parameter
information under one likelihood and lose relevant evidence under another.
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

## Try the running case

For the two-channel measurement, list one parameter of the physical source,
one latent state, one calibration nuisance parameter, and one
recorded datum. Then ask which of those roles would change if the same source
were used to validate a simulation rather than infer its parameters.

## Worked audit

The datum is not a parameter merely because it appears in an array, and the
calibration is not irrelevant merely because it is a nuisance. In a prediction
task the source parameters may be fixed inputs; in an inference task they become
unknowns. This role change is why a correct forward calculation is not yet a
complete statistical model.

:::{figure} ../figures/model-measurement-chain.svg
:name: fig-model-measurement-chain
:alt: Source parameters and state enter a physical model, then a measurement model with calibration, before recorded data. Model discrepancy and measurement error enter at different stages.

Model discrepancy, calibration, and measurement error can all alter recorded
data. They enter different parts of the chain and cannot be exchanged without
changing the scientific interpretation.
:::

:::{admonition} Keep the model chain visible

The physical map, measurement map, and error terms make different assumptions.
Treating them as one black box hides the source of a mismatch.
:::

::::{grid} 1 1 3 3

:::{card} Predict
Name the model type, target observable, assumptions, parameters, state, latent
quantities, and information intentionally discarded. Name one case where the
model should fail.
:::

:::{card} Compute
Evaluate the mapping with explicit units and provenance. Keep preparation,
runtime kernels, and measurement-model operations visible rather than hiding
them behind one undifferentiated function.
:::

:::{card} Audit
Separate implementation, numerical, predictive, residual-structure, and source
checks. Ask whether a discrepancy diagnoses code, numerics, data, or model
inadequacy.
:::

::::

:::{important} Claim boundary
State which mapping was tested, on which domain, and against which evidence.
Parameter recovery under simulated data does not establish universal
identifiability, and a small residual does not show that the model is physically
complete.
:::

:::{warning} A common mistake
A model is a purposeful representation, not "the truth with noise added." More
parameters do not guarantee more scientific information, and fewer data columns
do not automatically mean less relevant information.
:::

Continue to [](../mathematical-objects/linear-algebra-language-of-change.md) and
[](./models-inference-information.md).
