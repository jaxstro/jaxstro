---
title: What uncertainty represents
description: >-
  A scientific vocabulary for probability models, uncertain state, noise,
  numerical error, and model discrepancy before propagation is attempted.
---

# What uncertainty represents

Use this page when a result is called "uncertain" and you need to identify which
source of incomplete knowledge is actually being represented before choosing a
propagation method.

:::{important} Planned Jaxstro capability
`jaxstro.uncertainty` does not exist. This page proposes a domain-agnostic
representation boundary only; it is not runtime documentation. No implementation
schedule is promised by this guide.
:::

## The scientific question

What does a reported distribution, covariance, interval, or ensemble say about a
scientific quantity, and what does it leave unsaid? The answer must identify the
random object, its physical support and units, the conditioning information, and
the process that generated the representation. Without those details, propagating
numbers through a function can produce a precise-looking output with no stable
scientific interpretation.

At least five categories must remain distinct:

1. **Variation represented by a probability model** describes variation across a
   defined population or repeated process.
2. **Uncertainty about parameters or latent state** describes incomplete knowledge
   about a particular system, conditional on a model and observations.
3. **Measurement and noise models** describe how an instrument or data-generating
   process connects latent quantities to recorded values.
4. **Numerical approximation error** arises from discretization, iteration,
   truncation, finite sampling, or finite precision.
5. **Model discrepancy** represents systematic inadequacy between the scientific
   model and the process being studied.

These categories are not interchangeable. Every representation must state units
and support explicitly. More Monte Carlo draws reduce sampling error but do not
repair model discrepancy. A wider measurement-noise distribution does not
automatically represent uncertainty in a calibration parameter.

## Mathematical objects

Let the random vector $\mathbf{X}$ take values in a stated support
$\mathcal{X}$ with probability law $P_{\mathbf{X}}$. Its components have units;
therefore the mean $\boldsymbol{\mu}_{X}$ has the units of $\mathbf{X}$, while
the covariance entry $(\mathbf{C}_{X})_{ij}$ has units $[X_i][X_j]$. A covariance
between length and velocity, for example, has units of length times velocity.

Support is part of the contract. A positive mass, an angle on a circle, a bounded
fraction, and an unconstrained real parameter are different mathematical objects
even if each is stored as a floating-point scalar. Conditioning is also part of the
object: $P(\mathbf{X}\mid D,M)$ depends on data $D$ and model $M$ and is not the
same claim as a population distribution $P(\mathbf{X})$.

A covariance matrix is not a complete probability distribution. It does not by
itself specify support, tail behavior, skewness, multimodality, or the distinction
between probability mass and probability density. Two laws can share the same mean
and covariance while assigning very different probability to a scientific event.

## Core derivation

For a random vector with finite second moments, mean and covariance are summaries
of a probability law:

```{math}
:label: eq-uncertainty-moments
\boldsymbol{\mu}_{X}=\mathbb{E}[\mathbf{X}],
\qquad
\mathbf{C}_{X}=\mathbb{E}\!\left[
(\mathbf{X}-\boldsymbol{\mu}_{X})
(\mathbf{X}-\boldsymbol{\mu}_{X})^{\mathsf{T}}
\right].
```

Equation [](#eq-uncertainty-moments) follows by defining the centered variable
$\boldsymbol{\eta}=\mathbf{X}-\boldsymbol{\mu}_{X}$ and taking the expected outer
product $\mathbb{E}[\boldsymbol{\eta}\boldsymbol{\eta}^{\mathsf{T}}]$. The
diagonal records marginal variances; the off-diagonal entries record linear
cross-covariances. The derivation requires finite second moments and says nothing
about Gaussianity. It is a summary operation, not an inference algorithm.

When $\mathbf{Y}=f(\mathbf{X})$, the output law is the pushforward
$P_{\mathbf{Y}}=f_{\#}P_{\mathbf{X}}$. Linearization, sigma points, and ensembles
are different approximations or numerical representations of that pushforward.
They cannot determine whether the input law was scientifically warranted.

## Failure modes and interpretation limits

- A covariance-only description can conceal skewness, heavy tails, disconnected
  support, or multiple modes.
- Treating numerical approximation error as independent random noise can hide a
  deterministic bias or convergence failure.
- Combining measurement error and model discrepancy without a generative model can
  make their contributions non-identifiable.
- Mixing components with unrecorded units can make covariance addition meaningless.
- Ignoring support can assign probability to impossible states.
- Calling posterior uncertainty "population variation" changes the conditioning
  claim and the scientific question.
- A positive-semidefinite covariance is necessary for a second-moment summary, but
  it does not validate the underlying probability model.

These are representation failures, not problems that automatic differentiation or
larger sample counts can resolve.

## What Jaxstro may add

JAX owns transformations and random primitives. NumPyro and BlackJAX own
probabilistic inference and sampling mechanics. Informax owns inference-aware
scientific workflows in the Jaxstro ecosystem. A future
`jaxstro.uncertainty` would own only domain-agnostic propagation representations,
unit and shape conventions, deterministic key policy, provenance, and evidence
contracts. Covariance propagation does not perform inference and does not validate
a probability model.

Such a surface may define explicit records for moment summaries, keyed ensembles,
and propagation diagnostics. It must require the caller to state component order,
shape, units, support, and provenance. It must not construct posteriors, choose
likelihoods, or decide which uncertainty category a domain workflow should use.

## Evidence required before implementation

Implementation would require analytic pushforward cases, unit and shape checks,
positive-semidefinite diagnostics, singular and rank-deficient cases, deterministic
key replay, and comparisons among linearized, sigma-point, and ensemble results on
controlled nonlinear maps. Tests must show that category and provenance metadata
survive JAX transformations and serialization. Independent downstream cases must
demonstrate that the proposed records carry enough information without importing
inference policy into Jaxstro.

Evidence must also contain adversarial examples where equal covariances correspond
to different supports or tail probabilities. Passing a self-consistency check is
not evidence that an input probability model describes nature.

## Claim boundary

This page supplies vocabulary and moment definitions. It does not establish a
runtime module, certify a probability model, infer a posterior, combine error
sources automatically, or turn numerical error into probability. The proposed
owner is future `jaxstro.uncertainty`, limited to propagation contracts. Scientific
meaning remains with the model, data, and downstream workflow that created the
input representation.

## Connected representations, foundations, and methods

- Return to [](../representations.md) to compare uncertainty with units,
  coordinates, spectra, and parameter-state representations.
- Review [](../../10-foundations/mathematical-objects/probability-and-distributions.md)
  for probability mass, density, expectation, covariance, and support.
- Use [](../../10-foundations/models-and-computation/models-inference-information.md)
  to distinguish a model, likelihood, posterior, and information claim.
- Compare [](../../20-methods/probability-sampling/random.md) and
  [](../../20-methods/probability-sampling/sampling.md) for explicit key and
  sampling mechanics.
