---
title: Probability and distributions
description: Support, density, expectation, covariance, conditioning, and uncertainty.
---

# Probability and distributions

Use this page when probability, uncertainty, or a distribution's support needs
to be stated before an inference or sampling calculation.

Probability supplies a language for uncertainty and variation. It does not say
why a system is uncertain until we specify the experiment, model, and state of
knowledge.

## Mass, density, and support

For discrete outcomes, **probability mass** assigns probabilities that sum to
one. For continuous quantities, a **probability density** integrates to one but
is not itself the probability of an exact point. Density carries reciprocal
units of its variable. The **support** states which outcomes the model admits;
behavior outside support is part of the contract.

**Normalization** makes the total probability one. In a finite stellar-mass
power law, the normalization changes smoothly with the exponent and has a
logarithmic removable limit. A correct formula must preserve values and
parameter derivatives through that limit.

## Summaries and dependence

An **expectation** is a probability-weighted average under a specified
distribution. Variance measures squared spread about an expectation.
**Covariance** records linear co-variation and has product units; correlation is
its dimensionless normalized form. These summaries do not uniquely determine a
general distribution.

Conditioning, written $p(x\mid y)$, changes the distribution of one quantity
given information about another. It is not the same as numerical conditioning,
though both concepts appear in scientific inference.

## Transformations and sampling

A transformation changes density through its local volume factor, not merely by
substituting values. Sampling produces draws from a specified distribution; it
does not validate that distribution as a model of nature. Inverse-CDF sampling
depends on normalization, monotonicity, support, and a reliable quantile map.

## Kinds of uncertainty

**Aleatoric** uncertainty represents variation modeled as intrinsic to the data-
generating process. **Epistemic** uncertainty represents limited knowledge about
models, parameters, or missing structure. The boundary is model-dependent: a
latent physical variable treated as noise in one analysis may become explicit
state in another.

## Predict

Name the random variable, support, units, normalization, expected symmetries,
limiting cases, and which uncertainty is represented.

## Compute

Evaluate density, CDF, quantile, moments, or samples with explicit boundary
behavior. Use stable limiting kernels and keep random seeds and transformations
in the evidence record.

## Audit

Check normalization, nonnegativity, support boundaries, monotonic CDF behavior,
CDF/quantile round trips, analytic moments where available, and parameter
derivatives against an independent method.

## State the warranted claim

Numerical normalization and round trips warrant a distribution-implementation
claim. They do not show that the chosen distribution adequately represents a
stellar population or measurement process.

## Misconception check

> A density can exceed one because it is not probability mass. A random draw is
> not evidence that a model is random "in reality," and a normalized likelihood
> over data is not automatically a normalized posterior over parameters.

Continue to [](../models-and-computation/models-inference-information.md) or
Jaxstro's [](../../20-methods/probability-sampling/distributions.md) chapter.
