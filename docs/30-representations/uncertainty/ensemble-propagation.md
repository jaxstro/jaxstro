---
title: Ensemble propagation
description: >-
  Propagating explicit draws through scientific maps with honest sampling, key,
  batching, dependence, and memory contracts.
---

# Ensemble propagation

Use this page when the distribution-generation process is available as explicit
draws and nonlinear or multimodal output structure matters beyond a local moment
approximation.

:::{important} Planned Jaxstro capability
`jaxstro.uncertainty` does not exist. Keyed ensemble propagation is only a proposed
domain-agnostic representation contract. No implementation schedule is promised by
this guide.
:::

## The scientific question

What output variation results when an explicitly generated collection of inputs is
mapped through the same scientific function? Ensemble propagation evaluates
$f(\mathbf{x}_n)$ for each member and summarizes or retains the resulting empirical
distribution. It can expose skewness, tail behavior, or separated modes that a
single covariance misses.

An ensemble only represents the distribution-generation process used to create it.
Independent prior draws, posterior-chain states, bootstrap replicates, randomized
quadrature points, and perturbed numerical resolutions have different meanings even
when they have the same array shape. The generator, conditioning information,
weights, dependence, and random-key lineage must accompany the values.

## Mathematical objects

Let $\mathbf{x}_n\in\mathbb{R}^{d}$ for $n=1,\ldots,N$ be explicit input members,
and let $\mathbf{y}_n=f(\mathbf{x}_n)\in\mathbb{R}^{m}$. For the ordinary unbiased
sample covariance below, require `N >= 2`, equal weights, and members that justify
the usual independent-sample interpretation. Define
$\boldsymbol{\epsilon}_n=\mathbf{y}_n-\widehat{\boldsymbol{\mu}}_y$.

Array axes are semantic. One axis indexes ensemble members; remaining axes describe
one scientific output. Units belong to output components, and covariance entry
$(a,b)$ has units $[Y_a][Y_b]$. A batch axis used for device execution must not be
confused with an independent replicate axis.

Weighted ensembles require normalized weights and a covariance denominator matched
to their sampling design. Correlated Markov-chain states require autocorrelation or
effective-sample-size reasoning. Neither case is described by blindly replacing
$N$ in the ordinary formula.

## Core derivation

For equal-weight members, the empirical mean and Bessel-corrected sample covariance
are

```{math}
:label: eq-ensemble-propagation
\widehat{\boldsymbol{\mu}}_y=\frac{1}{N}\sum_{n=1}^{N}f(\mathbf{x}_n),
\qquad
\widehat{\mathbf{C}}_y=\frac{1}{N-1}\sum_{n=1}^{N}
\boldsymbol{\epsilon}_n\boldsymbol{\epsilon}_n^{\mathsf{T}},
\qquad N\geq 2.
```

Equation [](#eq-ensemble-propagation) uses $N-1$ because estimating the sample mean
imposes one linear constraint on the centered residuals. For independent draws with
finite second moments, this makes the sample covariance unbiased for the population
covariance. It does not make the sample standard deviation, inverse covariance, or
nonlinear functions of the covariance unbiased.

The empirical mean has Monte Carlo error that commonly scales as $N^{-1/2}$ under
independent finite-variance sampling, but the coefficient and applicability depend
on the observable and generator. Tail probabilities, high-dimensional covariance,
and rare-event observables may converge much more slowly in practice.

## Failure modes and interpretation limits

- Reusing one PRNG key can duplicate or correlate members while preserving plausible
  shapes and finite values.
- Treating correlated chain states as independent understates estimator uncertainty.
- Ignoring nonuniform importance weights changes the represented distribution.
- Small ensembles produce noisy, rank-deficient covariance estimates; rank is at
  most $N-1$ after centering.
- Rare modes or tails absent from the generator cannot appear after propagation.
- Device batching can exceed memory because both inputs and outputs may be retained;
  streaming moments reduce memory but discard sample-level diagnostics.
- Adaptive stopping based on observed ensemble values changes the sampling design
  and must be recorded.
- Numerical failures in selected members cannot be silently removed without
  changing weights and support.

## What Jaxstro may add

JAX owns transformations and random primitives. NumPyro and BlackJAX own
probabilistic inference and sampling mechanics. Informax owns inference-aware
scientific workflows in the Jaxstro ecosystem. A future
`jaxstro.uncertainty` would own only domain-agnostic propagation representations,
unit and shape conventions, deterministic key policy, provenance, and evidence
contracts. Covariance propagation does not perform inference and does not validate
a probability model.

A future surface may define member-axis metadata, deterministic key-splitting
policy, fixed-shape mapped evaluation, chunked accumulation, failure masks, and
provenance for the generator and executed map. It must not implement posterior
sampling, hide keys, infer independence from array shape, or relabel one ensemble as
a universally valid uncertainty distribution.

## Evidence required before implementation

Tests must reproduce analytic transformed moments for simple laws and maps, verify
the $N-1$ covariance convention on hand-calculated examples, and demonstrate Monte
Carlo convergence across independent replications rather than one lucky seed.
Deterministic replay must cover key roots, split order, member ordering, device
batching, and chunk sizes. Weighted and dependent inputs must either use explicitly
validated formulas or fail closed as unsupported.

Performance evidence must separate compilation, evaluation, transfer, and retained
memory. Scientific evidence must report ensemble size, effective rank, failed-member
policy, generator identity, component units, and convergence of the actual target
observable. Agreement with itself after reshaping is not an independent validation.

## Claim boundary

An ensemble supports claims about its documented generation process and finite
sample, not every distribution consistent with its members. This guide does not
provide a sampler, posterior, likelihood, convergence diagnosis for arbitrary
chains, or scientific acceptance policy. A future Jaxstro owner could propagate
and audit explicit ensembles only; inference and domain interpretation remain with
their established owners.

## Connected representations, foundations, and methods

- Begin with [](what-uncertainty-represents.md) to classify the ensemble's meaning.
- Return to [](../representations.md) for shape, unit, and provenance context.
- Review [](../../10-foundations/mathematical-objects/probability-and-distributions.md)
  for expectation and sample statistics.
- Use [](../../20-methods/probability-sampling/random.md),
  [](../../20-methods/probability-sampling/sampling.md), and
  [](linearized-propagation.md) to compare key mechanics, sampling mechanics, and
  local propagation.
