---
title: Sigma-point propagation
description: >-
  Deterministic weighted-point approximations to transformed means and covariances,
  with convention and finite-precision boundaries made explicit.
---

# Sigma-point propagation

Use this page when a nonlinear map makes first-order propagation inadequate but a
small deterministic set of representative input points is more practical than a
large ensemble.

:::{important} Planned Jaxstro capability
`jaxstro.uncertainty` does not exist. Sigma-point records and propagation helpers
remain a proposed representation, not an available runtime API. No implementation
schedule is promised by this guide.
:::

## The scientific question

Can selected points reproduce chosen input moments and approximate the output mean
and covariance after a nonlinear transformation? Sigma-point methods answer this by
replacing an input law with a finite weighted rule. The rule is deterministic once
the construction, scaling parameters, matrix factorization, component order, and
input moments are fixed.

This does not make the finite rule a set of random draws. It is a quadrature-like
representation designed to match specified moments under a convention. Different
conventions can use different point counts, locations, scaling rules, and weights,
so the convention name and parameters are part of the scientific record.

## Mathematical objects

Let $\boldsymbol{\chi}_{i}\in\mathbb{R}^{d}$ for $i=0,\ldots,L-1$ be sigma
points constructed from input mean $\boldsymbol{\mu}_{x}$ and covariance
$\mathbf{C}_{x}$. Let $w_i^{(m)}$ denote mean weights and $w_i^{(c)}$ covariance
weights. The mean and covariance weights may differ.

Typically the weighted input points satisfy
$\sum_i w_i^{(m)}\boldsymbol{\chi}_i=\boldsymbol{\mu}_x$ and a weighted centered
outer-product condition approximating or reproducing $\mathbf{C}_x$. Construction
often requires a square root of a scaled covariance. Cholesky, eigendecomposition,
and other roots can produce rotated point sets with different finite-precision
behavior. The covariance entries retain units $[X_i][X_j]$, so a factorization of
mixed-unit components requires an explicit scaling convention.

The transformed points are $\mathbf{y}_i=f(\boldsymbol{\chi}_i)$. Their ordering
and weights remain static convention data; their values may be dynamic JAX leaves
in a future representation.

## Core derivation

Approximate transformed moments by applying the weighted rule to $f$ and to the
centered outer product:

```{math}
:label: eq-sigma-point-propagation
\widehat{\boldsymbol{\mu}}_{y}=\sum_i w_i^{(m)}f(\boldsymbol{\chi}_i),
\qquad
\widehat{\mathbf{C}}_y=\sum_i w_i^{(c)}
\boldsymbol{\delta}_i\boldsymbol{\delta}_i^{\mathsf{T}},
\qquad
\boldsymbol{\delta}_i=f(\boldsymbol{\chi}_i)-
\widehat{\boldsymbol{\mu}}_y.
```

Equation [](#eq-sigma-point-propagation) is a weighted quadrature approximation,
not a sampling identity. If the transformed function belongs to a polynomial class
integrated exactly by the chosen rule under the assumed input law, selected moments
may be exact in exact arithmetic. Outside that class, omitted higher moments and
the point geometry control the error.

Sigma-point construction and weight conventions vary. Scaled unscented rules, for
example, may use different mean and covariance weights for the central point. Some
conventions allow negative weights to reproduce target moments. Negative covariance
weights mean positive-semidefinite output is not automatic under every
finite-precision choice, even though each outer product is symmetric.

## Failure modes and interpretation limits

- A point rule that matches mean and covariance does not encode arbitrary tails,
  support boundaries, skewness, or multimodality.
- Negative weights can amplify cancellation and roundoff, producing a covariance
  with small negative eigenvalues or worse failures under poor scaling.
- A failed or regularized covariance factorization changes the represented input;
  silent jitter is not a harmless implementation detail.
- Strong nonlinearity between sigma points can remain invisible to the rule.
- Point locations can leave bounded physical support even when the intended input
  law does not, requiring a documented transformed-space construction rather than
  clipping.
- Branches and discontinuities can make results depend abruptly on scaling
  parameters or point orientation.
- Agreement among two closely related sigma-point conventions is not independent
  validation of the input probability model.

## What Jaxstro may add

JAX owns transformations and random primitives. NumPyro and BlackJAX own
probabilistic inference and sampling mechanics. Informax owns inference-aware
scientific workflows in the Jaxstro ecosystem. A future
`jaxstro.uncertainty` would own only domain-agnostic propagation representations,
unit and shape conventions, deterministic key policy, provenance, and evidence
contracts. Covariance propagation does not perform inference and does not validate
a probability model.

A future surface may store point and weight conventions, verify weighted input
moments, apply a map with fixed-shape batching, and report factorization, symmetry,
rank, and positive-semidefinite diagnostics. It must never infer scaling parameters,
clip points without declaring a new representation, or present one convention as a
universal sigma-point rule.

## Evidence required before implementation

Analytic evidence must include constant, affine, and low-order polynomial maps with
known transformed moments. Tests must cover distinct mean and covariance weights,
negative-weight conventions, diagonal and correlated covariances, singular and
nearly singular inputs, mixed component scales, deterministic point ordering, and
float32 versus float64 behavior. The generated point set must reconstruct the
moments promised by its named convention within explicit tolerances.

Nonlinear comparisons should use independent high-sample ensembles and controlled
input laws, reporting error in mean, covariance, and scientifically relevant tail
events separately. JIT, VMAP, shape, dtype, unit metadata, and provenance behavior
must be tested. No comparison can substitute for validating the input probability
model or the downstream scientific assumptions.

## Claim boundary

Sigma points approximate selected transformed moments under a specified finite
rule. They are neither posterior samples nor proof that the input law is correct.
This page does not choose a sigma-point convention, guarantee positive-semidefinite
output for every weighted finite-precision computation, or promise a runtime
module. The proposed owner is limited to explicit propagation representations and
evidence contracts.

## Connected representations, foundations, and methods

- Read [](what-uncertainty-represents.md) before treating moments as a complete
  uncertainty description.
- Return to [](../representations.md) for current representation owners.
- Review [](../../10-foundations/mathematical-objects/linear-algebra-language-of-change.md)
  for covariance factorizations and quadratic forms.
- Compare [](linearized-propagation.md), [](ensemble-propagation.md), and
  [](../../20-methods/approximation-integration/quadrature.md) to separate local,
  weighted-rule, sampled, and integration viewpoints.
