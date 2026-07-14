---
title: Linearized propagation
description: >-
  First-order covariance pushforward through a differentiable map, with explicit
  unit, rank, conditioning, and local-linearity boundaries.
---

# Linearized propagation

Use this page when uncertainties are small enough that a local derivative may
summarize how a smooth scientific map changes their covariance.

:::{important} Planned Jaxstro capability
`jaxstro.uncertainty` does not exist. The covariance pushforward described here is
a proposed, local approximation rather than an importable API. No implementation
schedule is promised by this guide.
:::

## The scientific question

Given an input vector with a stated mean and covariance, how does a smooth map
change those moments near a chosen expansion point? Linearized propagation answers
that local question efficiently. It is useful for sensitivity diagnosis and for
small, approximately unimodal perturbations, but it is not an exact general
propagation rule.

The expansion point matters. A Jacobian evaluated at the mean answers a different
local question from a Jacobian evaluated at a nominal best fit or another reference
state. The caller must record that point and the units and ordering of every input
and output component.

## Mathematical objects

Let $f:\mathbb{R}^{d}\rightarrow\mathbb{R}^{m}$ be differentiable near
$\mathbf{x}_{0}$. Let $\mathbf{X}$ have mean approximately $\mathbf{x}_{0}$ and
covariance $\mathbf{C}_{x}\in\mathbb{R}^{d\times d}$. Define
$\mathbf{Y}=f(\mathbf{X})$ and the Jacobian
$\mathbf{J}\in\mathbb{R}^{m\times d}$ at $\mathbf{x}_{0}$ by
$J_{ai}=\partial f_a/\partial x_i$.

The units expose the contract. $J_{ai}$ has units $[Y_a]/[X_i]$,
$(C_x)_{ij}$ has units $[X_i][X_j]$, and $(C_y)_{ab}$ must have units
$[Y_a][Y_b]$. Off-diagonal entries are cross-covariances, not optional noise
terms. Dropping them changes both the input model and the propagated answer.

Rank is also informative. If $\mathbf{C}_{x}$ has rank $r$, the first-order output
covariance has rank at most $\min(r,\operatorname{rank}\mathbf{J})$. A singular
covariance can represent exact constraints or an under-resolved model; adding a
small diagonal floor changes that scientific statement.

## Core derivation

Write $\boldsymbol{\xi}=\mathbf{X}-\mathbf{x}_{0}$. A first-order Taylor
expansion gives
$f(\mathbf{X})\approx f(\mathbf{x}_{0})+\mathbf{J}\boldsymbol{\xi}$.
After centering and taking the expected outer product,

```{math}
:label: eq-linearized-covariance
\mathbf{C}_{y}\approx
\mathbf{J}\mathbf{C}_{x}\mathbf{J}^{\mathsf{T}}.
```

Equation [](#eq-linearized-covariance) follows because constants vanish under
centering and
$\mathbb{E}[\boldsymbol{\xi}\boldsymbol{\xi}^{\mathsf{T}}]=\mathbf{C}_{x}$.
For an affine map $f(\mathbf{x})=\mathbf{A}\mathbf{x}+\mathbf{b}$, the equation is
exact with $\mathbf{J}=\mathbf{A}$ whenever the second moments exist. For a
nonlinear map, neglected Hessian and higher-order terms can shift both the mean and
the covariance.

If inputs $\mathbf{X}$ and $\mathbf{Z}$ are jointly uncertain, their full block
covariance includes $\mathbf{C}_{xz}$. Propagating $f(\mathbf{X},\mathbf{Z})$
requires the augmented Jacobian and those cross-covariances. Propagating the blocks
separately silently assumes independence.

## Failure modes and interpretation limits

- Strong curvature makes a single tangent map unrepresentative across the input
  support and can bias both mean and covariance.
- Discontinuities, clipping, branch changes, thresholds, and discrete indices do
  not have one scientifically meaningful local derivative at their boundary.
- Multimodal inputs can share a covariance while mapping to separated output modes
  that no covariance ellipse describes.
- Ill-conditioned $\mathbf{J}$ can amplify small input or floating-point changes;
  a finite output is not evidence of a stable calculation.
- Rank deficiency can be physical, structural, or numerical. Regularization must
  state which interpretation it changes.
- A Jacobian at an arbitrary nominal point is not automatically a Jacobian at the
  expectation, posterior mode, or true state.
- A local AD result differentiates the executed program, including any smoothing,
  clipping, or branch semantics present in that program.

## What Jaxstro may add

JAX owns transformations and random primitives. NumPyro and BlackJAX own
probabilistic inference and sampling mechanics. Informax owns inference-aware
scientific workflows in the Jaxstro ecosystem. A future
`jaxstro.uncertainty` would own only domain-agnostic propagation representations,
unit and shape conventions, deterministic key policy, provenance, and evidence
contracts. Covariance propagation does not perform inference and does not validate
a probability model.

A future helper may compute Jacobian-vector or vector-Jacobian factorizations,
preserve component and unit metadata, and report symmetry, rank, and conditioning
diagnostics. It may not choose an inference model, insert undocumented covariance
floors, discard cross-covariances, or call the approximation exact.

## Evidence required before implementation

The minimum evidence includes analytic affine maps, independently calculated
Jacobians, unit-consistency tests, symmetry and positive-semidefinite checks,
rank-deficient examples, and central finite-difference comparisons on a smooth
domain. Nonlinear cases must compare the first-order result with high-resolution
ensemble propagation as input scale shrinks, demonstrating the expected local
convergence rather than agreement at one hand-picked scale.

Tests must cover correlated inputs, rectangular Jacobians, batched covariances,
JIT and VMAP behavior, float32 and float64 conditioning, and explicit failures for
nonfinite values or mismatched shapes. Evidence should report the expansion point
and must separate numerical agreement from validation of the probability model.

## Claim boundary

The formula on this page is exact for affine maps and a first-order approximation
for general smooth maps. It does not reconstruct tails, modes, support, or a
posterior; it does not justify independence assumptions; and it does not certify
that the input covariance is scientifically meaningful. No runtime surface exists,
and no implementation or release date is implied.

## Connected representations, foundations, and methods

- Start with [](what-uncertainty-represents.md) to identify the uncertainty being
  propagated.
- Return to [](../representations.md) for the larger representation boundary.
- Review [](../../10-foundations/mathematical-objects/what-is-a-derivative.md) and
  [](../../10-foundations/models-and-computation/sensitivity-conditioning-identifiability.md)
  for Jacobians and conditioning.
- Compare [](../../20-methods/change-constraints-evolution/autodiff.md) for executed
  derivative semantics and [](ensemble-propagation.md) for a nonlocal alternative.
