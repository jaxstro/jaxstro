---
title: Adaptive quadrature in the JAX ecosystem
description: Local error indicators, interval refinement, and delegated adaptive quadrature ownership.
---

# Adaptive quadrature in the JAX ecosystem

Use this page when an integral cannot be resolved efficiently by a fixed rule
and you need to understand how local estimates drive interval refinement.

:::{important} Ecosystem guide
[Quadax](https://quadax.readthedocs.io/en/) owns adaptive quadrature algorithms
for JAX. Jaxstro does not duplicate that runtime method family.
:::

## The scientific question

Adaptive quadrature asks where additional integrand evaluations are most useful.
It estimates local error on current intervals, refines selected intervals, and
combines local contributions into a global integral and error account.

The difficult part is often structural rather than merely numerical: endpoint
singularities, discontinuities, sharp interior features, oscillations, or
infinite domains can invalidate a naive error estimate or transformation.

## Mathematical objects

Let $I=\int_a^b f(x)\,dx$. Partition the finite domain into intervals
$[a_i,b_i]$. A paired or nested rule supplies two approximations, commonly of
different order or resolution, on each interval. Their difference is a local
error indicator, not an exact error.

For an infinite domain, an explicit change of variables maps a finite parameter
interval to the original domain and introduces a Jacobian. That transformation
changes the endpoint behavior seen by the quadrature rule.

## Core derivation

Write the local high-order estimate as $Q_i^{(h)}$ and its paired estimate as
$Q_i^{(l)}$. A basic adaptive account is:

```{math}
:label: eq-adaptive-quadrature-account

\begin{aligned}
\widehat{I} &= \sum_{i=1}^{M}Q_i^{(h)}, \\
e_i &= \left|Q_i^{(h)}-Q_i^{(l)}\right|, \\
E &= \sum_{i=1}^{M}e_i.
\end{aligned}
```

The algorithm can refine the interval with the largest $e_i$, update only the
affected terms in [](#eq-adaptive-quadrature-account), and stop when the chosen
global tolerance rule passes. The sum $E$ is conservative only under the
assumptions of the estimator and accounting policy; cancellation in the true
error is not a license to report a smaller unsupported bound.

## What the ecosystem already owns

[Quadax](https://quadax.readthedocs.io/en/) owns adaptive interval selection,
rule evaluation, refinement, stopping, and runtime result states for this
ecosystem. Jaxstro's current quadrature page covers fixed-node and cumulative
rules with different transform contracts; it is not an adaptive engine.

## What Jaxstro may add

A consumer-driven Jaxstro adapter may later define unit-aware integral input and
output contracts, attach domain transformations and singular-point declarations
to provenance, and publish evidence envelopes for named integral families. No
such adapter exists.

The adapter would delegate all adaptive decisions to Quadax and would not
reimplement interval trees, paired rules, or stopping logic.

## Evidence required before implementation

An adapter would require:

- analytic finite-domain integrals across smooth and endpoint-limited cases;
- independent references for discontinuous, peaked, and oscillatory examples;
- transformed infinite-domain cases with Jacobian and tail checks;
- tolerance sweeps that compare reported estimates with observed error;
- unit conversion tests for integrand, coordinate, and integral dimensions;
- nonfinite, exhausted-budget, and unresolved-singularity failure cases; and
- provenance checks for domain maps, tolerances, status, and evaluation count.

## Claim boundary

:::{warning}
A small paired-rule difference is not a universal error certificate. Both rules
can miss the same unresolved feature. Singularities and infinite-domain tails
must be modeled and audited explicitly.
:::

This guide makes no performance or convergence-rate claim for Quadax, and it
does not establish an adaptive Jaxstro API.

## Connected foundations and methods

Review [](../../10-foundations/mathematical-objects/functions-units-scales.md)
for dimensional maps. Compare fixed rules in [](./quadrature.md), sampled
accumulation in [](./cumulative-trapz.md), and finite-data approximation in
[](./interpolation.md). The conditioning perspective in
[](../../10-foundations/models-and-computation/sensitivity-conditioning-identifiability.md)
helps explain why narrow features can defeat an otherwise plausible estimate.
