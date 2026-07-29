---
title: Randomized quasi-Monte Carlo
description: Sobol construction, scrambling, replicate uncertainty, and bounded sequential stopping.
---

# Randomized quasi-Monte Carlo

## Scientific question

How can we integrate in moderate or high dimension when structured grids are
too expensive, while retaining reproducible randomized uncertainty evidence?

## Geometric picture

Sobol points fill the unit cube more evenly than independent random points.
Digital scrambling produces independent randomized replicates without
discarding the low-discrepancy structure inside each replicate.

:::{important}
A deterministic Sobol formula has no sampling confidence interval. Uncertainty
statements require independent scrambled replicates and apply only to the
declared randomized construction.
:::

## Derivation

A Sobol coordinate is a base-two digital construction:

```{math}
:label: eq-multidim-sobol
x_{n,j}
=
\sum_{k=1}^{B}b_k(n)v_{j,k}\pmod 2,
```

where \(v_{j,k}\) are direction numbers. With \(R\) scrambled replicates,

```{math}
:label: eq-multidim-rqmc-mean
\widehat{I}
=\frac{1}{R}\sum_{r=1}^{R}\widehat{I}_r,
\qquad
s_R^2=\frac{1}{R-1}\sum_{r=1}^{R}
(\widehat{I}_r-\widehat{I})^2.
```

The fixed-look Student interval has half-width

```{math}
:label: eq-multidim-rqmc-student
h=t_{1-\alpha/2,R-1}\frac{s_R}{\sqrt{R}}.
```

Sequential stopping instead uses bounded empirical-Bernstein evidence and a
predeclared spending schedule satisfying

```{math}
:label: eq-multidim-rqmc-spending
\sum_{k=1}^{K}\alpha_k\leq\alpha.
```

## Computational cost

Fixed cost is \(R2^m\) evaluations for level \(m\). Sequential schedules reuse
the declared maximum point and replicate capacities. Vectorized payloads can
increase memory even when logical point count is unchanged.

## What the estimator means

Fixed scrambled methods report replicate standard error or a fixed-look
confidence half-width. Sequential intervals require finite declared estimate
bounds. Coverage is empirically calibrated only for the frozen campaign.

## JAX and differentiation

Keys, scramble configuration, level, and replicate schedule are static or
stopped. Replay differentiates the accepted randomized formula for the supplied
key; it does not differentiate the randomization policy.

## Quantities and units

Finite quantity axes are normalized before points are mapped into the domain.
Sequential `estimate_bounds` may be quantities compatible with the result.

## Worked astrophysical example

A separable survey selection integral is

```{math}
:label: eq-multidim-selection
I=\int_{\Omega}
\prod_{j=1}^{d}
\operatorname{sigmoid}\!\left(\frac{c_j-x_j}{w_j}\right)d^d x.
```

Randomized QMC is useful when the number of survey or population coordinates
makes tensor rules impractical.

## Failure modes

:::{warning}
Replicates are not independent if keys are reused incorrectly. Fixed-look
intervals cannot be repeatedly inspected as sequential intervals, and bounded
sequential evidence is invalid when the declared estimate bounds are false.
:::

## Audit recipe

Record the root key policy, scramble, bits, level, replicate count or schedule,
confidence level, estimate bounds, status, coverage definition, evaluations,
and the exact evidence artifact.

## Warranted claim

Jaxstro provides deterministic Sobol, fixed scrambled Sobol, and bounded
sequential scrambled Sobol integration. Confidence claims are real-scalar and
limited to their declared randomized assumptions.
