---
title: Tensor-product integration
description: Fixed Gaussian products and globally adaptive nested tensor formulas.
---

# Tensor-product integration

## Scientific question

When is it reasonable to resolve every coordinate direction at the same
one-dimensional order? Tensor products are the direct answer for smooth,
low-dimensional integrands.

## Geometric picture

Place a one-dimensional rule on each axis, then evaluate every Cartesian
combination of nodes. In two dimensions the nodes form a lattice; in higher
dimensions the same construction becomes expensive quickly.

:::{tip}
Use a fixed tensor rule when polynomial exactness or a reproducible node set is
the scientific requirement. Use the adaptive tensor method when global nested
refinement is meaningful and the dimension is still modest.
:::

## Derivation

Given one-dimensional rules
\(Q_j[g]=\sum_{i=1}^{n_j}w_{j,i}g(x_{j,i})\), their product is

```{math}
:label: eq-multidim-tensor-rule
(Q_1\otimes\cdots\otimes Q_d)[f]
=
\sum_{i_1=1}^{n_1}\cdots\sum_{i_d=1}^{n_d}
\left(\prod_{j=1}^{d}w_{j,i_j}\right)
f(x_{1,i_1},\ldots,x_{d,i_d}).
```

For a common order \(n\), the point count is

```{math}
:label: eq-multidim-tensor-cost
N=n^d.
```

`AdaptiveTensorClenshawCurtis` compares nested global levels:

```{math}
:label: eq-multidim-tensor-refinement
E_L=\left\|Q_L^{\otimes d}[f]-Q_{L-1}^{\otimes d}[f]\right\|.
```

## Computational cost

Fixed cost is exactly the product of per-axis node counts. Global adaptive
refinement inherits exponential growth and fixed-capacity storage. Dimensions
five through eight therefore require explicit memory and runtime planning.

## What the estimator means

Fixed tensor rules return `ErrorKind.UNAVAILABLE`; exactness is a property of
the declared rule and integrand class, not a measured error. Adaptive tensor
rules return a refinement difference, which is evidence from two nested
formulas rather than a universal bound.

## JAX and differentiation

Both methods support first-order accepted-formula replay. JAX differentiates
the integrand, explicit `args`, and smooth finite bounds while treating the
chosen level and work record as fixed evidence.

## Quantities and units

Each tensor point is exposed as a `CoordinatePoint` in heterogeneous quantity
mode. The integrand may inspect each axis with `x.axis(j)`. Unit conversion is
performed outside the raw numerical engine.

## Worked astrophysical example

A diagonal bounded Gaussian factorizes:

```{math}
:label: eq-multidim-gaussian-mass
\int_{-L_1}^{L_1}\cdots\int_{-L_d}^{L_d}
\prod_{j=1}^{d}
\frac{\exp[-x_j^2/(2\sigma_j^2)]}
{\sqrt{2\pi}\sigma_j}\,d^d x
=
\prod_{j=1}^{d}
\operatorname{erf}\!\left(\frac{L_j}{\sqrt{2}\sigma_j}\right).
```

This is a strong tensor-rule validation because normalization and each
diagonal second moment have closed forms.

## Failure modes

:::{warning}
Tensor rules do not discover discontinuities or narrow features between
nodes. A smooth-looking result can still be wrong, and \(n^d\) becomes
prohibitive even when every one-dimensional rule is inexpensive.
:::

## Audit recipe

Report per-axis rule, order or level, dimension, exact point count, payload
shape, dtype, status, and comparison to a closed-form moment or higher-order
formula.

## Warranted claim

Tensor products are exact for their declared polynomial class and effective
for the frozen smooth validation cases. They are not a general solution to
high-dimensional integration.
