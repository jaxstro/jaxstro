---
title: Adaptive cubature
description: Local Genz-Malik refinement, embedded evidence, and bounded regional work.
---

# Adaptive cubature

## Scientific question

How can an integral spend work only where a smooth multidimensional integrand
needs local resolution? Adaptive cubature subdivides the most uncertain
region instead of refining the entire domain.

## Geometric picture

Begin with one hyperrectangle. Evaluate a symmetric rule, estimate local
error, split the selected region along one axis, and repeat. The active leaves
form a binary partition of the original domain.

:::{important}
Local adaptivity helps when difficult structure occupies a limited part of the
domain. It does not remove the dimensional growth of the underlying symmetric
rule.
:::

## Derivation

For a region \(R\), the Genz-Malik construction evaluates high- and low-degree
weighted sums on shared symmetric points:

```{math}
:label: eq-multidim-cubature-embedded
Q_R^{(h)}=\sum_{i=1}^{N_d}w_i^{(h)}f(\boldsymbol{x}_i),
\qquad
Q_R^{(l)}=\sum_{i=1}^{N_d}w_i^{(l)}f(\boldsymbol{x}_i).
```

The local evidence is

```{math}
:label: eq-multidim-cubature-error
E_R=\left\|Q_R^{(h)}-Q_R^{(l)}\right\|,
\qquad
E=\sum_{R\in\mathcal{L}}E_R.
```

The controller stops when \(E\leq\epsilon_{\mathrm{abs}}+
\epsilon_{\mathrm{rel}}\|Q\|\), or when a declared capacity is reached.

## Computational cost

Each split adds two child-rule evaluations. Cost depends on dimension, active
regions, payload size, and how quickly local error concentrates. Storage is
bounded by `max_regions` and evaluations by `max_evaluations`.

## What the estimator means

`ErrorKind.EMBEDDED_RULE` measures disagreement between two formulas on the
executed partition. It is not an exact bound. `MAX_REGIONS` and
`MAX_EVALUATIONS` preserve the best accumulated value while stating why the
requested tolerance was not certified.

## JAX and differentiation

The primal controller is JIT-compatible. Replay reconstructs the accepted leaf
formulas and differentiates their weighted sum to first order. Refinement
decisions and diagnostic fields are stopped.

## Quantities and units

Quantity axes and weighted measures are normalized before the raw cubature
controller. Every density must declare its unit, and the result unit includes
all coordinate-width factors.

## Worked astrophysical example

For a localized likelihood contribution,

```{math}
:label: eq-multidim-localized-peak
f(\boldsymbol{x})=
\exp\!\left[-\alpha\|\boldsymbol{x}-\boldsymbol{\mu}\|_2^2\right],
```

local subdivision can focus on the neighborhood of
\(\boldsymbol{\mu}\), unlike global tensor refinement.

## Failure modes

:::{warning}
Discontinuities that do not align with the partition can produce slow or
misleading convergence. In the frozen \(d=6\) continuous Genz case, bounded
capacity remains a documented limitation rather than a universal success.
:::

## Audit recipe

Record rule, dimension, tolerances, `max_regions`, `max_evaluations`, status,
active leaves, evaluations, embedded estimate, truth error, and the exact
integrand parameters.

## Warranted claim

Adaptive Genz-Malik cubature is validated on the declared smooth and Genz
cases through dimension eight. Accuracy outside those cases remains
problem-dependent and capacity-bounded.
