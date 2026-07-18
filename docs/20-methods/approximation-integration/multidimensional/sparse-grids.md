---
title: Sparse-grid integration
description: Smolyak hierarchy, exact node reuse, and dimension-adaptive surplus evidence.
---

# Sparse-grid integration

## Scientific question

How can we avoid resolving every high-order interaction in a tensor product?
Sparse grids retain selected hierarchical tensor increments and are effective
when the integrand has mixed smoothness or low effective dimension.

## Geometric picture

Imagine each axis carrying a refinement level. A multi-index selects one
combination of levels. A downward-closed set fills the lower-left portion of
this index lattice before admitting more expensive interactions.

:::{tip}
Choose fixed `Smolyak` for reproducible isotropic or anisotropic level sets.
Choose `AdaptiveSmolyak` when frontier surpluses are meaningful evidence for
which direction to refine next.
:::

## Derivation

Define nested one-dimensional differences
\(\Delta_\ell=Q_\ell-Q_{\ell-1}\), with \(Q_0=0\). For a downward-closed set
\(\mathcal{I}\),

```{math}
:label: eq-multidim-smolyak
A_{\mathcal{I}}[f]
=
\sum_{\boldsymbol{\ell}\in\mathcal{I}}
\left(\Delta_{\ell_1}\otimes\cdots\otimes
\Delta_{\ell_d}\right)[f].
```

Dimension-adaptive selection uses a profit such as

```{math}
:label: eq-multidim-sparse-profit
P_{\boldsymbol{\ell}}
=
\frac{\|\Delta_{\boldsymbol{\ell}}[f]\|}
{\max(1,\Delta N_{\boldsymbol{\ell}})},
```

and reports the active-frontier sum

```{math}
:label: eq-multidim-sparse-frontier
E_{\mathcal{F}}
=
\sum_{\boldsymbol{\ell}\in\mathcal{F}}
\|\Delta_{\boldsymbol{\ell}}[f]\|.
```

## Computational cost

Cost is the number of unique nested nodes, not the sum of all tensor sizes.
Jaxstro coalesces nodes by exact dyadic-angle identity. Capacities independently
bound accepted indices, frontier rows, nodes, and evaluations.

## What the estimator means

`ErrorKind.SPARSE_GRID_SURPLUS` measures the executed hierarchy's frontier.
It can be conservative, as in the frozen level-5 exponential case, or miss a
feature not represented by the current index set.

## JAX and differentiation

First-order replay materializes the accepted sparse formula and differentiates
its coalesced weighted sum. Index admission and surplus diagnostics are not
differentiated.

## Quantities and units

Heterogeneous axes work through `CoordinatePoint`. `ProductMeasure` can attach
one Lebesgue or weighted component to each axis while sharing explicit `args`.

## Worked astrophysical example

A population moment can combine heterogeneous coordinates:

```{math}
:label: eq-multidim-population-moment
I=\int M^2(1+Z)tD^2\,dM\,dZ\,dt\,dD.
```

When high-order interactions are weak, sparse increments can require far fewer
unique evaluations than a uniform tensor rule.

## Failure modes

:::{warning}
Sparse does not mean universally cheap. Narrow peaks, discontinuities, or
strong high-order interactions can exhaust `max_indices` or `max_nodes` before
the requested tolerance is supported.
:::

## Audit recipe

Save the level or initial level, anisotropy, all four capacities, unique-node
count, accepted-index count, frontier evidence, status, and an independent
truth or convergence sequence.

## Warranted claim

Jaxstro provides fixed and dimension-adaptive Smolyak formulas with exact node
reuse and honest surplus evidence. The surplus is not a universal error bound.
