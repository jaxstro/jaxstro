---
title: Choosing a multidimensional integration method
description: A research-task guide to tensor, cubature, sparse-grid, and randomized QMC methods.
---

# Choosing a multidimensional integration method

## Scientific question

Which method produces evidence that matches the scientific structure, not just
a number? Start with geometry, smoothness, effective dimension, uncertainty
requirements, and the derivative you need.

## Geometric picture

Tensor rules fill every coordinate combination. Cubature partitions physical
space. Sparse grids navigate a level lattice. Randomized QMC spreads replicated
digital nets across the whole domain.

:::{tip}
Choose the estimator meaning before choosing the implementation: exactness
class, embedded disagreement, sparse surplus, or randomized confidence.
:::

## Derivation

A useful first comparison is the work model:

```{math}
:label: eq-multidim-choice-work
N_{\mathrm{tensor}}\sim n^d,
\qquad
N_{\mathrm{RQMC}}=R2^m,
```

while adaptive cubature and sparse grids have problem-dependent work

```{math}
:label: eq-multidim-choice-adaptive
N_{\mathrm{adaptive}}
=N(\epsilon_{\mathrm{abs}},\epsilon_{\mathrm{rel}},
d,f,\text{capacities}).
```

No method removes dependence on the integrand class.

## Computational cost

| Scientific structure | First method to try | Main cost warning |
| --- | --- | --- |
| Smooth, low dimension, polynomial moment | `TensorProduct` | \(n^d\) |
| Localized smooth structure | `AdaptiveCubature` | regional growth |
| Mixed smoothness or low effective dimension | `Smolyak` | index and node growth |
| Unknown anisotropy | `AdaptiveSmolyak` | frontier capacity |
| Moderate or high dimension, deterministic estimate | `Sobol` | no error estimate |
| Randomized uncertainty required | `ScrambledSobol` | replicate cost |
| Sequential randomized stopping | `AdaptiveScrambledSobol` | valid bounds required |

## What the estimator means

Fixed rules may have no runtime estimator. Adaptive evidence is formula
disagreement, not a universal error bound. Randomized intervals describe
replicate uncertainty under a declared randomization.

## JAX and differentiation

All methods support first-order replay. Prefer fixed formulas when stable,
repeatable derivatives matter more than adaptive work. Always pass live
parameters through `args`.

## Quantities and units

Use raw normalized coordinates when that is the clearest model. Use
`Hyperrectangle.from_axes` when heterogeneous physical units materially improve
auditability. Quantity mode is optional and alpha.

## Worked astrophysical example

For a four-dimensional stellar population moment, start with a tensor rule if
the integrand is a low-degree separable polynomial. Move to a sparse grid for
smooth nonseparable corrections, or randomized QMC when selection effects add
many coordinates.

```{math}
:label: eq-multidim-choice-population
I=\int p(M,Z,t,D)\,S(M,Z,t,D)\,g(M,Z,t,D)\,dM\,dZ\,dt\,dD.
```

## Failure modes

:::{warning}
Do not interpret a faster method as a better scientific method unless domain,
nodes or family, tolerances, work, truth error, derivative error, and hardware
are calibrated. The B4 benchmark warrants no universal superiority claim.
:::

## Audit recipe

Write down why the geometry fits, what smoothness you expect, which estimator
you will trust, the stopping capacities, derivative mode, units, independent
truth check, and the fallback method.

## Warranted claim

Jaxstro now spans four complementary finite-hyperrectangle strategies. Method
choice remains a scientific modeling decision, and Phase C geometries are
outside this guide.
