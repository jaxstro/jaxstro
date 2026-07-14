---
title: Quasi-Monte Carlo
description: Low-discrepancy integration, randomized replication, and evidence requirements for a planned capability.
---

# Quasi-Monte Carlo

Use this page when an integral over a unit hypercube may benefit from structured
space filling and you need to separate deterministic approximation from
randomized uncertainty estimation.

:::{important} Planned Jaxstro capability
The proposed `jaxstro.numerics.qmc` module does not exist. This page defines
background and evidence gates, not an importable API or implementation result.
:::

## The scientific question

Quasi-Monte Carlo (QMC) replaces independent random samples with points designed
to cover a domain evenly. It asks whether lower discrepancy can reduce
integration error for the dimension, transform, and integrand structure at
hand.

Deterministic low-discrepancy points, independent random samples, and replicated
randomized scrambles are different experimental designs. They must not share an
uncertainty claim merely because each produces points in $[0,1)^d$.

## Mathematical objects

Let $f:[0,1)^d\rightarrow\mathbb{R}$ and
$I=\int_{[0,1)^d}f(\mathbf{u})\,d\mathbf{u}$. A point set has a construction,
dimension, ordering, and discrepancy criterion. A transform may map
$\mathbf{u}$ into a nonuniform target distribution, and that transform can
change smoothness and effective dimension.

A deterministic sequence supports reproducible approximation. It does not provide an uncertainty estimate by itself. Independent random sampling supports
classical sampling-error estimates. Replicated randomized scrambles preserve
structured coverage within each replicate while using between-replicate
variation for an empirical uncertainty estimate.

## Core derivation

For either a deterministic point set or one randomized replicate, the equal
weight estimator is:

```{math}
:label: eq-qmc-estimator

\widehat{I}_{N}=\frac{1}{N}\sum_{n=1}^{N}f(\mathbf{u}_{n}).
```

For $R \geq 2$ independent scrambles, apply [](#eq-qmc-estimator) separately to
obtain $\widehat{I}_{N,r}$, then estimate between-replicate variation:

```{math}
:label: eq-rqmc-replicate-variance

\widehat{\mathrm{Var}}(\overline{I})
=\frac{1}{R(R-1)}\sum_{r=1}^{R}
\left(\widehat{I}_{N,r}-\overline{I}\right)^2,
\qquad
\overline{I}=\frac{1}{R}\sum_{r=1}^{R}\widehat{I}_{N,r}.
```

This variance describes the randomized replicate design. It is not available
from one unscrambled deterministic sequence and is not a universal bound on
integration bias.

## What the ecosystem already owns

[JAX random numbers](https://docs.jax.dev/en/latest/random-numbers.html) provide
explicit keyed randomness and transformed array computation. JAX does not, by
that fact alone, establish a Jaxstro QMC sequence, scramble convention, error
model, or scientific validation claim.

Reference sequence definitions and independently generated fixtures would be
needed before choosing a runtime construction. This page intentionally does not
select an implementation source or claim ecosystem performance.

## What Jaxstro may add

The proposed `jaxstro.numerics.qmc` boundary may eventually own fixed-shape
Sobol and Latin-hypercube construction, explicit scramble keys, replicated
plans, discrepancy diagnostics, metadata, and provenance. The module would own
scientific conventions around those outputs, not a general probability or
inference framework.

Any API sketch remains deferred until a separate design chooses supported
sequence limits, dimension behavior, scrambling rules, dtypes, and failure
states.

## Evidence required before implementation

Readiness would require:

- exact point-prefix comparisons with independent authoritative fixtures;
- construction invariants for range, uniqueness where applicable, and nesting;
- deterministic reproduction for seeds, keys, dimensions, and skip settings;
- replicated-scramble calibration on analytic integrals;
- comparisons with independent Monte Carlo under matched evaluation budgets;
- stress tests for dimension, transforms, non-smooth integrands, and precision;
- `jit` and `vmap` checks for the explicitly supported fixed shapes; and
- strict provenance for sequence, scramble, key ownership, and replicate plan.

## Claim boundary

:::{warning}
Low discrepancy does not guarantee a better estimate for every integrand or
dimension. A single deterministic sequence does not provide an uncertainty
estimate, and a replicate standard error is not a proof that all bias has been
resolved.
:::

No convergence rate, speedup, supported sequence, or runtime capability is
claimed here.

## Connected foundations and methods

Review [](../../10-foundations/mathematical-objects/probability-and-distributions.md)
for probability measures and
[](../../10-foundations/models-and-computation/sensitivity-conditioning-identifiability.md)
for information geometry. Connect explicit keys in [](./random.md), ordinary
sampling and resampling in [](./sampling.md), fixed quadrature in
[](../approximation-integration/quadrature.md), and grid construction in
[](../discrete-space/grids.md).
