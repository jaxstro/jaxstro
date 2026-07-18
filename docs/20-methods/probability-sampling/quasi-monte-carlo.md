---
title: Structured randomness
description: Sobol integration, principled randomization, fixed-look intervals, and bounded sequential evidence.
---

# Structured randomness

Use this page when a finite-dimensional integral may benefit from structured
space filling and you need to distinguish a deterministic approximation from a
randomized uncertainty statement.

:::{important} Current Jaxstro capability
`jaxstro.quad` owns deterministic Sobol integration through dimension
$21{,}201$, digital-shift, LMS-plus-shift, and true nested Owen randomizations,
fixed-look replicated integration, and bounded sequential integration. The
randomized methods require an explicit JAX key and a scalar real integrand.
:::

## The big picture

Ordinary Monte Carlo uses independent points. Quasi-Monte Carlo (QMC) instead
chooses points that cover $[0,1)^d$ systematically. Randomized QMC (RQMC)
randomizes that structured design without discarding it.

```{mermaid}
flowchart LR
  A["Sobol prefix"] --> B["Deterministic estimate"]
  A --> C["Independent valid scrambles"]
  C --> D["Replicate estimates"]
  D --> E["Fixed-look Student-t interval"]
  D --> F["Bounded confidence sequence"]
```

These are different inferential objects. A deterministic prefix gives a
reproducible approximation but no sampling uncertainty. Replicated scrambles
give randomized evidence. Repeatedly inspecting that evidence requires a
sequentially valid policy.

## The estimator

Let $f:[0,1)^d\rightarrow\mathbb{R}$ and

```{math}
:label: eq-qmc-target

I=\int_{[0,1)^d} f(\mathbf{u})\,d\mathbf{u}.
```

For a power-of-two prefix $N=2^\ell$, one Sobol estimate is

```{math}
:label: eq-qmc-estimator

\widehat{I}_{N}
=\frac{1}{N}\sum_{n=0}^{N-1}f(\mathbf{u}_{n}).
```

The first Sobol point is retained; Jaxstro does not silently skip the origin.
A finite hyperrectangle is handled by the shared affine map, signed
orientation, measure density, and Jacobian owners in `jaxstro.quad`.

:::{warning} Deterministic evidence boundary
`Sobol` reports `ErrorKind.UNAVAILABLE` and
`QuadStatus.ERROR_ESTIMATE_UNAVAILABLE`. A low-discrepancy construction is not,
by itself, a runtime error estimator.
:::

## Three randomizations, three honest names

`DigitalShift` applies one coordinatewise bitwise shift. It is inexpensive and
reproducible, but it is not Owen scrambling.

`LinearMatrixScramble` applies a random unit lower-triangular matrix over
$\operatorname{GF}(2)$ and then an independently keyed digital shift. This is
the default for randomized integration.

`OwenScramble` applies a nested bit permutation. For coordinate $j$, bit $b$,
and already-scrambled prefix $p$, its permutation depends on the complete
triple $(j,b,p)$. That prefix dependence is the defining distinction from an
LMS construction.

:::{tip} Choosing a randomization
Start with `LinearMatrixScramble()` for routine replicated integration. Use
`DigitalShift()` when its cheaper transformation is scientifically adequate.
Use `OwenScramble()` when the nested-permutation contract matters enough to
justify its higher runtime cost.
:::

## Fixed-look uncertainty

For $R$ independent scrambles, let $\widehat{I}_{N,r}$ be the estimate from
replicate $r$. Jaxstro computes

```{math}
:label: eq-rqmc-mean-variance

\overline{I}
=\frac{1}{R}\sum_{r=1}^{R}\widehat{I}_{N,r},
\qquad
s^2
=\frac{1}{R-1}\sum_{r=1}^{R}
\left(\widehat{I}_{N,r}-\overline{I}\right)^2.
```

One predeclared inspection uses the half-width

```{math}
:label: eq-rqmc-student-half-width

h_{\mathrm{fixed}}
=t_{1-\alpha/2,R-1}\frac{s}{\sqrt{R}}.
```

`ScrambledSobol` requires $R\geq 8$ and returns
`ErrorKind.CONFIDENCE_INTERVAL_HALF_WIDTH`. The Student-$t$ quantile is stopped
with respect to automatic differentiation and is evaluated with
cancellation-resistant center and survival-probability formulas.

```python
import jax
import jax.numpy as jnp

from jaxstro import quad

result = quad.integrate(
    lambda x: jnp.exp(-jnp.sum(x, axis=-1)),
    quad.Hyperrectangle(jnp.zeros(8), jnp.ones(8)),
    method=quad.ScrambledSobol(level=8, replicates=16),
    key=jax.random.key(19),
    epsabs=1.0e-3,
    epsrel=0.0,
    max_evaluations=16 * 2**8,
    gradient="stop",
)
```

The key is explicit, and replicate $r$ uses `jax.random.fold_in(key, r)`.
Increasing replicate capacity therefore does not rewrite existing replicate
identities.

## Sequential uncertainty

A fixed-look interval must not be reinterpreted after repeated convergence
checks. `AdaptiveScrambledSobol` instead requires a complete static schedule

```{math}
:label: eq-rqmc-schedule

\mathcal{S}
=\bigl((\ell_1,R_1),\ldots,(\ell_K,R_K)\bigr)
```

with monotone strict progress and final growth in both $\ell$ and $R$. Existing
replicates reuse their Sobol prefix, and new replicates receive stable folded
keys.

At inspection $k$, the overall error probability $\alpha$ is allocated as

```{math}
:label: eq-rqmc-alpha-spending

\alpha_k
=\alpha\frac{6}{\pi^2(k+1)^2},
\qquad
\sum_{k=0}^{\infty}\alpha_k=\alpha.
```

Given certified replicate-estimate bounds $A\leq\widehat{I}_{N,r}\leq B$, the
empirical-Bernstein half-width is

```{math}
:label: eq-rqmc-bernstein

h_k
=\sqrt{\frac{2s_k^2\log(2/\alpha_k)}{R_k}}
+\frac{7(B-A)\log(2/\alpha_k)}{3(R_k-1)}.
```

The union bound preserves the overall confidence claim even though the reused
prefixes make inspections dependent.

```python
method = quad.AdaptiveScrambledSobol(
    schedule=((6, 8), (7, 16), (8, 32)),
    estimate_bounds=(0.0, 1.0),
)
```

Direct `estimate_bounds` apply to replicate estimates and may certify a signed
finite measure. `integrand_bounds` are derived automatically only for
`LebesgueMeasure`, including reversed orientation. Weighted-measure derivation
is rejected in Phase B3 because a pointwise integrand bound does not determine
a weighted integral bound without certified measure information.

:::{caution} Tight bounds matter
The sequential guarantee is finite-sample and can be conservative. In the
frozen 128-seed campaign, both sequential cases covered $128/128$ truths, but
their mean half-widths were more than $9{,}000$ times their observed RMSE.
Treat this as honest efficiency evidence: supply the tightest defensible bounds
and enough replicates, or use a fixed-look design when only one inspection is
scientifically intended.
:::

## Astrophysical uses

Structured randomness is useful when a calculation contains a modest number
of continuously varying nuisance coordinates:

- marginalizing calibration, extinction, or distance uncertainties in a
  forward model;
- integrating a smooth selection function over latent population variables;
- propagating a bounded set of halo, orbital, or stellar-population
  uncertainties into one scalar observable;
- evaluating expected survey yield over a finite design hyperrectangle; and
- computing evidence-like normalization integrals when the transform and
  integrand are sufficiently smooth.

The effective dimension matters more than the nominal column count. A
16-dimensional integrand dominated by two coordinates can be a better QMC
target than a discontinuous two-dimensional mask.

## What JAX differentiates

Methods and schedules are static PyTree metadata. Domains and keys may be
dynamic under `jax.jit`; randomized methods support `jax.vmap` over keys and
domains. Float32 coordinates retain at most $24$ digital bits, and float64
coordinates retain at most $53$. Float64 and integer operations above $32$
bits fail eagerly unless `jax_enable_x64=True`.

All B3 methods require `gradient="stop"`. Replay derivatives, multidimensional
quantity certification, and final backend/memory optimization remain Phase B4
work.

## Where the claim stops

The frozen campaign contains exactly four predeclared records and
$3{,}145{,}728$ primary point-integrand evaluations. The two fixed-look cases
covered $124/128$ and $120/128$ truths, both inside the exact 99% binomial
acceptance band $[115,127]$ for nominal 95% coverage. Sequential coverage is
required not to fall below the same lower validity bound; excess coverage is
reported as conservatism rather than misclassified as invalidity.

:::{warning}
Low discrepancy does not guarantee lower error for every integrand. Nominal
fixed-look coverage is not a universal finite-sample theorem, and sequential
coverage requires valid finite bounds. Discontinuities, high effective
dimension, poor transforms, and loose bounds can erase the practical advantage.
:::

## Connected foundations and methods

Review [](../../10-foundations/mathematical-objects/probability-and-distributions.md)
for probability measures and
[](../../10-foundations/models-and-computation/sensitivity-conditioning-identifiability.md)
for conditioning. Connect explicit keys in [](./random.md), fixed quadrature in
[](../approximation-integration/quadrature.md), sparse grids in
[](../approximation-integration/sparse-grid-quadrature.md), and the complete
public surface in [](../../50-api/approximation-integration/quad.md).
