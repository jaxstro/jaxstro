---
title: Parameters, constraints, and transforms
description: >-
  A selective bridge between structured Equinox models, constrained physical values,
  and flat unconstrained vectors.
---

Use this page when an optimizer or sampler needs a flat unconstrained vector but the
scientific model must remain a structured PyTree with explicit free and fixed leaves.

:::{important} Implemented Jaxstro capability
`jaxstro.params` implements `Parameterization` plus identity, positive, soft-positive,
and bounded scalar bijectors. It is a bridge, not an inference framework.
:::

## Representation contract

| Contract field | Current representation |
| --- | --- |
| Mathematical object | A static selection of free array leaves and per-leaf bijections between a structured physical PyTree and one unconstrained parameter vector. |
| Physical convention | `forward(u)` maps unconstrained values to physical values; the forward log absolute Jacobian supplies the change-of-variables term. |
| Runtime owner | `jaxstro.params` owns `Parameterization`, `Identity`, `Exp`, `Softplus`, and `Sigmoid`. |
| Shape and unit policy | Free leaves may have arbitrary array shapes and are raveled into shape `(n,)`; fixed/static leaves pass through, and leaf units remain caller-owned. |
| Transform boundary | `from_vector`, `to_vector`, and analytic bijectors support fixed-structure `jit`, `vmap`, and `grad`; leaf selection and PyTree structure are static. |
| Evidence | Unit and ML-integration tests check round trips, free-leaf order, fixed leaves, analytic log-Jacobians, extreme values, and gradient flow. |
| Downstream interpretation boundary | Optimizer choice, priors, likelihoods, posterior inference, identifiability, and the scientific meaning of each parameter remain downstream. |

## Structured state and flat coordinates

Let $T$ select the free leaves of a model $m$, apply inverse bijectors, and ravel the
result. Reconstruction applies the forward maps and recombines fixed leaves:

```{math}
:label: eq-parameters-vector-bridge

\mathbf{u}=T^{-1}(m_{\mathrm{free}}),
\qquad
m'=\operatorname{combine}\left(T(\mathbf{u}),m_{\mathrm{fixed}}\right).
```

`Parameterization.from_where` marks leaves through an Equinox selector.
`from_filter` accepts a low-level boolean PyTree. Ordering follows JAX PyTree leaf
order, not the textual order in a selector tuple.

## Constraint transforms

`Identity` leaves a value unconstrained. `Exp` and `Softplus` map real values to the
positive domain. `Sigmoid(lo, hi)` maps into a finite open interval. Each bijector
implements an analytic

```{math}
:label: eq-parameters-log-jacobian

\log\left|\frac{d\,\operatorname{forward}(u)}{du}\right|,
```

which can be summed by `log_det_jacobian` when a density is evaluated in
unconstrained coordinates.

```python
from jaxstro.params import Exp, Parameterization, Sigmoid

parameterization = Parameterization.from_where(
    model,
    where=lambda item: (item.radius, item.fraction),
    transforms=(Exp(), Sigmoid(0.0, 1.0)),
)
vector = parameterization.to_vector(model)
updated = parameterization.from_vector(model, vector)
```

:::{warning} Reconstruction does not rerun model initialization
If a model caches a leaf derived in `__init__`, replacing its source leaf does not
refresh the cache. Fit the leaf the observable actually reads or recompute derived
state explicitly at use time.
:::

The tests verify the structural and derivative contracts in
[](#eq-parameters-vector-bridge) and [](#eq-parameters-log-jacobian). They do not show
that a parameter is identifiable or that a chosen transform gives a well-behaved
posterior.
