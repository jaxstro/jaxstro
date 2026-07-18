---
title: Differentiating multidimensional integrals
description: First-order accepted-formula replay for parameters, bounds, measures, and quantity axes.
---

# Differentiating multidimensional integrals

## Scientific question

What derivative should JAX return when an adaptive or randomized algorithm
chose its formula using the current parameters? Jaxstro differentiates the
accepted numerical formula, not the discrete controller history.

## Geometric picture

The primal pass selects points, weights, leaves, levels, indices, or replicate
counts. Replay freezes that evidence, reconstructs one weighted sum, and asks
JAX for its first derivative.

:::{important}
Replay answers a local numerical question: how does the accepted formula change
at the current state? It does not differentiate branching decisions or prove
that a nearby state would accept the same formula.
:::

## Derivation

For an accepted formula

```{math}
:label: eq-multidim-replay-formula
\widehat{I}(\theta)
=
\sum_{i=1}^{N}
w_i(\theta)f(x_i(\theta),\theta),
```

replay computes

```{math}
:label: eq-multidim-replay-derivative
\frac{d\widehat{I}}{d\theta}
=
\sum_{i=1}^{N}
\left[
\frac{dw_i}{d\theta}f_i
+w_i\nabla_x f_i^\mathsf{T}\frac{dx_i}{d\theta}
+w_i\frac{\partial f_i}{\partial\theta}
\right].
```

For a moving one-dimensional bound, this recovers the Leibniz terms:

```{math}
:label: eq-multidim-leibniz
\frac{d}{d\theta}\int_{a(\theta)}^{b(\theta)}
f(x,\theta)\,dx
=
f(b,\theta)b'(\theta)-f(a,\theta)a'(\theta)
+\int_a^b\partial_\theta f(x,\theta)\,dx.
```

## Computational cost

Replay requires one accepted-formula evaluation plus JAX's first-order
transformation. Adaptive formulas can carry larger replay storage than their
final scalar value suggests.

## What the estimator means

The derivative inherits the primal formula's approximation error. Error
estimates, statuses, work counts, and controller choices have exact zero or
`float0` tangents and are evidence, not differentiable observables.

## JAX and differentiation

`jax.jvp`, `jax.vjp`, `jax.grad`, `jax.jacfwd`, and `jax.jacrev` are supported
at first order. Nested derivatives fail explicitly. Differentiable scientific
parameters must be passed through `args` or supported finite bounds.

## Quantities and units

Quantity metadata is static. Values are differentiated in normalized axis
units, then results are restored with the correct product unit. Heterogeneous
axis units therefore do not become traced Python objects.

## Worked astrophysical example

For the projected Plummer aperture fraction,

```{math}
:label: eq-multidim-plummer-gradient
F(R,a)=\frac{R^2}{R^2+a^2},
\qquad
\frac{\partial F}{\partial a}
=-\frac{2R^2a}{(R^2+a^2)^2}.
```

The B4 artifact compares the measured replay derivative with this analytic
result.

## Failure modes

:::{warning}
Higher derivatives are unsupported. A coincident axis produces invalid replay
evidence, and a parameter captured only in a Python closure is not a supported
live derivative input.
:::

## Audit recipe

Save the primal value, accepted status and work, differentiated parameter,
tangent direction, replay derivative, analytic or finite-difference reference,
dtype, and formula owner.

## Warranted claim

All Phase B multidimensional methods support first-order accepted-formula
replay for explicit parameters and smooth finite bounds. Controller derivatives
and higher derivatives are intentionally unsupported.
