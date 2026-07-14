---
title: Optimization helpers
description: >-
  Gradient descent, Armijo line search, robust residual losses, and explicit
  convergence diagnostics for scientific objectives.
---

## The question this method answers

Given a scalar objective $F(x)$, how can we move parameters toward a smaller
value while retaining enough diagnostics to decide whether the result is
credible? Jaxstro owns small loss, line-search, and stopping mechanics. It does
not own an optimizer stack, parameter schedule, or scientific acceptance policy.

:::{tip}
Use an optimizer library for full training or inference loops. Use these helpers
when a small auditable search, robust residual loss, or common convergence report
is the reusable numerical object.
:::

## Before computation: what should be true?

The objective must return one finite floating scalar for parameter arrays of a
fixed shape. Scales and units must make the residual definition meaningful.
For line search, the proposed direction $p_k$ should be a descent direction:
$\nabla F(x_k)^\mathsf{T}p_k<0$. Tolerances should be tied to a requested
scientific accuracy, not copied from an unrelated problem.

:::{important}
Optimization can only answer the question encoded by the objective. Establish
the parameter representation, bounds or transforms, residual scaling, and
identifiability assumptions before interpreting a minimum.
:::

See [](../../30-representations/parameters-state/parameters-and-transforms.md)
for Jaxstro's parameter bridge and its cached-derived-leaf boundary.

## Define the mathematical objects

Let $x\in\mathbb{R}^n$ be a parameter vector, $F:\mathbb{R}^n\to\mathbb{R}$ a
scalar objective, and $g_k=\nabla F(x_k)$ its gradient. A search direction
$p_k\in\mathbb{R}^n$ specifies the proposed motion, and a positive step length
$\alpha_k$ controls its size.

For residuals $r_i(x)$, the half-squared loss is $\rho(r)=r^2/2$. The Huber loss
is quadratic near zero and linear in the tails; the pseudo-Huber loss is the
smooth approximation

```{math}
\rho_\delta(r)=\delta^2\left(\sqrt{1+(r/\delta)^2}-1\right).
```

## Derive the method

Steepest descent chooses $p_k=-g_k$, giving

```{math}
:label: eq-gradient-descent
x_{k+1}=x_k-\alpha_k\nabla F(x_k).
```

A step that is too large can increase $F$. Armijo backtracking tests the more
general direction $p_k$ against the sufficient-decrease inequality

```{math}
:label: eq-armijo-decrease
F(x_k + \alpha_k p_k)
\le F(x_k)+c_1\alpha_k\nabla F(x_k)^\mathsf{T}p_k,
\qquad 0<c_1<1.
```

Starting from $\alpha_0$, the tested sequence is
$\alpha_i=\alpha_0\beta^i$ with $0<\beta<1$. The right-hand side decreases
below $F(x_k)$ only for a descent direction.

No single stopping statistic proves convergence. Jaxstro reports three:

```{math}
:label: eq-optimization-convergence
\frac{\lVert x_{k+1}-x_k\rVert_2}
{\max(\lVert x_k\rVert_2,s_0)}\le\tau_x,
\qquad
\lVert g_{k+1}\rVert_\infty\le\tau_g,
\qquad
\frac{|F_{k+1}-F_k|}{\max(|F_k|,1)}\le\tau_F.
```

The helper declares convergence only when all three inequalities pass.

## What the algorithm actually does

`squared_loss`, `huber_loss`, and `pseudo_huber_loss` act elementwise.
`objective_summary` returns loss, mean loss, RMSE, maximum absolute residual,
and element count; supplied weights multiply squared residuals and their sum is
the normalizer. A nonpositive normalizer is replaced by one for finite division,
but the original zero-weight scientific meaning is not repaired.

`armijo_backtracking` evaluates a fixed `max_steps` sequence with `lax.scan` and
records the first accepted candidate in `LineSearchResult(step, value, accepted,
iterations)`. If none passes, it returns the last backtracked candidate with
`accepted=False`. The scan count, not the first acceptance, fixes the executed
trace.

## What JAX differentiates

The smooth losses and diagnostics compose with JAX array transforms. Squared
and pseudo-Huber losses are smooth on their floating domains. Huber is
piecewise smooth and has a kink at $|r|=\delta$. Armijo's acceptance predicate
selects a discrete branch, so a derivative of the returned step is a derivative
of the selected finite program, not an implicit derivative of an optimum.

:::{warning}
`accepted=True`, a small step, or a small objective change does not establish a
minimum. Inspect gradient size, parameter motion, objective change, non-finite
values, and problem-specific residuals together. Do not differentiate through
the branch-selected line-search result as though it were a smooth optimizer map.
:::

## Using it in Jaxstro

```python
import jax
import jax.numpy as jnp

from jaxstro.numerics.optimization import (
    armijo_backtracking,
    convergence_summary,
)


def objective(x):
    return 0.5 * jnp.sum((x - jnp.array([1.0, -2.0])) ** 2)


x = jnp.array([4.0, 1.0])
grad = jax.grad(objective)(x)
search = armijo_backtracking(objective, x, -grad, grad, max_steps=12)
x_new = x + search.step * (-grad)
diagnostics = convergence_summary(
    x_new=x_new,
    x_old=x,
    grad=jax.grad(objective)(x_new),
    loss_new=objective(x_new),
    loss_old=objective(x),
)
```

The parameter arrays and direction must share a shape. `f` returns a scalar;
`max_steps` and `f` are static when the line search is itself JIT-compiled.
Diagnostics reduce array inputs to scalar JAX arrays.

## How to audit the result

First verify the gradient against a central finite difference along at least one
declared direction. Confirm $g_k^\mathsf{T}p_k<0$, reproduce every tested Armijo
inequality, and retain `accepted` and `iterations`. Run from multiple initial
conditions when local minima are possible. Refine all three convergence
tolerances and check that the scientifically relevant outputs remain stable.
For robust losses, place FD probes away from the Huber kink and include outliers
large enough to exercise the tails.

Executable method audits are indexed in [](../../60-validation/methods/validation-methods.md).

## Where the claim stops

These helpers do not prove convexity, uniqueness, identifiability, global
optimality, or posterior correctness. They do not manage PyTrees, constraints,
optimizer state, batching policy, or second-order solves. A converged numerical
summary supports only the tested objective and representation.

## Connected ideas

:::{seealso}
Build the mathematical context with
[](../../10-foundations/models-and-computation/models-inference-information.md),
connect parameters to
[](../../30-representations/parameters-state/parameters-and-transforms.md),
check the exact owner surface in [](../../50-api/change-constraints/optimization.md),
and calibrate evidence with [](../../60-validation/validation.md). Derivative
products used by curvature methods are in [](./autodiff.md).
:::
