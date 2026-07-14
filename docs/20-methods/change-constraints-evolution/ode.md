---
title: Fixed-step ODE integration
description: >-
  Euler, midpoint, RK4, and velocity-Verlet with visible truncation errors and
  fixed-route differentiation contracts.
---

## The question this method answers

Given a model for an instantaneous rate of change, what approximate state does
that model predict after a finite time? Jaxstro provides fixed-step methods for
small differentiable calculations. Adaptive step control, events, stiffness
policy, and production solver stacks remain delegated ecosystem concerns.

:::{tip}
Use RK4 as the ordinary high-accuracy fixed-step baseline, midpoint when a
second-order method is sufficient, and Euler mainly as a transparent reference.
Use velocity-Verlet when the model is a separable second-order system and
long-term geometric behavior matters.
:::

## Before computation: what should be true?

The model must be written as a first-order initial-value problem with a fixed
state shape. Choose a step $h$ small relative to the shortest relevant timescale
and make the integration interval, units, and initial state explicit. These
explicit methods are not a stiffness remedy.

:::{important}
Plan a step-refinement study before interpreting the trajectory. A solver that
runs, differentiates, and returns finite values can still have unacceptable
discretization error.
:::

Scientific state representations and fixed-shape PyTrees are connected in
[](../../30-representations/parameters-state/pytrees-as-scientific-state.md).

## Define the mathematical objects

An initial-value problem specifies

```{math}
\frac{dy}{dt}=f(y,t),\qquad y(t_0)=y_0,
```

where $t$ is the independent variable, $y(t)\in\mathbb{R}^m$ is the state, and
$f$ is the right-hand side (RHS). A grid uses $t_n=t_0+nh$ and a numerical state
$y_n\approx y(t_n)$. For order $p$, local truncation error measures one exact
step started from the exact state; global error measures accumulated trajectory
error after $O(1/h)$ steps over a fixed interval.

## Derive the method

Taylor expansion gives $y(t+h)=y(t)+h f(y,t)+O(h^2)$, yielding Euler:

```{math}
:label: eq-ode-euler
y_{n+1}=y_n+h k_1,\qquad k_1=f(y_n,t_n).
```

Euler estimates the slope only at the interval start. Explicit midpoint first
predicts the half-step state and evaluates a centered slope:

```{math}
:label: eq-ode-midpoint
k_1=f(y_n,t_n),\qquad
k_2=f\!\left(y_n+\frac{h}{2}k_1,t_n+\frac{h}{2}\right),\qquad
y_{n+1}=y_n+h k_2.
```

Classical RK4 combines four slope samples:

```{math}
:label: eq-ode-rk4
\begin{aligned}
k_1 &= f(y_n,t_n),\\
k_2 &= f(y_n+\tfrac{h}{2}k_1,t_n+\tfrac{h}{2}),\\
k_3 &= f(y_n+\tfrac{h}{2}k_2,t_n+\tfrac{h}{2}),\\
k_4 &= f(y_n+h k_3,t_n+h),\\
y_{n+1} &= y_n+\frac{h}{6}(k_1+2k_2+2k_3+k_4).
\end{aligned}
```

For a method of order $p$, the distinction between one-step and accumulated
error is

```{math}
:label: eq-ode-local-global
\text{local truncation error}=O(h^{p+1}),\qquad
\text{global error over fixed time}=O(h^p).
```

Thus Euler, midpoint, and RK4 have global orders one, two, and four for a smooth,
well-resolved problem. A smaller observed order is evidence about the regime,
implementation, precision, or model regularity.

## What the algorithm actually does

`euler`, `midpoint`, and `rk4` apply their step functions in a fixed-length
`lax.scan`. `ODEResult.t` has shape `(num_steps + 1,)`; `ODEResult.y` has shape
`(num_steps + 1, ...)` and includes the initial state. `solve_fixed_step`
dispatches the literal methods `"euler"`, `"midpoint"`, `"rk2"`, and `"rk4"`;
an unknown name raises `ValueError`.

`velocity_verlet` accepts acceleration `a(q, t)` and returns `VerletResult(t,
q, v)` with the same leading history length. It updates position using the old
acceleration, evaluates acceleration at the new position and time, then averages
the old and new accelerations in the velocity update. No method adapts $h$,
detects events, retries failures, or estimates error at runtime.

## What JAX differentiates

JAX differentiates the fixed sequence of arithmetic operations and RHS calls.
Gradients may flow through initial conditions, floating step values, and
parameters closed over by a smooth RHS. `rhs`, `acceleration`, `method`, and
`num_steps` are static when users JIT-compile wrappers around these APIs.

:::{warning}
The AD result is the sensitivity of the discretized fixed-step trajectory. It is
not automatically the sensitivity of the exact differential equation. Event
times, adaptive controller decisions, discontinuous RHS branches, and stiffness
are outside this contract.
:::

## Using it in Jaxstro

```python
import jax.numpy as jnp

from jaxstro.numerics.ode import solve_fixed_step


def decay(y, t):
    del t
    return -0.4 * y


result = solve_fixed_step(
    decay,
    y0=jnp.array([1.0, 2.0]),
    t0=0.0,
    dt=0.05,
    num_steps=40,
    method="rk4",
)
assert result.t.shape == (41,)
assert result.y.shape == (41, 2)
```

`y0` may have any fixed array shape that `rhs(y, t)` preserves. Time and state
dtypes follow the initial state conversion, so enable the intended precision
before creating arrays.

## How to audit the result

For a problem with an analytic solution, compute errors at $h$, $h/2$, and
$h/4$. An order-$p$ method should approach
$E(h)/E(h/2)\approx 2^p$ before roundoff dominates. Without an analytic
solution, compare nested refinements at common times. Separately compare AD for
a final-state scalar against a central finite difference in each claimed smooth
parameter. For velocity-Verlet, also track problem-specific invariants such as
energy or angular momentum; bounded drift is evidence, not an exact guarantee.

The executable audit map is in [](../../60-validation/methods/validation-methods.md).

## Where the claim stops

Jaxstro does not choose a scientifically adequate step, detect stiffness, bound
global error, locate events, or provide adaptive-step derivatives. Fixed-step
agreement on a smooth test problem does not validate a downstream dynamical
model or long-time behavior.

## Connected ideas

:::{seealso}
Relate model equations to executable programs in
[](../../10-foundations/models-and-computation/from-relations-to-differentiable-programs.md),
represent states with
[](../../30-representations/parameters-state/pytrees-as-scientific-state.md),
inspect owner signatures in [](../../50-api/change-constraints/ode.md), and
connect numerical evidence to [](../../60-validation/validation.md). The
delegated adaptive-method guide is [](./adaptive-differential-equations.md).
:::
