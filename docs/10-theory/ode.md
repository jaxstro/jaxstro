---
title: Fixed-step ODE integration
description: >-
  Scan-friendly Euler, midpoint, RK4, and velocity-Verlet helpers for small
  differentiable scientific integration problems.
---

`jaxstro.numerics.ode` provides fixed-step integration helpers, not a general
ODE solver suite. Adaptive step-size control, event handling, stiffness policy,
and large solver stacks belong in dedicated libraries or downstream packages.

The first slice covers explicit one-step methods for first-order systems and a
symplectic-style velocity-Verlet helper for separable second-order systems.

## First-order systems

The first-order API assumes a right-hand side

```{math}
\frac{dy}{dt} = f(y, t),
```

with call signature `rhs(y, t)`. `euler`, `midpoint`, and `rk4` return an
`ODEResult(t, y)` containing the initial state plus every fixed step. The scan
length `num_steps` is explicit and static under `jax.jit`.

The methods are intentionally ordinary:

- `euler_step` uses the forward Euler update.
- `midpoint_step` is the explicit midpoint / RK2 update.
- `rk4_step` is the classical fourth-order Runge-Kutta update.

`solve_fixed_step(..., method=...)` is only a small dispatcher over those methods.
It does not choose step sizes or diagnose stiffness.

## Velocity-Verlet

`velocity_verlet(acceleration, q0, v0, t0, dt, num_steps)` integrates separable
second-order systems with

```{math}
\frac{d^2q}{dt^2} = a(q, t).
```

It returns `VerletResult(t, q, v)` with the initial position and velocity
included. For harmonic-oscillator-like systems, the method has much better
long-term energy behavior than Euler at the same step size, but it is still a
fixed-step approximation and should be validated against the system scale.

## AD contract

All runtime loops use fixed-count `lax.scan`, so gradients flow through every
step. The validation suite checks FD-vs-AD agreement for RK4 final states and
velocity-Verlet final positions on smooth toy systems.

The package does not hide static arguments: `rhs`, `acceleration`, method choice,
and `num_steps` must be static when users JIT-compile wrappers around these
helpers.
