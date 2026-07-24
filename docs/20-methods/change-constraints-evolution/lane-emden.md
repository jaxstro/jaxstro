---
title: Lane-Emden self-gravitating spheres
description: >-
  Differentiable isothermal (Bonnor-Ebert) and polytropic Lane-Emden solves with
  a series-seeded origin start and an event-root surface, differentiable in the
  polytropic index.
---

## The question this method answers

Given a self-gravitating sphere in hydrostatic equilibrium with a prescribed
equation of state, what dimensionless density and enclosed-mass profile does that
equilibrium imply, and where is its edge? Jaxstro solves the Lane-Emden equation
in two branches: the polytropic branch (a finite polytrope of index $n$, whose
surface is the first zero of the solution) and the isothermal Bonnor-Ebert branch
(no intrinsic edge -- the truncation radius is a genuine physical input set by the
confining external pressure). Both branches are differentiable, which is what lets
them feed gradient-based inference and hydrostatic initial conditions downstream.

:::{tip}
Use `solve_polytrope` with `xi_max = polytrope_xi1(n)` when you want the physical
finite polytrope truncated at its own surface, and `solve_isothermal(xi_max)` when
the outer radius is externally imposed. Treat the isothermal case as the singular
$n\to\infty$ limit of the polytropic one, not as a large numerical `n`.
:::

## Before computation: what should be true?

The model must be a spherically symmetric, self-gravitating equilibrium whose
pressure-density relation is either a polytrope $P=K\rho^{\gamma}$ (with
$n=1/(\gamma-1)$) or isothermal $P\propto\rho$. Choose `xi_max` and `n_points`
knowing they are static: they size the output grid, not the physics. For a
polytrope, `xi_max` should not exceed the surface $\xi_1$ if you intend to
differentiate (see below). These ordinary-differential solves are singular at the
origin, so the integration starts just off it at `XI_0 = 1e-6`.

:::{important}
Plan the audit before trusting the profile: check the solve against the closed
forms at $n=0,1,5$ and the origin series, and confirm the returned enclosed mass
$m(\xi)$ is monotone wherever the density is positive. A solve that runs and
differentiates can still carry unacceptable truncation or edge error.
:::

Scientific state representations and fixed-shape PyTrees are connected in
[](../../30-representations/parameters-state/pytrees-as-scientific-state.md), and
the model-to-program framing is in
[](../../10-foundations/models-and-computation/from-relations-to-differentiable-programs.md).

## Define the mathematical objects

Hydrostatic equilibrium plus Poisson's equation for a self-gravitating sphere,
nondimensionalized with a density scale $\rho_c$ (the central density) and a
length scale, collapses to a single second-order ordinary differential equation in
the dimensionless radius $\xi$. The dependent variable is $\psi(\xi)$ (isothermal,
with $\rho=\rho_c e^{-\psi}$) or $\theta(\xi)$ (polytropic, with
$\rho=\rho_c\theta^{n}$). The dimensionless enclosed mass $m(\xi)$ is the radial
integral of $4\pi r^2\rho$ in scaled variables. The polytropic index $n$ is the
one traced, differentiable parameter; $\xi_1$, the first zero of $\theta$, is the
polytrope's surface and exists (is finite) only for $n<5$.

## Derive the method

The isothermal (Bonnor-Ebert) branch writes the scaled Poisson equation with an
exponential source and boundary conditions at the center:

```{math}
:label: eq-lane-emden-isothermal
\psi'' + \frac{2}{\xi}\psi' = e^{-\psi},
\qquad \psi(0)=0,\quad \psi'(0)=0,\quad \rho=\rho_c\,e^{-\psi}.
```

The polytropic branch replaces the exponential source with the power law
$-\theta^{n}$ and normalizes the center to unity:

```{math}
:label: eq-lane-emden-polytropic
\theta'' + \frac{2}{\xi}\theta' = -\theta^{n},
\qquad \theta(0)=1,\quad \theta'(0)=0,\quad \rho=\rho_c\,\theta^{n}.
```

Integrating $4\pi r^2\rho$ in scaled variables, and using each equation to
eliminate the second derivative, gives the dimensionless enclosed mass in closed
form on each branch, with a common cubic behavior near the origin:

```{math}
:label: eq-lane-emden-mass
m(\xi)=\xi^2\psi' \ \text{(isothermal)},\qquad
m(\xi)=-\xi^2\theta' \ \text{(polytropic)},\qquad
m(\xi)\to\frac{\xi^3}{3}\ \text{as}\ \xi\to 0.
```

Both right-hand sides carry a $2/\xi$ term that is singular at $\xi=0$, so the
solve cannot start from the bare boundary condition. Substituting a Taylor
expansion into each equation fixes the low-order coefficients and seeds the state
at `XI_0`:

$$
\psi(\xi)=\frac{\xi^2}{6}-\frac{\xi^4}{120}+O(\xi^6),
\qquad
\theta(\xi)=1-\frac{\xi^2}{6}+\frac{n\,\xi^4}{120}+O(\xi^6).
$$

The polytrope surface $\xi_1$ is the first root of $\theta$. It is located as a
differentiable event root -- the implicit function theorem applied to the solver's
`diffrax.Event` -- rather than as a grid `argmin`, which would have no usable
gradient in $n$.

## What the algorithm actually does

`solve_isothermal` and `solve_polytrope` build the corresponding first-order
`diffrax.ODETerm`, seed the state at `XI_0` from the series above, and integrate
with adaptive `Tsit5` under a `PIDController(rtol=1e-8, atol=1e-10)`, saving on a
`linspace(XI_0, xi_max, n_points)`. Each returns a `LaneEmdenSolution` with fields
`xi`, `y` ($\psi$ or $\theta$), `dy`, `m`, and `dm`, all length `n_points`. The
enclosed-mass derivative `dm` is supplied analytically ($\xi^2 e^{-\psi}$ or
$\xi^2\theta^{n}$), not by differencing. `polytrope_xi1` runs the same term with a
`diffrax.Event` whose condition is $\theta=0$, closed by an `optimistix.Newton`
root finder, and returns the scalar edge. Past the first zero the polytropic
source $\theta^{n}$ is floored at zero (a non-integer power of a negative number is
NaN); that floored continuation is not physical.

## What JAX differentiates

`jit`, `vmap`, and `grad` are supported. The gradient in the polytropic index $n$
flows through the adaptive solve and through the `polytrope_xi1` event root via the
implicit function theorem, so $\mathrm{d}\xi_1/\mathrm{d}n$ is well defined. On the
isothermal branch the truncation radius `xi_max` is a real physical input, and the
edge mass is differentiable in it. `xi_max` and `n_points` are static and must not
be differentiated.

:::{warning}
Differentiating `solve_polytrope` in `n` with `xi_max` fixed **beyond** $\xi_1(n)$
can return NaN gradients even when the forward value is finite: the floored
$\theta^{n}$ has an undefined derivative where $\theta=0$. Differentiate with
`xi_max = polytrope_xi1(n)` (the physical edge). For $n\ge 5$ no event fires and
`polytrope_xi1` runs to its give-up radius; the caller must reject that regime.
:::

## Using it in Jaxstro

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jaxstro.numerics.lane_emden import polytrope_xi1, solve_polytrope

# n = 1 has the closed-form solution theta = sin(xi)/xi with surface xi_1 = pi.
xi1 = polytrope_xi1(1.0)
assert abs(float(xi1) - jnp.pi) < 1e-4

solution = solve_polytrope(1.0, xi_max=float(jnp.pi), n_points=200)
assert solution.xi.shape == (200,)
assert solution.m.shape == (200,)

# The event root is differentiable in the polytropic index.
grad_xi1 = jax.grad(lambda n: polytrope_xi1(n))(1.5)
assert jnp.isfinite(grad_xi1)
```

`n` is a traced scalar and may be differentiated; `xi_max` and `n_points` size the
output grid and are static. Enable the intended precision before creating arrays,
since the adaptive tolerances are tight.

## How to audit the result

Compare against the three closed-form polytropes: $n=0$ gives
$\theta=1-\xi^2/6$ with $\xi_1=\sqrt{6}$; $n=1$ gives $\theta=\sin\xi/\xi$ with
$\xi_1=\pi$; $n=5$ gives $\theta=(1+\xi^2/3)^{-1/2}$, which never reaches zero
(infinite extent). Check the origin behavior against the series
$\psi=\xi^2/6-\xi^4/120$ and $\theta=1-\xi^2/6+n\xi^4/120$, and confirm the
solver's $\xi^4$ leading error shrinks at the expected rate under grid or
tolerance refinement. Verify the enclosed mass $m(\xi)$ approaches $\xi^3/3$ near
the origin and is monotone where the density is positive. Cross-check
$\mathrm{d}\xi_1/\mathrm{d}n$ from the event root against a central finite
difference. The executable audit map is in
[](../../60-validation/methods/validation-methods.md).

## Where the claim stops

Jaxstro solves the dimensionless equilibrium; it does not choose a physically
adequate `xi_max`, restore dimensions, or model stability, rotation, magnetic
support, or time evolution. Agreement at the closed-form indices validates the
solver, not any particular astrophysical application of the resulting profile. The
$n\ge 5$ regime and any `xi_max` beyond $\xi_1$ are outside the differentiable
contract and must be handled by the caller.

## Connected ideas

:::{seealso}
This solve underlies the Bonnor-Ebert sphere and other self-gravitating
equilibria, and it feeds hydrax's hydrostatic initial conditions. Relate the model
equations to executable programs in
[](../../10-foundations/models-and-computation/from-relations-to-differentiable-programs.md),
represent tabulated solutions with
[](../../30-representations/parameters-state/pytrees-as-scientific-state.md),
inspect owner signatures in
[](../../50-api/change-constraints/lane-emden.md), and connect numerical evidence
to [](../../60-validation/validation.md). The general fixed-step and adaptive
differential-equation surfaces are [](./ode.md) and
[](./adaptive-differential-equations.md).
:::
