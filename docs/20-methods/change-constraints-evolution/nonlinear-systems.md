---
title: Nonlinear systems and fixed points
description: Mathematical background and evidence boundaries for delegated nonlinear solving.
---

# Nonlinear systems and fixed points

Use this page when several coupled constraints must vanish together and you
need to distinguish a locally accurate Newton step from a globally reliable
solve.

:::{important} Ecosystem guide
[Optimistix](https://docs.kidger.site/optimistix/) owns general nonlinear root,
fixed-point, least-squares, and minimization solvers. Jaxstro does not duplicate
that runtime machinery.
:::

## The scientific question

A nonlinear system asks for a state whose coupled residuals all vanish. The
scientific task is not only to obtain a small residual. It is to decide whether
the variables, residual scales, constraints, and selected solution branch make
that state meaningful.

Local convergence describes behavior once an iterate is sufficiently near a
regular solution. Globalization describes how an algorithm tries to reach such
a neighborhood from a less favorable initial state. A successful local model
does not imply global convergence or uniqueness.

## Mathematical objects

Let $\mathbf{F}:\mathbb{R}^n\rightarrow\mathbb{R}^n$ map a state
$\mathbf{x}$ to residuals. Its Jacobian has entries
$J_{ij}=\partial F_i/\partial x_j$. Residual units can differ by equation, so
both state scaling and residual scaling are part of the numerical problem.

A termination policy normally combines a scaled residual norm, a scaled step
norm, iteration limits, and finite-value checks. A singular or ill-conditioned
Jacobian can make the Newton direction undefined or extremely sensitive even
when the current residual is finite.

## Core derivation

Linearize the residual around the current iterate and require the linear model
to vanish at the next state. This gives the Newton system:

```{math}
:label: eq-nonlinear-newton-step

\begin{aligned}
\mathbf{F}(\mathbf{x}) &= \mathbf{0}, \\
\mathbf{F}(\mathbf{x}_k+\Delta\mathbf{x}_k)
&\approx \mathbf{F}(\mathbf{x}_k)
+\mathbf{J}(\mathbf{x}_k)\Delta\mathbf{x}_k, \\
\mathbf{J}(\mathbf{x}_k)\Delta\mathbf{x}_k
&=-\mathbf{F}(\mathbf{x}_k), \\
\mathbf{x}_{k+1}&=\mathbf{x}_k+\Delta\mathbf{x}_k.
\end{aligned}
```

The linear solve in [](#eq-nonlinear-newton-step) is a local model, not a
certificate that the full nonlinear residual decreases. Damping, line search,
trust regions, or continuation can globalize a method, but each introduces its
own branch and stopping behavior.

## What the ecosystem already owns

[Optimistix](https://docs.kidger.site/optimistix/) is the ecosystem owner
for nonlinear solves and related fixed-point, least-squares, and minimization
problems. Its solver selection, iteration mechanics, transforms, termination,
and result diagnostics belong there. Linear subproblems should use the
operator and solver abstractions chosen by that ecosystem rather than a second
Jaxstro solver stack.

## What Jaxstro may add

A later consumer-driven adapter could attach unit-aware variable and residual
scales, record the solver configuration in provenance, and connect termination
telemetry to a scientific evidence report. No such adapter is implemented or
promised by this guide.

The adapter boundary would remain narrow: it could prepare a problem and
interpret evidence, but it would not fork Newton, fixed-point, or globalization
algorithms.

## Evidence required before implementation

Before an adapter could be called ready, it would need:

- problems with analytic solutions and independent residual checks;
- scale-change tests showing that equivalent unit choices do not alter the
  physical answer;
- singular and ill-conditioned Jacobian cases that fail or warn explicitly;
- multiple-root cases that report dependence on initialization and branch;
- transform tests for the exact derivative contract the adapter claims; and
- provenance round trips for tolerances, solver choice, status, and telemetry.

## Claim boundary

:::{warning}
A small residual is not evidence of uniqueness, good conditioning, global
convergence, or a scientifically correct model. Differentiating an executed
iteration is also not automatically the derivative of an ideal solution map.
:::

This page derives the local Newton model and states ownership. It does not
benchmark Optimistix, certify any solver for a particular science problem, or
claim a Jaxstro nonlinear-systems runtime.

## Connected foundations and methods

Review [](../../10-foundations/mathematical-objects/linear-algebra-language-of-change.md)
for Jacobians and linear maps, and
[](../../10-foundations/models-and-computation/sensitivity-conditioning-identifiability.md)
for conditioning and identifiability. Compare the scalar contract in
[](./rootfinding.md), objective-based formulations in [](./optimization.md),
and derivative mechanics in [](./autodiff.md).
