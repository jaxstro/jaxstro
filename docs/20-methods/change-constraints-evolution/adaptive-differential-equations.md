---
title: Adaptive differential equations
description: Error control, ownership, and evidence boundaries for delegated adaptive integration.
---

# Adaptive differential equations

Use this page when a differential equation needs variable step sizes and you
must interpret what local error control does, and does not, say about the
solution.

:::{important} Ecosystem guide
[Diffrax](https://docs.kidger.site/diffrax/) owns adaptive ODE, SDE, and CDE
solvers, step-size controllers, adjoints, events, and solver diagnostics.
Jaxstro does not duplicate those algorithms.
:::

## The scientific question

An adaptive integrator allocates more steps where the solution is difficult and
fewer where it is smooth. The scientific question is whether a local numerical
error estimate, measured against meaningful component scales, is an adequate
control signal for the observable and time interval of interest.

Local error control limits an estimate attached to one proposed step. Global
solution error accumulates and propagates across accepted steps. It depends on
stability, stiffness, events, tolerances, precision, and the dynamics, so it is
not equal to the requested local tolerance.

## Mathematical objects

Consider $d\mathbf{y}/dt=\mathbf{f}(t,\mathbf{y},\boldsymbol{\theta})$.
An embedded or paired method proposes a state and an error estimate
$\widehat{\mathbf{e}}$. Each component is compared with a scale that combines
an absolute floor and relative magnitude. The scaled norm, controller order,
safety factor, growth limits, and rejection history are part of the executed
algorithm.

Component units matter. One shared absolute tolerance can be meaningless when
state leaves represent different physical dimensions or very different
scientific scales.

## Core derivation

For a scalar component, define a tolerance scale and normalized error. A common
controller pattern is:

```{math}
:label: eq-adaptive-de-controller

\begin{aligned}
s &= \mathrm{atol}+\mathrm{rtol}|y|, \\
E &= \left\|\widehat{\mathbf{e}}/\mathbf{s}\right\|, \\
\mathrm{accept} &\iff E\leq 1, \\
h_{\mathrm{next}}
&=h\,\mathrm{clip}\left(
\gamma E^{-1/(p+1)},q_{\min},q_{\max}
\right).
\end{aligned}
```

In [](#eq-adaptive-de-controller), $p$ identifies the error model order,
$\gamma$ is a safety factor, and the clip prevents extreme step changes. An
implementation must define the norm, the zero-error branch, what state enters
$\mathbf{s}$, and whether rejected proposals affect diagnostics or adjoints.

## What the ecosystem already owns

[Diffrax](https://docs.kidger.site/diffrax/) is the runtime owner for adaptive
differential-equation solving in this ecosystem. Solver formulas, controllers,
dense output, events, adjoint strategies, result states, and transform behavior
belong to Diffrax. Jaxstro's fixed-step ODE helpers remain a smaller, separate
contract; they are not an adaptive-solver framework.

## What Jaxstro may add

A future adapter may translate unit-bearing state into an explicit per-leaf
tolerance policy, record solver and controller settings, and connect convergence
studies to provenance. It may also define evidence envelopes for particular
scientific observables. No adaptive Jaxstro adapter exists today.

Any adapter must expose Diffrax ownership and return its statuses rather than
masking them behind a success boolean.

## Evidence required before implementation

Required evidence would include:

- analytic and high-accuracy reference solutions across nonstiff and stiff
  examples relevant to the claimed scope;
- tolerance sweeps for state error and downstream observables;
- unit-rescaling parity for heterogeneous state leaves;
- event-time and discontinuity tests with explicitly bounded claims;
- adjoint comparisons against independent finite differences on smooth cases;
- failure cases for step underflow, nonfinite states, and exhausted steps; and
- deterministic provenance for solver, controller, tolerances, and statuses.

## Claim boundary

:::{warning}
Passing a local accept/reject test does not certify global accuracy, stability,
event accuracy, or an adjoint. Tightening tolerances is evidence only when an
independent convergence study shows the target observable stabilizing.
:::

This page does not claim that one solver or adjoint is best, does not report
performance, and does not add adaptive integration to Jaxstro.

## Connected foundations and methods

Use [](../../10-foundations/mathematical-objects/functions-units-scales.md) for
scales and units, and
[](../../10-foundations/models-and-computation/from-relations-to-differentiable-programs.md)
for the distinction between a mathematical relation and an executed program.
Compare the current fixed-step methods in [](./ode.md), derivative contracts in
[](./autodiff.md), and cumulative integral mechanics in
[](../approximation-integration/cumulative-trapz.md).
