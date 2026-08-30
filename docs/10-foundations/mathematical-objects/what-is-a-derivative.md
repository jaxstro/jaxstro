---
title: What is a derivative?
description: Change, local linear maps, sensitivity, and evidence in differentiable programs.
---

# What is a derivative?

Use this page when a gradient or sensitivity needs a precise mathematical and
scientific interpretation.

A derivative is one idea seen from several scientific viewpoints. It is a
**local rate of change**, the **best local linear map** approximating a function,
and a **scientific sensitivity** describing how a stated output responds to a
stated perturbation under stated assumptions. Connecting those views is the
foundation of differentiable programming.

## View 1: local change

For a scalar function, the derivative at $x$ is the limiting ratio

```{math}
f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}.
```

This says more than "slope on a graph." It names which input changes, which
output changes, their units, and the local point. For Newtonian gravity,
derivatives with respect to mass and separation answer different scientific
questions. For Stefan-Boltzmann luminosity, temperature sensitivity carries
luminosity-per-temperature units.

## View 2: the best local linear map

Near $x$, a differentiable function satisfies

```{math}
f(x+\delta x) \approx f(x) + Df(x)[\delta x].
```

$Df(x)$ is a linear map from input perturbations to output perturbations. In
coordinates it becomes a derivative, gradient, or Jacobian depending on shapes
and conventions. A directional derivative applies the map to one direction.
This formulation scales from one variable to millions.

A **JVP** computes $Jv$: push one input perturbation forward. A **VJP** computes
$u^T J$: pull one output sensitivity backward. Reverse-mode gradients are VJPs
of a scalar output, not a magical new type of derivative. Tangent vectors carry
input perturbations; cotangent vectors carry linear measurements of output
perturbations.

## View 3: scientific sensitivity and evidence

A derivative always belongs to a mapping. In statistics, the gradient of a log
likelihood is the **likelihood score**. A Hessian records local curvature. Fisher
information summarizes expected score geometry under a specified probabilistic
model. None of these is meaningful without the model, parameterization, data
distribution, and point of evaluation.

For a root defined by $f(x^\star,\theta)=0$, an **implicit sensitivity** can be
derived on a unique smooth branch:

```{math}
\frac{dx^\star}{d\theta}=-\frac{\partial f/\partial\theta}{\partial f/\partial x}.
```

This derivative becomes unstable when the denominator is poorly conditioned and
is not justified when uniqueness or smoothness fails.

## The derivative of a relation versus an executed program

JAX differentiates the represented map according to primitive and custom
transformation rules. Ordinary pathwise AD follows selected branches, finite
iteration counts, clamps, and represented arithmetic. But `custom_jvp`,
`custom_vjp`, and `custom_root` deliberately install derivative semantics that
need not differentiate the executed primal iteration history. Jaxstro's
certified implicit API uses the latter pattern. It therefore distinguishes
smooth finite-map derivatives, value-first iterative maps, validation-only
finite-difference checks, and separately defined implicit derivatives.

At a branch boundary, knot, absolute-value kink, clipping threshold, repeated
root, or discrete selection, a classical derivative can fail to exist or become
branch-dependent. A finite number returned by AD does not settle that question.

## Why this motivates differentiable programming

Differentiable programming makes local linear maps composable across a program.
The same machinery supports physical sensitivity analysis, inverse problems,
uncertainty propagation, optimization, control, experimental design, and
probabilistic inference. Its power comes from composition; its scientific
validity still comes from model and derivative contracts.

## Try the running case

Consider two calibrated measurements of one source, predicted by a parameter
vector $\theta$. At a reference point, decide whether the question is about the
change in predicted measurements $d$ under a proposed $\delta\theta$, or about
the change in a scalar loss after those measurements are compared with data.

## Worked audit

The first question is a JVP, $J\,\delta\theta$, and retains the direction and
units of the predicted measurement change. The second is a VJP after a scalar
objective has supplied an output cotangent. They can share implementation
machinery, but they answer different scientific questions and should not be
reported as the same sensitivity.

## Predict

Name the input, output, evaluation point, units, held-fixed quantities, desired
derivative meaning, smooth branch, and expected sign before invoking AD.

## Compute

Choose forward or reverse products from input/output geometry. Retain solver
certificates and control-flow information. Do not hide a nonsmooth selection
inside an unqualified gradient claim.

## Audit

Compare with an analytic derivative, central finite differences at several step
sizes, complex-step where admissible, or an independent implicit calculation.
Probe both sides of suspected boundaries and monitor conditioning.

## State the warranted claim

State "the AD derivative of this represented smooth map agrees with this
independent check on this domain." Claim an ideal implicit sensitivity only when
uniqueness and smoothness are independently justified and declared, and the
runtime convergence, finiteness, residual, width, and slope gates pass.

## Misconception check

> A derivative is not just a symbolic rule and not just the number produced by
> `grad`. It is a local linear claim about a specified map. A correct derivative
> of an inadequate model remains a correct derivative of an inadequate model.

Continue to
[](../models-and-computation/sensitivity-conditioning-identifiability.md),
[](../models-and-computation/from-relations-to-differentiable-programs.md), or
Jaxstro's [](../../20-methods/change-constraints-evolution/autodiff.md) and
[](../../20-methods/change-constraints-evolution/rootfinding.md) chapters.
