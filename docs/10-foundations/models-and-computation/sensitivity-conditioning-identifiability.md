---
title: Sensitivity, conditioning, and identifiability
description: Distinguish derivative magnitude, numerical stability, and learnable parameter combinations.
---

# Sensitivity, conditioning, and identifiability

Use this page when a changing output, unstable computation, or parameter
degeneracy needs to be diagnosed without conflating those problems.

Sensitivity, conditioning, and identifiability all ask how conclusions respond
to change, but they are not synonyms.

## Sensitivity

Sensitivity describes how an output changes under a specified input
perturbation. It can be local or global, dimensional or normalized, and tied to
a physical parameter, initial state, observation, or numerical control. A large
derivative may be physically expected rather than numerically problematic.

For a local observable map, the first-order response is

```{math}
\delta d \approx J\,\delta\theta.
```

When a covariance $C$ is an appropriate positive-definite measurement-error
model, the local information geometry is often summarized by

```{math}
F = J^{\mathsf{T}}C^{-1}J.
```

This Fisher form is conditional on the local linearization and covariance
assumptions; it is not a proof of global identifiability.

## Conditioning

**Conditioning** describes how perturbations in inputs or arithmetic can be
amplified by a problem. It belongs to the problem and representation; stability
describes how an algorithm handles that problem. Parallax inversion becomes
poorly conditioned as parallax approaches zero. A root sensitivity becomes
poorly conditioned when the residual slope approaches zero.

Scaling can improve numerical representation. It cannot remove physical
degeneracy, and a sophisticated solver cannot make an ill-conditioned question
well-conditioned.

## Identifiability and degeneracy

**Identifiability** asks whether different parameter values imply distinguishable
predictions under a model and experiment. A **degeneracy** is a direction or
manifold along which predictions change little or not at all. Locally, a
Jacobian null space contains a **null direction** invisible to first order.

Identifiability may be structural, practical, local, or global. A full-rank local
Jacobian does not exclude distant alternative solutions. A prior can select
among weakly identified values without changing the likelihood information.

## Derivative evidence

**Automatic differentiation** computes derivatives of represented programs
accurately up to floating arithmetic, but it does not decide whether the intended
mathematical derivative exists. **Finite differences** supply an independent
discretization with truncation and cancellation error. Agreement across a stable
step range is evidence; disagreement is a diagnostic, not a reason to choose the
more convenient answer.

Use analytic derivatives, JVPs/VJPs, central differences, convergence, singular
values, and model perturbations as complementary evidence. The correct audit
depends on branch smoothness and the scientific question.

## Try the running case

Consider two calibrated measurements of one source and two source parameters.
If the Jacobian columns are nearly parallel, decide which statement follows:
the code has a bug, the numerical representation is poorly scaled, or the
measurements weakly distinguish one parameter combination.

## Worked audit

Near-parallel columns establish a local weak direction of the stated observable
map. They do not by themselves diagnose a code bug or forbid a stable forward
calculation. Rescaling can improve arithmetic; an additional observable or a
justified prior can change the inference. Those are different interventions and
must be evaluated against the scientific question.

## Predict

Predict sensitive directions, expected signs and units, possible degeneracies,
boundary behavior, and which conclusions should change under rescaling or model
perturbation.

## Compute

Evaluate local derivatives and normalized sensitivities; inspect Jacobian or
Fisher singular directions; record numerical tolerances, parameterization, and
the experiment defining the observable map.

## Audit

Compare AD with analytic or finite-difference results away from nonsmooth
boundaries. Change scaling and parameterization, probe global alternatives, and
test whether added observations constrain the predicted null directions.

## State the warranted claim

Distinguish "this derivative is numerically validated," "this inverse is locally
well-conditioned," and "these parameters are identifiable under this experiment."
Evidence for one statement does not automatically support the others.

## Misconception check

> A small gradient can mean insensitivity, a stationary point, poor scaling, or
> a blocked branch. A large condition number is not fixed by reporting more
> digits. A prior-constrained parameter is not necessarily data-identified.

Continue to [](./from-relations-to-differentiable-programs.md).
