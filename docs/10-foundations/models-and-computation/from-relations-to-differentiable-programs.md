---
title: From mathematical relations to differentiable programs
description: What JAX differentiates, how program structure matters, and where scientific ownership remains.
---

# From mathematical relations to differentiable programs

Use this page when the derivative of an executed JAX program must be separated
from the derivative of an ideal mathematical relation.

A root, interpolant, or likelihood begins as a mathematical relation. A program
evaluates an approximation to that relation with finite precision, control flow,
iteration counts, branch choices, and transformation rules. JAX differentiates
that executed structure unless a public contract deliberately defines another
derivative.

## Program structure is part of the map

An `if` statement, minimum, sort, table-cell choice, stopping rule, or clipping
operation can change the selected branch. Smooth code within a branch does not
make the branch boundary smooth. A fixed scan gives static execution shape and
an auditable finite map; it does not turn finite iterations into an exact
mathematical solution.

JAX represents nested parameter and state structures as a **PyTree**. `jit`
stages compatible numerical work for compilation. `vmap` batches a function by
adding an array axis. Both preserve particular value semantics under documented
conditions; neither guarantees that expensive inactive lanes avoid all physical
work.

For an iterative relation with exact solution $x^\star(\theta)$, a finite
executed program instead exposes a finite map after $K$ steps:

```{math}
x_K(\theta) = \Phi_K(\theta),
\qquad
\frac{d x_K}{d\theta}.
```

The derivative on the right is the derivative of the executed finite map. It
need not equal $d x^\star/d\theta$ unless an additional implicit-derivative
contract and its assumptions are satisfied.

## Which derivative is being computed?

A smooth finite algorithm ordinarily has a pathwise derivative of its executed
operations. JAX also permits custom derivative rules: `custom_jvp`, `custom_vjp`,
and `custom_root` can assign derivative semantics that do not follow the primal
iteration history. A **value-first** iterative solver promises a converged value
and telemetry but no ideal-root derivative. Jaxstro's **implicit derivative**
records caller assertions of uniqueness and smoothness, then checks numerical
convergence, finite state, residual, width, and conditioning gates before
exposing the custom-root sensitivity.

This distinction generalizes. Differentiating an interpolator at an interior
point is different from differentiating the discrete selection of an
interpolation cell. Differentiating a likelihood is different from validating
the probability model.

## Ownership boundaries

Jaxstro owns generic, dependency-light mechanics and their numerical/JAX
contracts. A downstream scientific package owns domain equations, admissibility,
state acceptance, retry policy, data semantics, and physical validation. An
astronomy example can motivate a primitive without moving stellar, particle, or
instrument policy into Jaxstro runtime logic.

## Try the running case

For the two-channel measurement, suppose the parameters are found by an
iterative solve. If a stopping rule, a clamp, or a selected interpolation
cell changes as the measurements change, predict whether ordinary pathwise AD
is differentiating an ideal solution or one selected finite program path.

## Worked audit

Ordinary pathwise AD differentiates the selected finite program path. That can
be exactly the desired finite-map sensitivity, but it does not by itself certify
the derivative of an ideal root or of a branch-crossing model. Record the branch
and solver status, then use the derivative contract matching the scientific
claim.

:::{figure} ../figures/executed-program-map.svg
:name: fig-executed-program-map
:alt: A smooth ideal solution is compared with finite solver iterates and a branch-selected output. Ordinary automatic differentiation follows the selected finite path, while an implicit derivative needs additional assumptions and numerical gates.

Finite iteration and branch selection define an executable map. A certified
implicit sensitivity is a separate claim with separate assumptions and gates.
:::

## Predict

Write the ideal relation, the finite algorithm, static and dynamic values,
selected branches, expected transforms, and desired derivative semantics. Name
where the scientific package takes ownership.

## Compute

Use JAX-compatible arrays and control flow, explicit fixed shapes, typed status,
and deterministic telemetry. Apply `jit`, `vmap`, scan, JVP, or VJP only where
the public contract and domain support them.

## Audit

Compare values with analytic or independent methods; inspect JAXPR when
architecture matters; test JIT/VMAP parity; compare derivatives with finite
differences on smooth branches; and force failure paths to verify fail-closed
behavior.

## State the warranted claim

Name whether the evidence supports a value, finite-map derivative, certified
implicit sensitivity, transform compatibility, or physical conclusion. Do not
let one passing evidence category silently stand in for another.

## Misconception check

> "Differentiable" does not mean smooth everywhere, physically correct, or
> differentiable with respect to every captured value. Compilation is not
> validation, and a finite gradient is not an implicit function theorem proof.

For the concrete implicit-root distinction, see the
[](../../20-methods/change-constraints-evolution/rootfinding.md) method,
the [](../../50-api/change-constraints/rootfinding.md) public contract, and
the [](../../60-validation/numerical/implicit-root-gradients.md) qualified
gradient evidence. Continue to the [](../../20-methods/methods.md) module
chapters and the generated [](../../50-api/research-infrastructure/contracts.md)
contract registry.
