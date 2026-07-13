---
title: From mathematical relations to differentiable programs
description: What JAX differentiates, how program structure matters, and where scientific ownership remains.
---

# From mathematical relations to differentiable programs

A **mathematical relation** states how ideal quantities are connected. A
computational method approximates or evaluates that relation. An **executed
program** additionally contains data representation, finite precision, control
flow, iteration counts, branch choices, and transformation rules. Differentiable
programming composes derivatives through that executable structure.

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

## Which derivative is being computed?

A smooth finite algorithm has a pathwise derivative of its executed operations.
A **value-first** iterative solver promises a robust value and telemetry but no
ideal-root derivative. A separately certified **implicit derivative** applies a
mathematical sensitivity rule only after uniqueness, smoothness, convergence,
residual, width, and conditioning gates pass.

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
let one passing class silently stand in for another.

## Misconception check

> “Differentiable” does not mean smooth everywhere, physically correct, or
> differentiable with respect to every captured value. Compilation is not
> validation, and a finite gradient is not an implicit function theorem proof.

Continue to the [](../10-theory/index.md) module chapters and the generated
[](../40-api/contracts.md) contract registry.

