---
title: Root values and sensitivities
description: Separate a safeguarded root value from a certified implicit derivative.
---

# Root values and sensitivities

**Research question.** When does a computed root value support a parameter
sensitivity claim?

**Connected foundations.**
[](../10-foundations/mathematical-objects/what-is-a-derivative.md) and
[](../10-foundations/models-and-computation/from-relations-to-differentiable-programs.md).

## Predict

For $x^2-\theta=0$ on the positive branch, predict the root, derivative sign,
units, conditioning, bracket behavior, and failure if uniqueness or slope
evidence is rejected. Distinguish the value-first branch history from the ideal
implicit sensitivity before running code.

## Compute

Run the repository-owned example:

```bash
uv run --no-sync python -m examples.investigations.root_values_and_sensitivities
```

The command prints a complete report: prediction, the required units-explicit
metric table, audit status and evidence, and the calibrated warranted claim. A
failed audit is visible and gives the command a nonzero exit status. The source
retains typed solver status, final bracket, evaluation count, derivative
certificate, and the analytic fixture.

## Audit

Check the analytic square-root identity, signed residual, final bracket width,
certificate, and analytic implicit derivative independently. Then inspect the
registered [](../validation/rootfinding-performance.md) and
[](../validation/implicit-root-gradients.md) evidence.

## Misconception check

> A converged value-first root does not inherit the derivative of an ideal root.
> The certified API makes that separate claim only after its gates pass.

## State the warranted claim

The analytic positive branch independently supplies uniqueness and smoothness.
The runtime certificate records those caller assertions and checks convergence,
finite state, residual, width, and slope conditioning. Together they support the
fixture's derivative; they do not prove the assumptions for arbitrary residuals
or downstream physical models.
