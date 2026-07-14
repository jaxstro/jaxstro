---
title: Interpolation boundary policies
description: Separate branch-stable interior derivatives from clamp, fill, and reject behavior.
---

# Interpolation boundary policies

Use this page when interior interpolation derivatives and boundary policies
must be tested without conflating their claims.

**Research question.** Which interpolation claims survive when a query crosses
a knot or leaves the tabulated domain?

## Predict

Predict exact recovery of affine data inside a cell. Separately predict the
observable results of clamp, fill, and reject policies outside the grid. Do not
extend an interior derivative prediction across a discrete cell or policy change.

## Compute

```bash
uv run --no-sync python -m examples.investigations.interpolation_boundary_policies
```

The example evaluates public one-dimensional and regular-grid APIs on analytic
affine fixtures and records interior value/derivative errors and boundary events.

## Audit

Compare values and derivatives with the analytic affine map. Exercise all three
boundary policies deliberately. Then follow the callable contracts in
[](../../40-api/contracts.md) to their validation targets and limitations.

## Misconception check

> Smooth interpolation within a selected cell does not imply differentiability
> of the cell selection or the clamp/fill/reject boundary.

## State the warranted claim

The affine fixtures validate branch-stable interior values and derivatives plus
explicit boundary policy behavior. They make no derivative claim at knots or
policy transitions.
