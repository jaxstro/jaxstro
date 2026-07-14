---
title: Optimization helpers
---

# Optimization helpers

## Owner import path

`jaxstro.numerics.optimization`

## Purpose

Small optimizer-agnostic loss, line-search, and convergence mechanics.

## Public records and callables

`LineSearchResult`, `squared_loss`, `huber_loss`, `pseudo_huber_loss`,
`objective_summary`, `armijo_backtracking`, `gradient_inf_norm`,
`relative_step_norm`, and `convergence_summary`.

## Shape and dtype expectations

Losses accept floating residual arrays. Diagnostics reduce arrays to scalar
summaries; line search requires a scalar objective and compatible PyTrees.

## JAX transforms and AD classification

Losses are smooth except at their documented piecewise boundaries. The Armijo
helper uses a fixed iteration count suitable for JIT; branch-selected line
search paths are not implicit optimizer derivatives.

## Failure behavior

Invalid shapes and callback errors propagate. Non-finite objectives remain
visible in the returned diagnostics rather than being silently accepted.

## Contract and evidence links

See [](../../20-methods/change-constraints-evolution/optimization.md) and
[](../../60-validation/validation.md).

## Canonical import example

```python
from jaxstro.numerics.optimization import armijo_backtracking
```
