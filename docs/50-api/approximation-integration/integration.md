---
title: Sampled integration
---

# Sampled integration

## Owner import path

`jaxstro.numerics.integration`

## Purpose

Trapezoid and Simpson integration for sampled arrays, including cumulative
forms.

## Public records and callables

`trapz`, `cumulative_trapz`, `simpson`, and `cumulative_simpson`.

## Shape and dtype expectations

Samples are floating arrays with an explicit integration axis. Coordinates are
one-dimensional or the caller supplies scalar uniform spacing.

## JAX transforms and AD classification

Reductions compose with JIT, VMAP, and AD. The canonical uniform cumulative
trapezoid path sums panels before multiplying by scalar `dx`.

## Failure behavior

Incompatible sample/coordinate lengths raise. The routines do not estimate
discretization error or certify convergence.

## Contract and evidence links

See [](../../20-methods/approximation-integration/cumulative-trapz.md),
[](../../20-methods/approximation-integration/quadrature.md), and
[](../../60-validation/index.md).

## Canonical import example

```python
from jaxstro.numerics.integration import cumulative_trapz
```
