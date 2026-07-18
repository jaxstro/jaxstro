---
title: Sparse-grid quadrature API
description: Exact contracts for fixed and dimension-adaptive Smolyak integration.
---

# Sparse-grid quadrature API

## Owner import path

`jaxstro.quad.sparse`, exposed through `jaxstro.quad`.

## Purpose

Use sparse grids when mixed smoothness or low effective dimension makes a full
tensor product wasteful. Fixed `Smolyak` owns a reproducible index set;
`AdaptiveSmolyak` grows a downward-closed set using hierarchical surplus
evidence.

## Public records and callables

```python
from jaxstro.quad import AdaptiveSmolyak, Smolyak, integrate
```

```python
Smolyak(level, anisotropy=None)
AdaptiveSmolyak(initial_level=1)
```

Both use `integrate` with explicit `max_evaluations`, `max_indices`,
`max_frontier`, and `max_nodes`. The fixed declaration may use a tuple of
positive anisotropy weights whose length matches the domain dimension.

## Shape and dtype expectations

The integrand receives `(n, dimension)` points and returns `(n,)` or
`(n, ...)`. The node axis is reduced. Dimension, payload shape, levels,
anisotropy, and all capacity bounds remain static under JIT. Exact nested-node
identities are used to avoid duplicate evaluations.

## JAX transforms and AD classification

`jit` and `vmap` are supported under the static-shape contract.
`gradient="replay"` differentiates the accepted unique-node weighted formula.
It does not differentiate frontier admission or index selection.
`gradient="stop"` is available explicitly; higher derivatives are not claimed.

## Failure behavior

Malformed declarations or insufficient fixed capacities raise eagerly.
Dynamic invalid domains, nonfinite integrands, and exhausted adaptive
capacities return typed statuses. `QuadError.kind` is
`SPARSE_GRID_SURPLUS`; the reported surplus is refinement evidence, not a
universal bound on true error.

## Contract and evidence links

- [Sparse-grid derivation](../../20-methods/approximation-integration/multidimensional/sparse-grids.md)
- [Method choice](../../20-methods/approximation-integration/multidimensional/choosing-a-method.md)
- [Phase B validation](../../60-validation/numerical/quadrature-multidimensional.md)
- [Truth artifact](../../validation/quad-multidim-truth.json)
- [Replay artifact](../../validation/quad-multidim-replay.json)

## Canonical import example

```python
from jaxstro.quad import AdaptiveSmolyak, Hyperrectangle, integrate

result = integrate(
    lambda x: x[:, 0] ** 2 + x[:, 1] ** 2,
    Hyperrectangle([0.0, 0.0], [1.0, 1.0]),
    method=AdaptiveSmolyak(initial_level=1),
    epsabs=1e-8,
    epsrel=1e-8,
    max_evaluations=20_000,
    max_indices=256,
    max_frontier=512,
    max_nodes=20_000,
    gradient="replay",
)
```

