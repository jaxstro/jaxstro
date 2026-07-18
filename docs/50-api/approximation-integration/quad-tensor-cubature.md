---
title: Tensor products and adaptive cubature API
description: Exact contracts for finite-hyperrectangle tensor and Genz-Malik integration.
---

# Tensor products and adaptive cubature API

## Owner import path

`jaxstro.quad.tensor` and `jaxstro.quad.cubature`, exposed through `jaxstro.quad`.

## Purpose

Use this family for smooth low-dimensional integrals over a finite
`Hyperrectangle`. Tensor products resolve every coordinate combination;
adaptive cubature spends regional work where an embedded Genz-Malik pair
reports disagreement.

## Public records and callables

```python
from jaxstro.quad import (
    AdaptiveCubature,
    AdaptiveTensorClenshawCurtis,
    GenzMalik,
    TensorProduct,
    integrate,
)
```

The public declarations are:

```python
TensorProduct(rules)
AdaptiveTensorClenshawCurtis(initial_level=2)
GenzMalik()
AdaptiveCubature(rule=GenzMalik())
```

All are evaluated through `integrate`. `TensorProduct` requires only
`max_evaluations`; adaptive tensor refinement also uses that bound.
`AdaptiveCubature` additionally requires `max_regions`.

## Shape and dtype expectations

The integrand receives points with shape `(n, dimension)` and returns `(n,)`
or `(n, ...)`. The leading node axis is reduced. Dimension, payload shape,
rule declarations, levels, and capacities are static under JIT. Reference
validation uses float64.

## JAX transforms and AD classification

`jit` and `vmap` are supported with static method configuration. For
cost-sensitive heterogeneous cubature batches, wrap scalar calls in
`jax.lax.map`; select-style `vmap` preserves logical results but may evaluate
inactive child branches. `gradient="replay"` differentiates the accepted
formula once. `gradient="stop"` stops the complete result. Higher derivatives
are outside the contract.

## Failure behavior

Invalid traced domains and nonfinite integrands return typed `QuadStatus`
values. Capacity declarations that cannot hold the requested fixed structure
raise eagerly. A fixed tensor product reports `ERROR_ESTIMATE_UNAVAILABLE`;
adaptive tensor evidence is a global successive-level difference, while
adaptive cubature reports embedded-rule evidence. None is an exact true-error
certificate.

## Contract and evidence links

- [Method choice](../../20-methods/approximation-integration/multidimensional/choosing-a-method.md)
- [Tensor derivation](../../20-methods/approximation-integration/multidimensional/tensor-product.md)
- [Cubature derivation](../../20-methods/approximation-integration/multidimensional/adaptive-cubature.md)
- [Phase B validation](../../60-validation/numerical/quadrature-multidimensional.md)
- [Truth artifact](../../validation/quad-multidim-truth.json)
- [Replay artifact](../../validation/quad-multidim-replay.json)

## Canonical import example

```python
from jaxstro.quad import AdaptiveCubature, GenzMalik, Hyperrectangle, integrate

result = integrate(
    lambda x: x[:, 0] * x[:, 1],
    Hyperrectangle([0.0, 0.0], [1.0, 1.0]),
    method=AdaptiveCubature(GenzMalik()),
    epsabs=1e-10,
    epsrel=1e-10,
    max_evaluations=10_000,
    max_regions=128,
    gradient="replay",
)
```

