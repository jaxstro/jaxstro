---
title: Parameter bridges
---

# Parameter bridges

## Owner import path

`jaxstro.params`

## Purpose

Selective Equinox PyTree-to-vector parameter bridges and bijectors for bounded
physical leaves.

## Public records and callables

`Parameterization`, `AbstractBijector`, `Identity`, `Exp`, `Softplus`, and
`Sigmoid`.

## Shape and dtype expectations

Free leaves are JAX arrays flattened into one floating vector. Static and fixed
leaves remain in the model PyTree.

## JAX transforms and AD classification

Vector conversion and bijectors compose with JIT and AD for fixed leaf
selection. Selection metadata is static.

## Failure behavior

Selection and vector-shape mismatches raise. `from_vector` replaces leaves and
does not rerun model initialization, so cached derived leaves may remain stale.

## Contract and evidence links

See [](../../30-representations/parameters-state/parameters-and-transforms.md)
and [](../../70-project/decisions/0009-jaxstro-params-selective-inference.md).

## Canonical import example

```python
from jaxstro.params import Parameterization
```
