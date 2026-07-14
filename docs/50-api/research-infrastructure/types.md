---
title: Numerical type aliases
---

# Numerical type aliases

## Owner import path

`jaxstro.numerics.types`

## Purpose

Shared annotation aliases for array-valued numerical functions.

## Public records and callables

`Array` aliases `jax.numpy.ndarray`. `ScalarFn` describes a callable that
accepts one `Array` and returns one `Array`.

## Shape and dtype expectations

These aliases do not constrain array shape or dtype at runtime. A function
annotated as `ScalarFn` still owns its concrete scalar, shape, and dtype
contract.

## JAX transforms and AD classification

The aliases add no runtime behavior and make no transformation or AD claim.
The annotated function's implementation determines JIT, VMAP, and derivative
behavior.

## Failure behavior

Importing or using an alias performs no validation. Shape, dtype, and callable
contract failures remain the responsibility of the consuming API.

## Contract and evidence links

See the [API import policy](../api.md) and [](./checks.md) for concrete runtime
validation helpers.

## Canonical import example

```python
from jaxstro.numerics.types import Array, ScalarFn
```
