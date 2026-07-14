---
title: JAX configuration
---

# JAX configuration

## Owner import path

`jaxstro.jaxconfig`

## Purpose

Explicitly enable float64 arrays and highest matmul precision before numerical
work begins.

## Public records and callables

`enable_high_precision`.

## Shape and dtype expectations

The function has no array input. It changes process-wide JAX configuration for
subsequently created or compiled arrays.

## JAX transforms and AD classification

Configuration is a host-side setup action, not a traced or differentiated
kernel.

## Failure behavior

Calling after arrays or compiled functions exist may be too late for the
intended precision contract; Jaxstro does not configure precision at import.

## Contract and evidence links

See [](../../00-start-here/first-research-calculation.md) and the generated
[](./contracts.md) module contract.

## Canonical import example

```python
from jaxstro.jaxconfig import enable_high_precision
```
