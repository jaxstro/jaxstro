---
title: Sobol and randomized QMC API
description: Exact contracts for deterministic, fixed-look, and bounded sequential Sobol integration.
---

# Sobol and randomized QMC API

## Owner import path

`jaxstro.quad.qmc`, exposed through `jaxstro.quad`.

## Purpose

Use this family when structured grids are too expensive and a unit-cube Sobol
construction matches the problem. Deterministic `Sobol` provides no confidence
interval. Randomized methods use independent scrambled replicates to attach
declared uncertainty evidence.

## Public records and callables

```python
from jaxstro.quad import (
    AdaptiveScrambledSobol,
    DigitalShift,
    LinearMatrixScramble,
    OwenScramble,
    ScrambledSobol,
    Sobol,
    integrate,
)
```

```python
Sobol(level, bits=None)
ScrambledSobol(
    level,
    replicates=8,
    scramble=LinearMatrixScramble(),
    confidence_level=0.95,
)
AdaptiveScrambledSobol(
    schedule,
    estimate_bounds=None,
    integrand_bounds=None,
    scramble=LinearMatrixScramble(),
    confidence_level=0.95,
)
```

Randomized calls require an explicit JAX `key`. The sequential declaration
requires a predeclared monotone schedule and exactly one valid boundedness
contract.

## Shape and dtype expectations

The integrand receives `(n, dimension)` points and returns `(n,)` or
`(n, ...)`. Confidence intervals are restricted to real scalar outputs.
Dimension, level, replicate count, scramble type, schedule, payload shape, and
capacities are static under JIT.

## JAX transforms and AD classification

`jit` and `vmap` preserve explicit-key semantics. Reusing a key intentionally
reproduces the same randomized formula. `gradient="replay"` differentiates
that accepted formula, including the accepted sequential level and replicate
count, while stopping confidence construction and controller decisions.
Higher derivatives are unsupported.

## Failure behavior

Missing keys, malformed schedules, unsupported output shapes for confidence
intervals, and infeasible capacities raise eagerly. Dynamic invalid domains,
nonfinite integrands, and exhausted sequential schedules return typed statuses.
Fixed-look Student-t intervals and bounded sequential empirical-Bernstein
intervals have different meanings and must not be interchanged.

## Contract and evidence links

- [Randomized QMC derivation](../../20-methods/approximation-integration/multidimensional/multidimensional-randomized-qmc.md)
- [Differentiating accepted formulas](../../20-methods/approximation-integration/multidimensional/differentiating-multidimensional-integrals.md)
- [Phase B validation](../../60-validation/numerical/quadrature-multidimensional.md)
- [RQMC calibration artifact](../../validation/quad-rqmc-calibration.json)
- [Replay artifact](../../validation/quad-multidim-replay.json)

## Canonical import example

```python
import jax
from jaxstro.quad import Hyperrectangle, LinearMatrixScramble, ScrambledSobol, integrate

result = integrate(
    lambda x: 1.0 / (1.0 + x[:, 0] ** 2 + x[:, 1] ** 2),
    Hyperrectangle([0.0, 0.0], [1.0, 1.0]),
    method=ScrambledSobol(
        level=10,
        replicates=16,
        scramble=LinearMatrixScramble(),
        confidence_level=0.95,
    ),
    key=jax.random.key(4),
    epsabs=1e-4,
    epsrel=1e-4,
    max_evaluations=16_384,
    gradient="replay",
)
```
