---
title: Jaxstro quadrature foundation
description: Canonical sampled and fixed-rule facade plus Phase A0 integration contracts.
---

# Jaxstro quadrature foundation

## Owner import path

`jaxstro.quad`

## Purpose

This is the canonical integration namespace. In Phase A0 it exposes the
currently supported sampled and fixed-node helpers together with typed domain,
measure, tolerance, and result foundations. It does not yet provide adaptive integration.

## Public records and callables

Sampled values:

- `trapezoid`
- `cumulative_trapezoid`
- `simpson`
- `cumulative_simpson`

Fixed-rule helpers:

- `gauss_legendre_nodes`
- `gauss_laguerre_nodes`
- `gauss_hermite_nodes`
- `clenshaw_curtis_nodes`
- `hermite_e_basis`
- `hermite_coefficients`

Domains and measures:

- `Interval`, `RightInfinite`, `LeftInfinite`, and `Infinite`
- `LebesgueMeasure`, `WeightedMeasure`, `JacobiMeasure`,
  `LaguerreMeasure`, `PhysicistsHermiteMeasure`, and `StandardNormalMeasure`

Results and tolerances:

- `QuadStatus`, `ErrorKind`, `QuadError`, `QuadWork`, and `QuadResult`
- `MaxNorm`, `L1Norm`, `L2Norm`, `error_norm`, and `tolerance_threshold`

## Shape and dtype expectations

Sampled functions reduce or cumulatively retain one selected array axis under
their existing contracts. Node factories return two arrays with shape `(n,)`.
`hermite_e_basis(g, n_max)` returns shape `(n_max + 1, g.shape[0])`, and
`hermite_coefficients` returns shape `(n_max + 1,)`. Domain endpoints are
scalar numerical PyTree leaves; breakpoint count is static.

## JAX transforms and AD classification

The sampled functions preserve their current JIT and differentiation behavior.
Fixed nodes and weights are generated as host-side constants; downstream JAX
calculations differentiate through integrand values, not node construction.
Result records and domains are PyTrees. Method-level replay AD does not exist
until Phase A3.

## Failure behavior

Existing sampled-grid shape, uniformity, parity, and rule-order failures remain
unchanged. Phase A0 defines adaptive status codes but no controller emits them
yet. Infinite-domain declarations are configuration only until a later method
supplies and validates the corresponding transformation.

## Contract and evidence links

Review [integration from samples](../../20-methods/approximation-integration/cumulative-trapz.md),
[fixed-node quadrature](../../20-methods/approximation-integration/quadrature.md),
and the [validation index](../../60-validation/validation.md).

## Canonical import example

```python
from jaxstro import quad
from jaxstro.quad import Interval

nodes, weights = quad.gauss_legendre_nodes(8)
domain = Interval(-1.0, 1.0)
```

The old `jaxstro.numerics.integration` and
`jaxstro.numerics.quadrature` paths are temporary compatibility surfaces. A0
preserves exact callable identity and does not issue deprecation warnings.
