---
title: Jaxstro quadrature
description: Canonical sampled-data and one-dimensional fixed-quadrature API.
---

# Jaxstro quadrature

## Owner import path

`jaxstro.quad`

## Purpose

This is the canonical integration namespace. Phase A1 provides sampled-data
integration, a common fixed evaluator, classical Gaussian rules,
Clenshaw-Curtis, Fejer type I and II, fixed tanh-sinh, domains, measures, and
typed result foundations. It does not yet provide adaptive integration.

```python
from jaxstro import quad
from jaxstro.quad import fixed
```

## Public records and callables

```python
quad.fixed(fun, domain, *, args=(), rule, measure=None)
```

The integrand receives a node array with shape `(n,)` and returns `(n,)` or
`(n, ...)`. The result has the trailing payload shape. Rule and measure types,
rule order or level, breakpoint count, and payload shape are static under JIT.
Bounds, breakpoint values, and explicit `args` leaves may be dynamic.

Supported rule declarations:

- `GaussianRule`
- `ClenshawCurtisRule`
- `FejerIRule`
- `FejerIIRule`
- `TanhSinhRule`

`GaussianRule` dispatches to Gauss-Legendre, Gauss-Jacobi, generalized
Gauss-Laguerre, physicists' Gauss-Hermite, or standard-normal Gauss-Hermite from
the declared domain and measure.

### Complete public inventory

Sampled values:

- `trapezoid`
- `cumulative_trapezoid`
- `simpson`
- `cumulative_simpson`

Fixed evaluation and rules:

- `fixed`
- `GaussianRule`
- `ClenshawCurtisRule`
- `FejerIRule`
- `FejerIIRule`
- `TanhSinhRule`

Compatibility and expansion helpers:

- `gauss_legendre_nodes`
- `gauss_laguerre_nodes`
- `gauss_hermite_nodes`
- `clenshaw_curtis_nodes`
- `hermite_e_basis`
- `hermite_coefficients`

Domains and transformations:

- `Interval`
- `RightInfinite`
- `LeftInfinite`
- `Infinite`
- `interval_orientation`
- `sorted_breakpoints`
- `interval_is_valid`
- `AffineMapResult`
- `DomainMapResult`
- `map_interval`
- `map_domain`

Measures:

- `LebesgueMeasure`
- `WeightedMeasure`
- `JacobiMeasure`
- `LaguerreMeasure`
- `PhysicistsHermiteMeasure`
- `StandardNormalMeasure`

Results and tolerances reserved for the adaptive layer:

- `QuadStatus`
- `ErrorKind`
- `QuadError`
- `QuadWork`
- `QuadResult`
- `ErrorNorm`
- `MaxNorm`
- `L1Norm`
- `L2Norm`
- `error_norm`
- `tolerance_threshold`

## Shape and dtype expectations

The node input has shape `(n,)`; the integrand returns `(n,)` or `(n, ...)`;
and `fixed` removes the leading node axis. Sampled methods reduce or retain the
selected array axis according to their individual contracts. Bounds and
breakpoint values follow the active JAX dtype policy. Reference validation uses
float64.

## Failure behavior

### Supported fixed pairings

| Rule | Domain | Measure |
| --- | --- | --- |
| `GaussianRule` | `Interval` | `LebesgueMeasure`, or `JacobiMeasure` without breakpoints |
| `GaussianRule` | `RightInfinite` | `LaguerreMeasure` |
| `GaussianRule` | `Infinite` | `PhysicistsHermiteMeasure` or `StandardNormalMeasure` |
| `ClenshawCurtisRule`, `FejerIRule`, `FejerIIRule` | `Interval` | `LebesgueMeasure` or `WeightedMeasure` |
| `TanhSinhRule` | Any Phase A domain | `LebesgueMeasure` or `WeightedMeasure` |

Unsupported structural pairings raise eagerly. Value-dependent invalid finite
domains return `nan` when traced. Zero-width finite intervals return an exact
zero after static payload-shape inference.

`JacobiMeasure(alpha, beta)` means
$(1-t)^{\alpha}(1+t)^{\beta}\,\mathrm{d}t$ on the reference interval. An
arbitrary finite interval uses the documented affine reference-density
convention; it does not reinterpret `alpha` and `beta` as an unstated physical
density. `LaguerreMeasure(alpha)` on `RightInfinite(lower)` uses
$u=x-\mathtt{lower}$ and the density $u^{\alpha}e^{-u}\,\mathrm{d}u$. See the
[method derivation](../../20-methods/approximation-integration/quadrature.md#classical-measure-conventions)
for the masses and normalization equations.

## JAX transforms and AD classification

`fixed` supports `jax.jit` and `jax.vmap` with the static boundaries above. It
has smooth pathwise AD semantics for the executed fixed formula. Gradients may
flow through explicit integrand parameters and smooth finite bounds. The rule
configuration, node count, and discrete breakpoint partition are not
differentiated.

Reference validation uses float64. Normal calls follow the active JAX precision
policy. Phase A1 accepts raw arrays only; quantity-valued inputs remain Phase A3
work.

## Contract and evidence links

Review [fixed and weighted quadrature](../../20-methods/approximation-integration/quadrature.md)
for derivations and audit procedures, and the
[validation index](../../60-validation/validation.md) for evidence boundaries.

## Canonical import example

```python
import jax.numpy as jnp

from jaxstro import quad

value = quad.fixed(
    lambda x: jnp.exp(-(x**2)),
    quad.Infinite(),
    rule=quad.TanhSinhRule(6),
)
```

### Compatibility boundary

`jaxstro.numerics.integration` and `jaxstro.numerics.quadrature` are temporary compatibility
paths. Their existing public names remain exact aliases and emit
no deprecation warning. The legacy probabilists' Hermite helper retains its
byte-compatible NumPy construction until a declared breaking release.
