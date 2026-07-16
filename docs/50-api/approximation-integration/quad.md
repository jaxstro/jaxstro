---
title: Jaxstro quadrature
description: Canonical sampled-data, fixed, and adaptive one-dimensional quadrature API.
---

# Jaxstro quadrature

## Owner import path

`jaxstro.quad`

## Purpose

This is the canonical integration namespace. It provides sampled-data
integration, fixed rules, five adaptive method families, domains, measures,
typed error and work evidence, and deterministic stopping statuses.

```python
from jaxstro import quad
from jaxstro.quad import fixed, integrate
```

## Public records and callables

```python
quad.fixed(fun, domain, *, args=(), rule, measure=None)

quad.integrate(
    fun,
    domain,
    *,
    args=(),
    method,
    measure=None,
    epsabs,
    epsrel,
    max_evaluations,
    max_regions,
    error_norm=quad.MaxNorm(),
    gradient="stop",
)
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

Supported adaptive declarations are `GaussKronrod`,
`AdaptiveClenshawCurtis`, `AdaptiveTanhSinh`, `Romberg`, and
`RombergTanhSinh`. `integrate` returns a `QuadResult` containing the primal
value, `QuadError`, effective tolerance, `QuadStatus`, and `QuadWork`.

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

Adaptive evaluation and methods:

- `integrate`
- `GaussKronrod`
- `AdaptiveClenshawCurtis`
- `AdaptiveTanhSinh`
- `Romberg`
- `RombergTanhSinh`

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

Results, statuses, work, and tolerances:

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

Adaptive integrands follow the same leading-node convention and may return
scalar, complex, vector, or higher-rank trailing payloads. `epsabs` and
`epsrel` are scalar real values. Method type and configuration,
`max_evaluations`, `max_regions`, breakpoint count, and payload shape remain
static under JIT.

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

### Supported adaptive pairings

| Method | Domain | Breakpoints | Measure | Current error kind |
| --- | --- | --- | --- | --- |
| `GaussKronrod` | finite `Interval` | yes | `LebesgueMeasure`, `WeightedMeasure` | `EMBEDDED_RULE` |
| `AdaptiveClenshawCurtis` | finite `Interval` | yes | `LebesgueMeasure`, `WeightedMeasure` | `REFINEMENT_DIFFERENCE` |
| `AdaptiveTanhSinh` | any current domain | finite intervals only | `LebesgueMeasure`, `WeightedMeasure` | `REFINEMENT_DIFFERENCE` |
| `Romberg` | finite `Interval` | no | `LebesgueMeasure`, `WeightedMeasure` | `REFINEMENT_DIFFERENCE` |
| `RombergTanhSinh` | any current domain | no | `LebesgueMeasure`, `WeightedMeasure` | `REFINEMENT_DIFFERENCE` |

Structural incompatibilities raise before tracing. Dynamic invalid inputs,
nonfinite integrands, roundoff limits, and capacity exhaustion return a typed
result. Current status precedence is invalid input, nonfinite integrand,
convergence, roundoff limitation, and then exhausted capacity. Regional
controllers distinguish `MAX_EVALUATIONS` and `MAX_REGIONS`.
`DIVERGENCE_SUSPECTED` and `ERROR_ESTIMATE_UNAVAILABLE` are reserved statuses,
not current A2 outputs. Sparse-grid and replicate error kinds are likewise
reserved.

## JAX transforms and AD classification

`fixed` supports `jax.jit` and `jax.vmap` with the static boundaries above. It
has smooth pathwise AD semantics for the executed fixed formula. Gradients may
flow through explicit integrand parameters and smooth finite bounds. The rule
configuration, node count, and discrete breakpoint partition are not
differentiated.

`integrate` supports `jax.jit` and `jax.vmap` under its static capacity and
configuration boundaries. Its only current policy is `gradient="stop"`: the
complete primal result tree is passed through `jax.lax.stop_gradient`.
Consequently a zero derivative means AD was deliberately stopped, not that the
mathematical integral has zero derivative. VMAP runs one bounded adaptive
controller per batch member.

Reference validation uses float64. Normal calls follow the active JAX precision
policy. The current API accepts raw arrays only; quantity-valued inputs and
adaptive replay derivatives remain later work.

## Contract and evidence links

Review [fixed and weighted quadrature](../../20-methods/approximation-integration/quadrature.md)
and [adaptive quadrature](../../20-methods/approximation-integration/adaptive-quadrature.md)
for derivations and audit procedures. The
[validation index](../../60-validation/validation.md) names the executable
envelope, and
[`quad-adaptive-envelope.json`](../../validation/quad-adaptive-envelope.json)
records the generated tolerance sweeps.

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

An adaptive call has the same canonical owner:

```python
adaptive = quad.integrate(
    lambda x: x**2,
    quad.Interval(0.0, 1.0),
    method=quad.GaussKronrod(pair=21),
    epsabs=1e-8,
    epsrel=1e-8,
    max_evaluations=2048,
    max_regions=64,
    gradient="stop",
)
```

`QuadWork.evaluations` means logical integrand evaluations, not padded device
lanes or wall time. For an `n`-node regional rule, `M` initial regions, and `r`
splits, the count is `n * (M + 2 * r)`. Classical Romberg reports `2**k + 1`
at completed level `k`; Romberg-tanh-sinh reports the active-node count at its
finest completed level.

The reported estimator is not an exact error certificate. Related rules can
miss the same unresolved narrow feature even when the returned status is
`CONVERGED`.

### Compatibility boundary

`jaxstro.numerics.integration` and `jaxstro.numerics.quadrature` are temporary compatibility
paths. Their existing public names remain exact aliases and emit
no deprecation warning. The legacy probabilists' Hermite helper retains its
byte-compatible NumPy construction until a declared breaking release.
