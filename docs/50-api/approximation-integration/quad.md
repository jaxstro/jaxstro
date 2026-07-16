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

For replay differentiation, smooth finite bounds and explicit floating or
complex `args` leaves are differentiable. Breakpoint motion, method and measure
configuration, capacities, error norms, payload shape, refinement decisions,
statuses, error estimates, and work records are static or stopped. A parameter
to be differentiated must be passed through `args` or a supported bound; hiding
it in the integrand closure is unsupported.

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
scalar, complex, vector, or higher-rank trailing payloads. In raw mode,
`epsabs` and `epsrel` are scalar real values. Method type and configuration,
`max_evaluations`, `max_regions`, breakpoint count, and payload shape remain
static under JIT.

### Quantity activation

Quantity handling belongs only to `quad.integrate` and is alpha. The adapter
validates and unwraps units before calling the same raw engine, then restores
the integral unit on `value`, `error.estimate`, `error.norm`, and `tolerance`.
Status, work, error kind, and confidence level remain unitless.

| Input condition | Mode and requirement |
| --- | --- |
| Any quantity-valued bound or breakpoint | Quantity mode; all dimensional coordinates must be compatible quantities |
| `Infinite(unit=unit)` | Quantity mode with the declared coordinate unit |
| Quantity `epsabs` with a raw domain | Quantity mode with dimensionless coordinates |
| Quantity integrand output without a quantity trigger | Eager error explaining that quantity `epsabs` activates a dimensionless quantity domain |
| Quantity mode integrand | Must return a `Quantity` with one stable output unit |
| Quantity mode `epsabs` | Required and compatible with the complete integral unit |
| Quantity `epsrel` | Must be dimensionless |
| Quantity `WeightedMeasure` density | Receives quantity coordinates and must match `density_unit` |

`quad.fixed`, `map_domain`, and `map_interval` reject quantity-valued domains,
including `Infinite(unit=...)`. The raw `Infinite()` form is unchanged.

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
result. The current status precedence for completed estimates resolves invalid
input, then nonfinite values, then
convergence, then explicit representability or stagnation evidence. Before a
new regional split, midpoint collapse precedes exhausted evaluation capacity,
which precedes exhausted region capacity. A roundoff-scale error floor alone
does not emit `ROUNDOFF_LIMITED`. Regional controllers distinguish
`MAX_EVALUATIONS` and `MAX_REGIONS`.
`DIVERGENCE_SUSPECTED` and `ERROR_ESTIMATE_UNAVAILABLE` are reserved statuses,
not current controller outputs. Sparse-grid and replicate error kinds are likewise
reserved.

## JAX transforms and AD classification

`fixed` supports `jax.jit` and `jax.vmap` with the static boundaries above. It
has smooth pathwise AD semantics for the executed fixed formula. Gradients may
flow through explicit integrand parameters and smooth finite bounds. The rule
configuration, node count, and discrete breakpoint partition are not
differentiated.

`integrate` supports `jax.jit` and `jax.vmap` under its static capacity and
configuration boundaries. `gradient="replay"` returns the exact primal result
tree while differentiating the accepted fixed formula. Only `value` receives
that derivative; diagnostic tangents are exact zero or JAX `float0`.
`gradient="stop"` passes the complete result tree through
`jax.lax.stop_gradient`. VMAP runs one bounded adaptive controller per batch
member.

Replay supports JVP, selected VJP projections, value-only `jacfwd` and
`jacrev`, JIT, VMAP, real-to-complex realified Jacobians, complex-to-real JAX
cotangents, and realified complex-to-complex Jacobians. Do not apply `jacrev`
to the integer-bearing complete `QuadResult`.

For `INVALID_INPUT` and `NONFINITE_INTEGRAND`, the primal value is nonfinite
and derivatives are undefined. No tangent-layout promise is made for those
statuses.

Reference validation uses float64. Normal calls follow the active JAX precision
policy. A quantity result JVP retains the static integral unit. To obtain a
physical Jacobian unit, differentiate selected numerical values and declare
the input and output units. Direct differentiation of a `Quantity` PyTree does
not infer quotient-unit algebra.

## Contract and evidence links

Review [fixed and weighted quadrature](../../20-methods/approximation-integration/quadrature.md)
and [adaptive quadrature](../../20-methods/approximation-integration/adaptive-quadrature.md)
for primal derivations. [Differentiating an integral](../../20-methods/approximation-integration/differentiating-an-integral.md)
derives replay, moving-bound, complex, and unit contracts. The
[validation index](../../60-validation/validation.md) names the executable
envelopes,
[`quad-replay-derivatives.json`](../../validation/quad-replay-derivatives.json)
records replay evidence, and
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

`QuadWork` uses these exact current meanings:

| Field | Regional methods | Global Romberg families |
| --- | --- | --- |
| `evaluations` | `n * (M + 2 * r)` logical integrand evaluations | `2**k + 1` for classical Romberg; finest active-node count for Romberg-tanh-sinh |
| `refinements` | completed region bisections `r` | zero-based finest completed level `k` |
| `active_regions` | current active partition size | `1` |
| `levels` | `0` | number of completed levels, `k + 1` |
| `replicates` | `0` | `0` |

Logical evaluations are not padded device lanes, compile time, or wall time.
An exact zero-width finite interval returns an all-zero `QuadWork` record.

The reported estimator is not an exact error certificate. Related rules can
miss the same unresolved narrow feature even when the returned status is
`CONVERGED`.

### Compatibility boundary

`jaxstro.numerics.integration` and `jaxstro.numerics.quadrature` are temporary compatibility
paths. Their existing public names remain exact aliases and emit
no deprecation warning. The legacy probabilists' Hermite helper retains its
byte-compatible NumPy construction until a declared breaking release.
