---
title: Jaxstro quadrature
description: Canonical one-dimensional quadrature API and Phase B multidimensional structural contracts.
---

# Jaxstro quadrature

## Owner import path

`jaxstro.quad`

## Purpose

This is the canonical integration namespace. It provides sampled-data
integration, fixed rules, five adaptive method families, domains, measures,
typed error and work evidence, deterministic stopping statuses, and the
structural foundation for finite multidimensional integration.

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
    max_regions=None,
    max_indices=None,
    max_frontier=None,
    max_nodes=None,
    key=None,
    error_norm=quad.MaxNorm(),
    gradient="replay",
)
```

The integrand receives a node array with shape `(n,)` and returns `(n,)` or
`(n, ...)`. The result has the trailing payload shape. Rule and measure types,
rule order or level, breakpoint count, and payload shape are static under JIT.
Bounds, breakpoint values, and explicit `args` leaves may be dynamic.

Improper domains accept a keyword-only characteristic scale:

```python
quad.RightInfinite(lower, scale=scale)
quad.LeftInfinite(upper, scale=scale)
quad.Infinite(unit=unit, scale=scale)
```

Raw calls may omit `scale` and retain the legacy numerical value `1`. A
dimensional quantity-mode improper domain must provide a compatible, scalar,
real, positive, finite quantity scale. Scale is stopped algorithmic provenance:
it controls the map and conditioning, but is not a differentiable scientific
parameter.

For replay differentiation, smooth finite bounds, the finite boundary of a
supported semi-infinite domain, and explicit floating or complex `args` leaves
are differentiable. Breakpoint motion, method and measure configuration,
`epsabs`, `epsrel`, capacities, error norms, payload shape, refinement
decisions, statuses, error estimates, and work records are static or stopped.
A parameter to be differentiated must be passed through `args` or a supported
bound; hiding it in the integrand closure is unsupported.

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

:::{important} Classical `Romberg` stopping contract
Classical `Romberg` has a fixed level-$4$ alias-protection floor. It cannot
report `CONVERGED` before $2^4+1=17$ logical evaluations, regardless of an
earlier Richardson estimate. Set `max_evaluations` to at least $17$ when
convergence is required; smaller valid capacities can terminate with
`MAX_EVALUATIONS`. This floor does not apply to `RombergTanhSinh`.
:::

## Capability status map

The status belongs to a capability, not to the package as a whole.

| Status | Current quadrature scope |
| --- | --- |
| shipped and validated | Sampled-data integration, fixed one-dimensional rules, five adaptive one-dimensional families, typed failure and work evidence, first-order accepted-formula replay, and the Phase B finite-hyperrectangle structural substrate |
| benchmarking | The Apple M2 Max CPU comparison is accepted; additional backends, precisions, batch regimes, and method families remain future benchmarking coverage |
| alpha | Opt-in quantity normalization through `quad.integrate`; downstream ecosystem adoption is not implied |
| approved but planned | Numerical integration over multidimensional hyperrectangles, adaptive cubature, sparse grids, randomized QMC, and later scientific geometries; no multidimensional numerical method is available yet |
| intentionally unsupported | Posterior inference, experimental-design policy, general Monte Carlo inference, and domain-specific scientific acceptance |

### Reading comparison labels

Performance ratios are interpreted only after both results pass their declared
truth and derivative gates. The first lane asks a family-matched research
question. **Exact** means the same embedded rule family and order under matched
domain, tolerance, norm, and capacity controls where those controls apply.
**Strong-match** means closely matched global-refinement capacity.
**Node-matched** means the same local node count with potentially different
estimators. **Family-matched** means the same broad method family while
acknowledging algorithmic differences. A **capability comparison** asks whether
related public capabilities solve the same research task without implying
algorithmic equivalence.

The second lane asks a practical choice question. Its `best_method` label uses
predeclared library-specific settings intended to represent a reasonable
public method in each library. It is not an algorithm-equivalence label and is
not mixed into family-matched superiority claims.

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
- `Hyperrectangle`
- `interval_orientation`
- `sorted_breakpoints`
- `interval_is_valid`
- `hyperrectangle_orientation`
- `hyperrectangle_is_valid`
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

## Multidimensional domain contract

`Hyperrectangle(lower, upper)` represents the finite Cartesian product

$$
\prod_{j=1}^{d} [a_j,b_j].
$$

Both bounds are one-dimensional arrays with the same positive, static length
$d$. Their values are dynamic PyTree leaves. Concrete nonfinite bounds raise
eagerly; traced nonfinite bounds make `hyperrectangle_is_valid(domain)` return
false. Reversed and zero-width axes are represented explicitly rather than
reordered. The signed orientation is the product of the per-axis signs, while
the map Jacobian is nonnegative.

```python
import jax.numpy as jnp

from jaxstro import quad

domain = quad.Hyperrectangle(
    jnp.array([0.0, 0.0]),
    jnp.array([1.0, 2.0]),
)
# x passed to a Phase B integrand has shape (point_count, dimension).
```

This domain is $[0,1]\times[0,2]$. Phase B multidimensional integrands use a
coordinate-last point array with shape `(point_count, dimension)` and return an
array whose leading axis is `point_count`.

## Phase B dispatcher boundary

`quad.integrate` is the sole public family dispatcher. One-dimensional domains
continue to delegate to the existing adaptive owner with complete
`QuadResult` PyTree parity. The names `max_indices`, `max_frontier`,
`max_nodes`, and `key` reserve explicit capacity and random-state boundaries
for later multidimensional families; one-dimensional calls reject them.

:::{warning}
`Hyperrectangle` and the thin dispatcher are structural B0 contracts. A
multidimensional numerical method is not available until its family passes the
B1, B2, or B3 validation gate.
:::

## Quantity activation

Quantity handling belongs only to `quad.integrate` and is alpha. The adapter
validates and unwraps units before calling the same raw engine, then restores
the integral unit on `value`, `error.estimate`, `error.norm`, and `tolerance`.
Status, work, error kind, and confidence level remain unitless.

| Input condition | Mode and requirement |
| --- | --- |
| Any quantity-valued bound or breakpoint | Quantity mode; all dimensional coordinates must be compatible quantities |
| `Infinite(unit=unit, scale=scale)` | Quantity mode with the declared coordinate unit and required compatible physical scale |
| Dimensional `RightInfinite` or `LeftInfinite` | Quantity mode; requires a compatible physical `scale` |
| Quantity `epsabs` with a raw domain | Quantity mode with dimensionless coordinates |
| Quantity integrand output without a quantity trigger | Eager error explaining that quantity `epsabs` activates a dimensionless quantity domain |
| Quantity mode integrand | Must return a `Quantity` with one stable output unit |
| Quantity mode `epsabs` | Required and compatible with the complete integral unit |
| Quantity `epsrel` | Must be dimensionless |
| Quantity `WeightedMeasure` density | Receives quantity coordinates and must match `density_unit` |

`quad.fixed`, `map_domain`, and `map_interval` reject quantity-valued domains,
including `Infinite(unit=...)`. The raw `Infinite()` form is unchanged.

For improper maps, `scale` must be scalar, real, finite, and strictly positive.
Invalid traced scales fail closed with `QuadStatus.INVALID_INPUT`. Equivalent physical scales
expressed in different compatible units produce the same normalized map.
Different physical scales may change convergence, error estimates, and work,
so scale selection belongs in the numerical-method record.

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
records the generated tolerance sweeps. The
[quadrature performance and comparison](../../60-validation/numerical/quadrature-performance.md)
page explains the matched comparison labels, recorded hardware, accepted
Romberg optimization, and warranted non-claims.

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
    gradient="replay",
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

## Migrating to `jaxstro.quad`

The canonical owner is available now, but compatibility paths remain while
sibling packages are audited one repository at a time. Switching imports does
not change behavior: each row resolves to the same implementation or preserves
the explicitly documented legacy alias.

| Compatibility import | Canonical owner |
| --- | --- |
| `jaxstro.numerics.integration.trapezoid` | `jaxstro.quad.trapezoid` |
| `jaxstro.numerics.integration.trapz` | `jaxstro.quad.trapezoid` |
| `jaxstro.numerics.integration.cumulative_trapezoid` | `jaxstro.quad.cumulative_trapezoid` |
| `jaxstro.numerics.integration.cumulative_trapz` | `jaxstro.quad.cumulative_trapezoid` |
| `jaxstro.numerics.integration.simpson` | `jaxstro.quad.simpson` |
| `jaxstro.numerics.integration.cumulative_simpson` | `jaxstro.quad.cumulative_simpson` |
| `jaxstro.numerics.quadrature.gauss_legendre_nodes` | `jaxstro.quad.gauss_legendre_nodes` |
| `jaxstro.numerics.quadrature.gauss_laguerre_nodes` | `jaxstro.quad.gauss_laguerre_nodes` |
| `jaxstro.numerics.quadrature.gauss_hermite_nodes` | `jaxstro.quad.gauss_hermite_nodes` |
| `jaxstro.numerics.quadrature.clenshaw_curtis_nodes` | `jaxstro.quad.clenshaw_curtis_nodes` |
| `jaxstro.numerics.quadrature.hermite_e_basis` | `jaxstro.quad.hermite_e_basis` |
| `jaxstro.numerics.quadrature.hermite_coefficients` | `jaxstro.quad.hermite_coefficients` |

No compatibility path will be removed until downstream audits and migrations
are complete. The current aliases emit no deprecation warning, and this Phase
A closeout does not modify any sibling package.
