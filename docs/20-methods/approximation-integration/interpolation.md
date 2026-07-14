---
title: Cubic interpolation
description: >-
  Cubic Hermite, natural cubic spline, and PCHIP interpolation with explicit
  smoothness, shape, clamping, and gradient contracts.
---

Linear interpolation is hard to beat for honesty: it does not invent curvature.
But many table-like models need a smoother derivative than straight-line
segments provide. Cubic interpolation gives that smoothness, but an unconstrained
cubic can overshoot between samples and create values the table never implied.

jaxstro's shape-preserving interpolation layer adds a small middle ground:

```python
from jaxstro.jaxconfig import enable_high_precision

enable_high_precision()  # before creating JAX arrays

import jax.numpy as jnp

from jaxstro.numerics import interpolation

x_grid = jnp.arange(5.0)
values = jnp.array([0.0, 0.01, 0.9, 0.91, 1.0])
x_new = jnp.linspace(0.0, 4.0, 801)
dydx = interpolation.pchip_slopes(x_grid, values)

hermite = interpolation.cubic_hermite_interp(
    x_grid, values, dydx, x_new
)
natural = interpolation.eval_cubic_spline(
    x_grid,
    interpolation.natural_cubic_spline_coeffs(x_grid, values),
    x_new,
)
monotone = interpolation.monotone_cubic_interp(x_grid, values, x_new)
table = interpolation.MonotoneTabulatedFunction1D(x_grid, values)
wrapped_monotone = table(x_new)

assert natural.min() < -0.1  # smooth, but below the monotone sample range
assert monotone.min() >= -1e-12
assert monotone.max() <= 1.0 + 1e-12
assert jnp.all(jnp.diff(monotone) >= -1e-12)
assert jnp.allclose(hermite, monotone)
assert jnp.allclose(wrapped_monotone, monotone)
```

:::{figure} ../../10-theory/figures/interpolation-shape-contracts.webp
:name: fig-interpolation-shape-contracts
:alt: Two-panel comparison of natural cubic and PCHIP interpolation for the same monotone samples, showing natural-spline undershoot and nonnegative PCHIP increments

Both panels are computed from the public APIs and the same five monotone samples.
The natural cubic is globally smoother but dips below the sampled range and has
negative successive increments. PCHIP stays within `[0, 1]` and remains monotone
on the displayed query grid. This fixture demonstrates a contract distinction,
not universal superiority or an error benchmark.
:::

`cubic_hermite_interp(...)` is the explicit primitive: callers provide both
values and derivatives at the knots. `natural_cubic_spline_coeffs(...)` computes
the per-interval coefficients for a twice-continuously differentiable cubic
spline with natural boundary conditions. `monotone_cubic_interp(...)` computes
PCHIP-style limited slopes from the values and then uses the same Hermite
evaluator when avoiding overshoot is more important than global smoothness.

## Learning objectives

After this chapter, you should be able to choose an interpolation contract from
the scientific shape information, predict boundary behavior, and audit local
derivatives without treating smoothness as evidence of physical correctness.

### Concept check: smooth or shape-preserving?

Before plotting, predict whether a natural cubic can leave the range of monotone
samples and whether PCHIP can. Compute both, then audit successive increments
and range bounds rather than judging the curves by appearance.

## Boundary Policy

The default policy matches `interp1d`: query points outside the grid clamp to
the endpoint values. No extrapolated curve is invented. This means gradients
with respect to `x_new` saturate outside the data domain, just as they do for
linear interpolation and clamped B-splines.

Callers can request `extrapolate=True` when they explicitly want the endpoint
Hermite segment continued beyond the table: numerical continuation, not a physical guarantee.

Natural cubic spline evaluation clamps query coordinates to the data domain
before choosing an interval, so out-of-range queries return endpoint values.

The Python wrappers validate a concrete grid eagerly. Value-dependent exceptions
cannot fire on a tracer, so eager validation is skipped while the grid is traced;
the caller must supply a strictly increasing grid under `jax.jit`.

## Natural Cubic Spline

The natural cubic spline uses the standard second-derivative moment system in
{cite:t}`deBoor2001`. With

```{math}
h_i = x_{i+1} - x_i,\qquad
s_i = \frac{y_{i+1} - y_i}{h_i},
```

the natural boundary conditions are

```{math}
m_0 = m_n = 0,
```

and the interior moments solve

```{math}
h_{i-1}m_{i-1} + 2(h_{i-1}+h_i)m_i + h_i m_{i+1}
= 6(s_i - s_{i-1}).
```

`natural_cubic_spline_coeffs(x, y)` returns per-interval coefficients
`(a, b, c, d)` for

```{math}
S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)^2 + d_i(x-x_i)^3.
```

The solve is written with JAX arrays and `jnp.linalg.solve`, so gradients flow
through the coefficients and table values. This helper is intended for smooth
tables such as fixed extinction-curve anchors where global `C^2` smoothness is
the desired contract.

## PCHIP Slopes

{cite:t}`FritschButland1984` constructs local piecewise-cubic interpolants for
monotone data by choosing node derivatives from neighboring secant slopes. The
implemented weighted harmonic mean is the paper's spacing-aware construction.

For intervals with width

```{math}
h_i = x_{i+1} - x_i
```

and secant slope

```{math}
d_i = \frac{y_{i+1} - y_i}{h_i},
```

the interior PCHIP slope is zero when adjacent secants change sign. Otherwise it
uses the weighted harmonic mean:

```{math}
m_i =
\frac{w_1 + w_2}{w_1 / d_{i-1} + w_2 / d_i},
\quad
w_1 = 2h_i + h_{i-1},
\quad
w_2 = h_i + 2h_{i-1}.
```

This is the important limiter: if the table is monotone, the interpolant does
not introduce a new interior extremum. If the table contains a plateau or a
turning point, the local derivative is set to zero at the relevant node.

## Differentiability

Interpolation does not have one universal derivative contract:

```{list-table} Interpolation gradient contracts
:header-rows: 1
:label: tbl-interpolation-gradient-contracts

* - Operation
  - Contract
  - Supported claim
  - Boundary
* - Hermite values and supplied derivatives
  - `smooth_pathwise`
  - AD agrees with finite differences for values, supplied derivatives, and an
    interior query in a fixed segment.
  - The query is away from knots and clamping.
* - Natural-spline values and interior query
  - `smooth_pathwise`
  - The coefficient solve and evaluation carry gradients through table values
    and smooth interior queries.
  - The grid is strictly increasing and the selected interval is fixed locally.
* - PCHIP inside a fixed limiter branch
  - `smooth_pathwise`
  - AD is meaningful while adjacent secant signs and endpoint-limiter decisions
    remain unchanged.
  - Crossing a limiter decision changes the local derivative rule.
* - Clamped exterior query
  - `known_zero`
  - The default public evaluators are constant beyond the endpoint values.
  - This zero is saturation, not an inference direction.
* - Knots and limiter transitions
  - `validation_only`
  - Values, bounds, and named one-sided behavior can be checked directly.
  - No universal smooth derivative is claimed at a knot or branch transition.
```

Natural cubic splines are differentiable with respect to table values and query
positions away from knots and clamp boundaries. Inside a fixed PCHIP limiter
branch, the monotone interpolant is differentiable with respect to sample values
and query positions. The validation suite checks these claims with finite
differences at smooth interior points.

At branch boundaries, such as a slope changing sign or an endpoint limiter
clipping a derivative to zero, the PCHIP construction is only piecewise smooth.
That is the correct mathematical behavior. Tests cover the branch behavior as a
shape invariant, and the FD-vs-AD audit uses interior smooth cases for gradient
claims.

## What This Does Not Do

This is still a 1D table primitive. It does not handle regular grids,
unstructured scattered data, or multidimensional monotonicity constraints. Those
belong to separate chunks because they need explicit axis, boundary, and
validation policies.

## From explanation to evidence

Use the [](../../50-api/approximation-integration/interpolation.md) for signatures and
ownership, the [](../../60-validation/index.md) for measured interpolation anchors,
and the [](../methods.md#gradient-contracts) for the package-wide contract taxonomy.
