---
title: Cubic interpolation
description: >-
  Cubic Hermite, natural cubic spline, and PCHIP-style interpolation for smooth
  one-dimensional table evaluation.
---

Linear interpolation is hard to beat for honesty: it does not invent curvature.
But many table-like models need a smoother derivative than straight-line
segments provide. Cubic interpolation gives that smoothness, but an unconstrained
cubic can overshoot between samples and create values the table never implied.

jaxstro's shape-preserving interpolation layer adds a small middle ground:

```python
from jaxstro.numerics import interpolation

y = interpolation.cubic_hermite_interp(x_grid, values, dydx, x_new)
y_spline = interpolation.eval_cubic_spline(
    x_grid,
    interpolation.natural_cubic_spline_coeffs(x_grid, values),
    x_new,
)
y_mono = interpolation.monotone_cubic_interp(x_grid, values, x_new)
table = interpolation.MonotoneTabulatedFunction1D(x_grid, values)
```

`cubic_hermite_interp(...)` is the explicit primitive: callers provide both
values and derivatives at the knots. `natural_cubic_spline_coeffs(...)` computes
the per-interval coefficients for a twice-continuously differentiable cubic
spline with natural boundary conditions. `monotone_cubic_interp(...)` computes
PCHIP-style limited slopes from the values and then uses the same Hermite
evaluator when avoiding overshoot is more important than global smoothness.

## Boundary Policy

The default policy matches `interp1d`: query points outside the grid clamp to
the endpoint values. No extrapolated curve is invented. This means gradients
with respect to `x_new` saturate outside the data domain, just as they do for
linear interpolation and clamped B-splines.

Callers can request `extrapolate=True` when they explicitly want the endpoint
Hermite segment continued beyond the table. That is a numerical choice, not a
physical guarantee.

Natural cubic spline evaluation clamps query coordinates to the data domain
before choosing an interval, so out-of-range queries return endpoint values.

## Natural Cubic Spline

The natural cubic spline uses the standard second-derivative moment system
described in Numerical Recipes §3.3 and de Boor's spline text. With

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
