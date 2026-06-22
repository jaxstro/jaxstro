---
title: Shape-preserving interpolation
description: >-
  Cubic Hermite and PCHIP-style interpolation for smooth table evaluation
  without inventing overshoot.
---

Linear interpolation is hard to beat for honesty: it does not invent curvature.
But many table-like models need a smoother derivative than straight-line
segments provide. Cubic interpolation gives that smoothness, but an unconstrained
cubic can overshoot between samples and create values the table never implied.

jaxstro's shape-preserving interpolation layer adds a small middle ground:

```python
from jaxstro.numerics import interpolation

y = interpolation.cubic_hermite_interp(x_grid, values, dydx, x_new)
y_mono = interpolation.monotone_cubic_interp(x_grid, values, x_new)
table = interpolation.MonotoneTabulatedFunction1D(x_grid, values)
```

`cubic_hermite_interp(...)` is the explicit primitive: callers provide both
values and derivatives at the knots. `monotone_cubic_interp(...)` computes
PCHIP-style limited slopes from the values and then uses the same Hermite
evaluator.

## Boundary Policy

The default policy matches `interp1d`: query points outside the grid clamp to
the endpoint values. No extrapolated curve is invented. This means gradients
with respect to `x_new` saturate outside the data domain, just as they do for
linear interpolation and clamped B-splines.

Callers can request `extrapolate=True` when they explicitly want the endpoint
Hermite segment continued beyond the table. That is a numerical choice, not a
physical guarantee.

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

Inside a fixed limiter branch, the interpolant is differentiable with respect to
the sample values and query positions. The validation suite checks this with
finite differences at points where the limiter branch is stable.

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
