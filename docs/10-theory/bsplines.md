---
title: B-splines
description: >-
  Local, smooth, AD-friendly basis functions for representing 1D tabulated
  functions without global-polynomial pathologies.
---

B-splines are a way to represent a smooth function as a local weighted sum:

```{math}
S(x) = \sum_i c_i B_{i,p}(x),
```

where `p` is the degree, `c_i` are coefficients, and `B_{i,p}` are basis
functions defined by a knot vector. The reason they belong in a differentiable
foundation package is simple: each basis function has local support, the basis
is nonnegative, and inside the active knot domain the basis functions form a
partition of unity.

That gives a stable primitive for downstream packages that need smooth
table-like functions: atmosphere-grid interpolation, microphysics tables,
stellar tracks, calibration curves, or any other place where global polynomials
would be too eager to oscillate.

## The current boundary

jaxstro's spline surface is deliberately fixed-knot first:

```python
from jaxstro.numerics import (
    BSpline1D,
    bspline_derivative,
    bspline_eval,
    fit_bspline_lstsq,
    open_uniform_knots,
)

knots = open_uniform_knots(0.0, 1.0, n_basis=6, degree=3)
coeffs = ...
y = bspline_eval(knots, coeffs, x, degree=3)
dy_dx = bspline_derivative(knots, coeffs, x, degree=3)

coeffs_fit = fit_bspline_lstsq(knots, x_samples, y_samples, degree=3)
spline = BSpline1D(knots, coeffs, degree=3)
```

It evaluates supplied coefficients, computes derivative values, exposes the
sample design matrix, and solves ordinary least-squares fits for fixed knots. It
does not smooth noisy observations, adapt knot locations, construct
tensor-product splines, or choose a regularization policy. Those are important
follow-up capabilities, but each adds a modeling choice: penalty type, knot
selection, boundary behavior, and validation targets. The foundation primitive
stays small and trusted.

## Knots and clamping

A clamped open-uniform knot vector repeats the first and last knot `degree + 1`
times. For a cubic spline on `[0, 1]`, a single-span knot vector is:

```text
0 0 0 0 1 1 1 1
```

This makes the first and last coefficients control the endpoint values. jaxstro's
`open_uniform_knots(...)` constructs this layout for any valid `n_basis` and
degree.

Inputs outside the active knot domain are clamped to the endpoint basis values.
This matches the existing fail-closed posture of `interp1d`: no extrapolated
curve is invented. The trade-off is the same as any hard saturation: gradients
with respect to `x` are zero outside the active domain. If an optimizer needs to
move an out-of-domain `x` back into range, the caller should handle the domain
constraint explicitly rather than relying on spline extrapolation.

## Cox-de Boor recurrence

The degree-zero basis is an interval indicator:

```{math}
B_{i,0}(x) =
\begin{cases}
1, & t_i \le x < t_{i+1} \\
0, & \text{otherwise}.
\end{cases}
```

Higher degrees are built recursively:

```{math}
B_{i,p}(x) =
\frac{x - t_i}{t_{i+p} - t_i} B_{i,p-1}(x)
+
\frac{t_{i+p+1} - x}{t_{i+p+1} - t_{i+1}} B_{i+1,p-1}(x).
```

Repeated knots make some denominators zero. The implementation uses the standard
safe convention: a term with a zero denominator contributes zero. This is also
the AD-safe convention. The denominator is sanitized before division, so a dead
zero-width term does not leak `NaN` into the backward pass.

## Differentiability

For fixed knots, spline evaluation is linear in the coefficients:

```{math}
\frac{\partial S(x)}{\partial c_i} = B_{i,p}(x).
```

That property is tested directly: the AD gradient with respect to coefficients
matches the basis vector. Gradients with respect to interior `x` are checked
against finite differences in the validation suite. At knots, the derivative
order depends on the knot multiplicity, so tests use interior points rather than
pretending every knot is smooth.

The analytic derivative uses the standard coefficient transform:

```{math}
c'_i =
p \frac{c_{i+1} - c_i}{t_{i+p+1} - t_{i+1}},
```

then evaluates a degree `p - 1` spline on the trimmed knot vector. Zero-width
denominators use the same safe-zero convention as the basis recurrence. Outside
the active knot domain, `bspline_derivative(...)` returns zero, matching the
gradient of the public clamped evaluator with respect to `x`.

Fixed-knot least-squares fitting solves the linear design problem:

```{math}
\mathbf{B}\mathbf{c} \approx \mathbf{y}.
```

It is a convenience around the basis matrix, not a smoothing spline. If noisy
data require penalties or priors, the caller should build that objective
explicitly until jaxstro has a separately validated regularized fitter.

## Why not de Boor first?

The de Boor algorithm is the standard stable evaluator for a single spline value
when you already know the active knot span. jaxstro's first implementation
instead exposes the full basis vector because it is the more useful foundation
primitive:

- it makes `dS/dc` obvious and testable;
- it supports vector-valued coefficients with a single contraction;
- it lets downstream packages build sparse or tensor-product designs later;
- it keeps the v1 API close to the mathematical object being validated.

A future de Boor evaluator can be added as an optimized kernel once profiling
shows a real bottleneck. The public mathematical contract should remain the same.
