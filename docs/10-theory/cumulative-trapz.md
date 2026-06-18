---
title: Newton–Cotes integration
short_title: Cumulative trapz
description: >-
  The trapezoidal rule, why the uniform path multiplies by dx outside the cumsum,
  and the ~1-ulp parity that ordering choice reconciles across the ecosystem.
---

You have a function sampled on a grid and you want its integral — or its running
integral, a CDF from a PDF. That is the trapezoidal rule, the simplest member of
the Newton–Cotes family. It looks trivial, and it almost is, except for one
ordering choice that decides whether jaxstro and progenax agree to the last bit.
This page explains the method, the choice, and why it matters.

## The trapezoidal rule

Newton–Cotes rules approximate $\int f\,dx$ by fitting a low-degree polynomial
through equally weighted samples. The trapezoidal rule is the degree-1 case: connect
adjacent samples with straight lines and sum the trapezoids. For samples
$y_0,\dots,y_{n-1}$ at spacing $h$,

```{math}
:label: eq-trapz
\int f\,dx \approx h \sum_{i=0}^{n-2} \tfrac{1}{2}\,(y_i + y_{i+1}).
```

The **cumulative** form keeps the running sum instead of the total, giving a result
the same length as the input with a leading zero — exactly what you need to turn a
density into a CDF. jaxstro provides both `trapz` (the total) and `cumulative_trapz`
(the running integral). Both are pure `jax.numpy`, so they differentiate through the
**values** $y_i$ (principle [7](./index.md#p7-quadrature)); the grid is data, not a
parameter you backprop through.

## Two spacing modes

`cumulative_trapz` has two paths, and which one runs depends on whether you pass
the grid:

- **Uniform** — you omit `x` and optionally pass a scalar `dx` (default `1.0`).
  Every trapezoid has the same width, so the width factors out of the sum.
- **Non-uniform** — you pass the grid `x`, and the spacing $\mathrm{diff}(x)$ varies
  per interval. There is no single width to factor out; `dx` is ignored.

## The dx-outside ordering (uniform path)

Here is the choice. On the uniform path you can apply the constant width $h$ in two
mathematically identical ways:

```{math}
:label: eq-dx-inside
\text{dx-inside:}\quad \mathrm{cumsum}\big(\tfrac{1}{2}\,h\,(y_i + y_{i+1})\big),
```

```{math}
:label: eq-dx-outside
\text{dx-outside:}\quad \mathrm{cumsum}\big(\tfrac{1}{2}\,(y_i + y_{i+1})\big)\times h.
```

The two are equal in exact arithmetic. In floating point they differ by **at most
about one unit in the last place (ulp)**, because the constant $h$ enters the
summation at a different point and is therefore rounded against a different running
total (principle [5](./index.md#p5-floating-point)). jaxstro standardizes on
**dx-outside**: accumulate the raw trapezoid increments first, then multiply by the
scalar `dx` exactly once at the end.

Two reasons drive the choice:

1. **Parity across the ecosystem.** dx-outside is the ordering used by the majority
   of progenax's call sites and by its local `cumulative_trapezoid`. Standardizing on
   it makes jaxstro's `cumulative_trapz` **byte-for-byte identical** to progenax's on
   shared inputs, so hoisting the function into the foundation does not perturb any
   downstream result beyond the documented ~1-ulp drift at the few former dx-inside
   sites.
2. **Marginally better numerics.** Deferring a single multiply to the end keeps one
   constant out of every term of the running sum.

:::{note} The ~1-ulp drift is expected, not a regression
Migrating a former dx-inside call site to this function can shift its result by ~1
ulp. That is the rounding difference between [](#eq-dx-inside) and
[](#eq-dx-outside), not a bug. The affected progenax sites carry test budgets that
allow it. See [](../95-release/index.md) for the reconciliation note.
:::

The non-uniform path cannot use this trick: each increment carries its own width
$\mathrm{diff}(x)_i$ *inside* the cumulative sum, because there is no shared scalar
to factor out.

## A note on Simpson's rule

The same module ships `simpson`, the degree-2 Newton–Cotes rule, which assumes
**uniform spacing** and an odd number of samples. Because a non-uniform grid would
be silently mis-integrated, the Python wrapper raises on a concrete non-uniform grid
*before* the jitted core runs — but under `jit` the grid is a tracer and the check
cannot fire, so callers inside `jit` must pass a uniform grid themselves. This is
principle [9](./index.md#p9-correctness): fail loudly where you can, document the
contract where you cannot.

## What we just established

The trapezoidal rule is degree-1 Newton–Cotes; the only subtlety is *when* you
multiply by the spacing, and the answer — dx-outside — is the one that keeps jaxstro
and progenax bit-identical. For exact integration of polynomials at far fewer
points, Gaussian quadrature is the next step up; see the API entry for the
quadrature factory in [](../40-api/index.md). The call signatures for `trapz`,
`cumulative_trapz`, and `simpson` live there too.
