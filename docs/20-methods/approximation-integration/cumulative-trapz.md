---
title: Sampled Newton-Cotes integration
short_title: Cumulative trapezoid
description: >-
  Trapezoid accumulation, local and global discretization error, and the
  canonical dx-outside floating-point ordering.
---

## The question this method answers

Given function values sampled along one coordinate, what integral or running
integral do those samples imply under a piecewise-linear approximation? The
trapezoidal rule is the degree-one Newton-Cotes rule. Its cumulative form can,
for example, turn sampled density values into an approximate cumulative integral.

:::{tip}
Use `cumulative_trapezoid` when a running value is needed at every sample. Use
`cumulative_simpson` only when a uniform odd-length grid and panel-endpoint
output match the problem.
:::

## Before computation: what should be true?

The integrated axis (the last axis by default) must contain the intended ordered
values. If `x` is provided it must be one-dimensional and match that axis length. A meaningful integral also requires coordinate
units, value units, and enough resolution for the unresolved curvature. Simpson
rules additionally require at least three, an odd number of samples, and
uniform spacing.

:::{important}
An integration rule computes the integral of an interpolating approximation.
It does not estimate its own discretization error. Plan a grid-refinement audit
and track the product of coordinate and value units.
:::

Coordinate semantics connect to
[](../../10-foundations/mathematical-objects/functions-units-scales.md) and
explicit unit representations to
[](../../30-representations/units-quantities/quantities.md).

## Define the mathematical objects

Let $x_0<\cdots<x_{n-1}$ be sample coordinates and $y_i=f(x_i)$ their values.
The width of panel $i$ is $h_i=x_{i+1}-x_i$. A cumulative integral $C_j$
approximates $\int_{x_0}^{x_j}f(x)\,dx$ and therefore has the same leading sample
count when a zero is stored at $j=0$.

On a uniform grid, $h_i=h$. Local quadrature error is the error on one panel;
global error is the sum across all panels over a fixed interval.

## Derive the method

Integrating the straight line through adjacent samples gives one trapezoid:

```{math}
:label: eq-trapezoid-panel
T_i=\frac{x_{i+1}-x_i}{2}(y_i+y_{i+1}).
```

The running integral is the prefix sum

```{math}
:label: eq-cumulative-trapezoid
C_0=0,\qquad C_j=\sum_{i=0}^{j-1}T_i,
\qquad j=1,\ldots,n-1.
```

For a twice continuously differentiable function on a uniform grid, Taylor
expansion of one panel and accumulation over $O(1/h)$ panels give

```{math}
:label: eq-trapezoid-error
\text{local panel error}=O(h^3),\qquad
\text{global fixed-interval error}=O(h^2).
```

The order statement is asymptotic and depends on smoothness; it is not an error
bar for one grid.

For uniform spacing, exact arithmetic permits either
$\operatorname{cumsum}[(y_i+y_{i+1})/2]h$ or
$\operatorname{cumsum}[h(y_i+y_{i+1})/2]$. Floating-point rounding makes their
last bits differ.

## What the algorithm actually does

`trapezoid(y, x=None, *, dx=1.0, axis=-1)` returns the total along `axis`. With no
`x`, it uses the scalar spacing `dx`; with `x`, each panel carries `diff(x)` inside
the reduction and `dx` is ignored. `axis` is a static argument; uniform and
nonuniform spacing are supported on any axis, with `diff(x)` broadcast along
`axis`.

`cumulative_trapezoid(y, x=None, *, dx=1.0, axis=-1)` returns the same shape as `y`
with a leading zero along `axis`, under the same spacing rules.

The canonical uniform path is dx-outside: it accumulates
`0.5 * (y_left + y_right)` first and multiplies by scalar `dx` once afterward.
This is the ecosystem parity contract. The mathematically equivalent dx-inside
ordering can differ by about one unit in the last place because the multiply is
rounded at a different stage. On a nonuniform grid, every `diff(x)` must remain
inside its panel before the cumulative sum and the scalar `dx` argument is
ignored. Until 2026-09-24 the widths were broadcast along the last axis
regardless of `axis`; for a shape such as `(7, 4)` with `axis=0` that returned
wrong values without an error.

`simpson` returns the total of two-interval quadratic panels. With `x`, each
panel $[x_0,x_2]$ with widths $h_0$, $h_1$ uses

```{math}
:label: eq-nonuniform-simpson
\int_{x_0}^{x_2}y\,\mathrm{d}x\approx\frac{h_0+h_1}{6}\left[
\left(2-\frac{h_1}{h_0}\right)y_0+\frac{(h_0+h_1)^2}{h_0h_1}y_1
+\left(2-\frac{h_0}{h_1}\right)y_2\right],
```

which is exact for quadratics on any grid and reduces to $h(y_0+4y_1+y_2)/3$
when $h_0=h_1=h$. Before 2026-09-24 a nonuniform `x` raised only when concrete;
under `jit` the uniform formula was applied to it silently.
`cumulative_simpson` returns only panel endpoints: input length $n$ becomes
$(n+1)/2$ along the integration axis.

## What JAX differentiates

For fixed coordinates, trapezoid and Simpson outputs are linear combinations of
the sampled values, so AD returns the quadrature weights. On the nonuniform
trapezoid path, JAX can also differentiate the arithmetic in `diff(x)` while
the grid ordering and shape stay fixed. That coordinate derivative represents
motion of the sampled abscissae, not automatic differentiation of an underlying
continuous function between them.

Sample count and Simpson panel count are shape choices. `axis` is static and
selects which array dimension is integrated; it is not a differentiable input.

:::{warning}
Concrete shape errors raise, but a value-dependent uniform-spacing check is
skipped under tracing. Compiled callers own the uniform-grid precondition for
Simpson rules. Differentiability through sample values does not validate grid
resolution or convert the discrete integral into an exact continuous one.
:::

## Using it in Jaxstro

```python
import jax.numpy as jnp

from jaxstro import quad

x = jnp.linspace(0.0, 1.0, 101)
y = x**2
running = quad.cumulative_trapezoid(y, x)
total = quad.trapezoid(y, x)

assert running.shape == y.shape
assert running[0] == 0.0
assert jnp.allclose(running[-1], total)
```

For a multidimensional `y`, place the integration coordinate on the last axis;
`x` describes that last axis and all preceding axes remain payload axes.

## How to audit the result

Integrate constants and linear functions, which trapezoids reproduce exactly.
For a smooth curved function, compare grids with spacing $h$, $h/2$, and $h/4$;
the error ratio should approach four when the global $O(h^2)$ regime is reached.
Check cumulative shape, leading zero, final-value parity with the total, both
spacing modes on the default last axis, uniform spacing on an explicit axis, and
dx-outside byte parity. Compare AD in sample values with independently computed
trapezoid weights. Keep an explicit failure probe for nonuniform multidimensional
integration on a non-last axis so that this limitation cannot be mistaken for a
supported path.

The package evidence index is [](../../60-validation/validation.md).

## Where the claim stops

The routines do not sort coordinates, estimate truncation error, detect
under-resolution, attach units, or certify convergence. The roughly one-ulp
dx-ordering difference is a floating-point implementation fact, not a bound on
the much larger possible discretization error. Simpson's nominal order does not
apply to a nonuniform or nonsmooth case outside its assumptions.

## Connected ideas

:::{seealso}
Connect integration units to
[](../../30-representations/units-quantities/quantities.md), approximation error
to [](../../10-foundations/models-and-computation/sensitivity-conditioning-identifiability.md),
owner signatures to [](../../50-api/approximation-integration/quad.md),
and executable checks to [](../../60-validation/validation.md). The
[legacy sampled-integration page](../../50-api/approximation-integration/integration.md)
records the temporary import-name mapping. Fixed-node Gaussian rules are in
[](./quadrature.md).
:::
