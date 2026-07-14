---
title: Sampled Newton-Cotes integration
short_title: Cumulative trapz
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
Use `cumulative_trapz` when a running value is needed at every sample. Use
`cumulative_simpson` only when a uniform odd-length grid and panel-endpoint
output match the problem.
:::

## Before computation: what should be true?

The selected sample axis must contain the intended ordered values. If `x` is
provided it must be one-dimensional and match that axis length. A meaningful
integral also requires coordinate units, value units, and enough resolution for
the unresolved curvature. Simpson rules additionally require at least three,
an odd number of samples, and uniform spacing.

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

`trapz(y, x=None, axis=-1)` returns a total. With no `x`, it uses unit spacing;
with `x`, each panel carries `diff(x)` inside the reduction.
`cumulative_trapz(y, x=None, dx=1.0, axis=-1)` returns the same shape as `y`
with a leading zero along `axis`.

The canonical uniform path is dx-outside: it accumulates
`0.5 * (y_left + y_right)` first and multiplies by scalar `dx` once afterward.
This is the ecosystem parity contract. The mathematically equivalent dx-inside
ordering can differ by about one unit in the last place because the multiply is
rounded at a different stage. On a nonuniform grid, every `diff(x)` must remain
inside its panel before the cumulative sum and the scalar `dx` argument is
ignored.

`simpson` returns the total of uniform two-interval quadratic panels.
`cumulative_simpson` returns only panel endpoints: input length $n$ becomes
$(n+1)/2$ along the integration axis. Concrete nonuniform `x` raises in the
wrapper, but traced value-dependent uniformity validation cannot raise.

## What JAX differentiates

For fixed coordinates, trapezoid and Simpson outputs are linear combinations of
the sampled values, so AD returns the quadrature weights. On the nonuniform
trapezoid path, JAX can also differentiate the arithmetic in `diff(x)` while
the grid ordering and shape stay fixed. That coordinate derivative represents
motion of the sampled abscissae, not automatic differentiation of an underlying
continuous function between them.

`axis` controls array structure, and cumulative integration treats it as static
under JIT. Sample count and Simpson panel count are shape choices.

:::{warning}
Concrete shape errors raise, but a value-dependent uniform-spacing check is
skipped under tracing. Compiled callers own the uniform-grid precondition for
Simpson rules. Differentiability through sample values does not validate grid
resolution or convert the discrete integral into an exact continuous one.
:::

## Using it in Jaxstro

```python
import jax.numpy as jnp

from jaxstro.numerics.integration import cumulative_trapz, trapz

x = jnp.linspace(0.0, 1.0, 101)
y = x**2
running = cumulative_trapz(y, x)
total = trapz(y, x)

assert running.shape == y.shape
assert running[0] == 0.0
assert jnp.allclose(running[-1], total)
```

For a multidimensional `y`, `x` still describes only the selected integration
axis. All other axes remain payload axes.

## How to audit the result

Integrate constants and linear functions, which trapezoids reproduce exactly.
For a smooth curved function, compare grids with spacing $h$, $h/2$, and $h/4$;
the error ratio should approach four when the global $O(h^2)$ regime is reached.
Check cumulative shape, leading zero, final-value parity with the total, both
spacing modes, negative axes, and dx-outside byte parity. Compare AD in sample
values with independently computed trapezoid weights.

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
owner signatures to [](../../50-api/approximation-integration/integration.md),
and executable checks to [](../../60-validation/validation.md). Fixed-node
Gaussian rules are in [](./quadrature.md).
:::
