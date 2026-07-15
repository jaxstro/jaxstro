---
title: Fixed and weighted quadrature
description: >-
  Gaussian, Clenshaw-Curtis, Fejer, and tanh-sinh formulas with explicit
  measures, domain maps, exactness, JAX behavior, and audit boundaries.
---

## The question this method answers

How can we approximate a one-dimensional integral when we can choose where to
evaluate the integrand, but do not need an adaptive error controller? Fixed
quadrature replaces the continuous integral by a finite weighted sum. It is a
good fit when the integrand is inexpensive, the domain and measure are known,
and convergence can be audited by increasing a static order or level.

:::{tip}
Start from the structure of the integral. Use a matched Gaussian rule when the
measure is classical, a Chebyshev rule for a smooth finite-interval integrand,
and tanh-sinh when endpoint behavior or an infinite-domain map is central.
:::

## Before computation: what should be true?

An integral depends on more than a function. It also depends on a domain
$D$ and a measure $\mu$. In density form,

```{math}
:label: eq-fixed-node-overview
I[f]
=\int_D f(x)\,\mathrm{d}\mu(x)
=\int_D f(x)\omega(x)\,\mathrm{d}x
\approx Q_n[f]
=\sum_{i=1}^{n}w_i f(x_i).
```

The nodes $x_i$ say where to evaluate the integrand. The weights $w_i$ encode
the rule, the measure, and any domain transformation. A fixed formula does not
observe its own error. Accuracy must be established from exactness identities,
convergence across orders, or an independent reference.

:::{important}
The integrand must accept the complete node array and return an array whose
leading axis is the node axis. Scalar, vector, complex, and higher-rank payloads
are supported after that leading axis.
:::

## Define the mathematical objects

The numerical problem consists of an integrand, a one-dimensional `Domain`, a
declared `Measure`, and a static fixed `Rule`. `Interval` may contain a static
number of dynamic breakpoint values. `RightInfinite`, `LeftInfinite`, and
`Infinite` identify improper domains without hiding a transformation choice.

`GaussianRule`, `ClenshawCurtisRule`, `FejerIRule`, `FejerIIRule`, and
`TanhSinhRule` are frozen configuration objects. Constructed nodes and weights
are arrays; rule exactness and nesting are static metadata.

## Derive the method

Every rule in this family evaluates the same fixed-sum abstraction,

```{math}
:label: eq-fixed-node-quadrature
I[f]=\int_D f(x)\omega(x)\,\mathrm{d}x
\approx Q_n[f]=\sum_{i=1}^{n}w_i f(x_i).
```

### Gaussian rules from one recurrence engine

Let $p_k$ be polynomials orthogonal under the declared measure. Their
three-term recurrence defines a symmetric Jacobi matrix,

```{math}
:label: eq-gaussian-jacobi-matrix
J_n=
\begin{bmatrix}
a_0 & \sqrt{b_1} \\
\sqrt{b_1} & a_1 & \sqrt{b_2} \\
& \ddots & \ddots & \ddots \\
&& \sqrt{b_{n-1}} & a_{n-1}
\end{bmatrix}.
```

The eigenvalues of $J_n$ are the Gaussian nodes. If $v_i$ is the normalized
eigenvector associated with node $x_i$, then the weight is the measure mass
$\mu_0$ times the square of the first eigenvector component,

```{math}
:label: eq-gaussian-eigenweight
w_i=\mu_0\left(v_{0i}\right)^2.
```

This single construction produces Gauss-Legendre, Gauss-Jacobi,
Gauss-Laguerre, generalized Gauss-Laguerre, physicists' Gauss-Hermite, and
standard-normal Gauss-Hermite rules. An $n$-node Gaussian rule satisfies

```{math}
:label: eq-gaussian-exactness
\sum_{i=1}^n w_i p(x_i)
=\int_D p(x)\omega(x)\,\mathrm{d}x,
\qquad \deg p\le 2n-1.
```

The degree statement is exact for the matched polynomial class. It is not a
general error estimate.

#### Classical measure conventions

Jaxstro fixes the density and parameter orientation rather than relying on a
family name alone. The Gaussian recurrence uses the following reference
measures:

| Declaration | Coordinate and support | Unnormalized density | Total mass |
| --- | --- | --- | --- |
| `LebesgueMeasure()` | $t\in[-1,1]$ | $1$ | $2$ |
| `JacobiMeasure(alpha, beta)` | $t\in[-1,1]$ | $(1-t)^{\alpha}(1+t)^{\beta}$ | $2^{\alpha+\beta+1}B(\alpha+1,\beta+1)$ |
| `LaguerreMeasure(alpha)` | $u\in[0,\infty)$ | $u^{\alpha}e^{-u}$ | $\Gamma(\alpha+1)$ |
| `PhysicistsHermiteMeasure()` | $x\in(-\infty,\infty)$ | $e^{-x^2}$ | $\sqrt{\pi}$ |
| `StandardNormalMeasure()` | $x\in(-\infty,\infty)$ | $e^{-x^2/2}/\sqrt{2\pi}$ | $1$ |

For Jacobi and generalized Laguerre, $\alpha>-1$ and $\beta>-1$. Setting
`normalized=True` divides the reference weights by the total mass in the last
column. It does not estimate a normalization numerically.

On `Interval(a, b)`, let

```{math}
:label: eq-jacobi-reference-pushforward
m=\frac{a+b}{2},
\qquad
h=\frac{|b-a|}{2},
\qquad
s=\operatorname{sign}(b-a).
```

Jaxstro interprets the Jacobi density in the reference coordinate $t$ and
returns

```{math}
:label: eq-jacobi-interval-convention
Q_n[f]
=s h\sum_{i=1}^{n}w_i f(m+h t_i)
\approx
s h\int_{-1}^{1}f(m+h t)
(1-t)^{\alpha}(1+t)^{\beta}\,\mathrm{d}t.
```

Thus `alpha` belongs to the $t=+1$ endpoint and `beta` belongs to the $t=-1$
endpoint. This is a reference-density convention; it is not silently replaced
by the physical density $(b-x)^{\alpha}(x-a)^{\beta}$. Jacobi rules reject
breakpoints because applying a new reference density on every segment would
change the declared measure.

For `RightInfinite(lower)`, generalized Laguerre uses the shifted coordinate
$u=x-\mathtt{lower}$:

```{math}
:label: eq-laguerre-shift-convention
Q_n[f]
=\sum_{i=1}^{n}w_i f(\mathtt{lower}+u_i)
\approx
\int_{0}^{\infty}f(\mathtt{lower}+u)u^{\alpha}e^{-u}\,\mathrm{d}u.
```

#### The standard-normal convention

The legacy compatibility helper begins with the physicists' Hermite rule and
uses $g=\sqrt{2}z$. The normalized weights are

```{math}
:label: eq-standard-normal-hermite
g_i=\sqrt{2}\,z_i,
\qquad
\widetilde{w}_i=\frac{w_i}{\sqrt{\pi}},
\qquad
\mathbb{E}_{g\sim\mathcal{N}(0,1)}[f(g)]
\approx\sum_i\widetilde{w}_i f(g_i).
```

That helper remains byte-compatible with the earlier public implementation.
New `GaussianRule` construction uses the shared JAX recurrence engine.

### Finite domains and weighted measures

For a finite interval with ordered physical endpoints $x_{\min}$ and
$x_{\max}$, Jaxstro maps $t\in[-1,1]$ by

```{math}
:label: eq-fixed-affine-map
x(t)=\frac{x_{\min}+x_{\max}}{2}
+\frac{x_{\max}-x_{\min}}{2}t,
\qquad
\left|\frac{\mathrm{d}x}{\mathrm{d}t}\right|
=\frac{x_{\max}-x_{\min}}{2}.
```

The orientation sign is stored separately, so reversing the requested bounds
negates the result without making the measure Jacobian negative. Breakpoints
produce a static collection of subintervals evaluated together for Lebesgue
and general weighted formulas. Their values are stopped in derivatives, and
Jacobi rules reject them for the measure reason above.

`WeightedMeasure` evaluates its declared density exactly once. A matched
Gaussian rule already contains its classical weight and therefore does not
multiply that weight into the integrand again. `normalized=True` changes only
the declared classical measure mass; it does not trigger a hidden numerical
normalization.

:::{warning}
A classical measure is part of the quadrature formula. Do not manually multiply
the same density into `fun` when using a matched `GaussianRule`.
:::

### Clenshaw-Curtis and Fejer rules

The Chebyshev families interpolate the integrand at cosine-spaced nodes. Their
weights are obtained by matching the exact Chebyshev moments

```{math}
:label: eq-chebyshev-moments
\int_{-1}^{1}T_k(x)\,\mathrm{d}x
=
\begin{cases}
\dfrac{2}{1-k^2}, & k\ \text{even},\\
0, & k\ \text{odd}.
\end{cases}
```

Clenshaw-Curtis includes both endpoints and is nested when the number of
intervals doubles. Fejer type I and type II exclude the endpoints. All three
families share the same cosine-interpolation substrate rather than duplicating
weight formulas.

### Fixed tanh-sinh

Tanh-sinh begins with an evenly spaced parameter $s_k=kh$ and maps it to the
reference interval by

```{math}
:label: eq-tanh-sinh-map
t(s)=\tanh\!\left(\frac{\pi}{2}\sinh s\right),
\qquad
\frac{\mathrm{d}t}{\mathrm{d}s}
=\frac{\pi}{2}
\frac{\cosh s}{\cosh^2\!\left(\frac{\pi}{2}\sinh s\right)}.
```

The derivative decays double-exponentially near $t=\pm1$. Jaxstro composes
this formula with explicit maps for finite, semi-infinite, and full-line
domains. Representable endpoint distance eventually limits float64 accuracy
for an integrand that diverges exactly at an endpoint; increasing the level
past that point cannot recover information absent from the dtype.

## What the algorithm actually does

`quad.fixed` performs the following static computation:

1. Select a rule construction from the static rule and measure types.
2. Construct nodes and weights at the static order or level.
3. Map all nodes to the requested domain and breakpoint segments.
4. Evaluate the integrand with one leading node axis.
5. Apply a general density exactly once when one is declared.
6. Reduce the node axis and sum the static segment axis.

For $n$ nodes and $m$ breakpoint segments, the integrand receives $mn$ points.
Gaussian construction includes a symmetric eigensolve of size $n$. Chebyshev
construction solves the static cosine interpolation system. Repeated workloads
should close over the rule so compilation can treat its construction as static.

## What JAX differentiates

Rule type, order or level, measure type, breakpoint count, and payload shape are
static. Bounds, breakpoint values, and explicit integrand parameters may be JAX
arrays. The fixed evaluator supports `jax.jit` and `jax.vmap` under those
conditions.

JAX differentiates the executed weighted sum. For smooth finite bounds this
includes the affine node motion and Jacobian. This is a fixed-formula
derivative, not proof that quadrature error is sufficiently small for the
derivative integrand.

:::{note}
Adaptive replay differentiation is a separate Phase A3 contract. No adaptive
controller is implemented by the fixed evaluator.
:::

### Units, shapes, and precision

Phase A1 accepts raw arrays. The caller owns units and must ensure that the
integrand value multiplied by the measure has the intended integral dimension.
Quantity-valued boundaries remain planned for Phase A3.

The node input has shape `(n,)`. The integrand returns `(n,)` or `(n, ...)`, and
the result has shape `(...)`. Scientific reference tests use float64. The active
JAX precision policy controls normal execution.

## Using it in Jaxstro

```python
import jax.numpy as jnp

from jaxstro import quad

polynomial = quad.fixed(
    lambda x: x**4,
    quad.Interval(-1.0, 1.0),
    rule=quad.GaussianRule(3),
)

normal_variance = quad.fixed(
    lambda x: x**2,
    quad.Infinite(),
    rule=quad.GaussianRule(12),
    measure=quad.StandardNormalMeasure(),
)

assert jnp.allclose(polynomial, 2.0 / 5.0)
assert jnp.allclose(normal_variance, 1.0)
```

The compatibility node helpers remain available:

```python
nodes, weights = quad.gauss_legendre_nodes(8)
assert nodes.shape == weights.shape == (8,)
```

## How to audit the result

For Gaussian rules, verify analytic moments through degree $2n-1$. For
Clenshaw-Curtis and Fejer rules, verify their declared interpolatory degree and
then compare increasing orders on the actual integrand. For tanh-sinh, sweep
levels until the result stabilizes before the dtype endpoint floor is reached.

The implementation is checked against independent SciPy roots and weights for
all classical Gaussian families. JIT, VMAP, parameter gradients, moving-bound
gradients, complex payloads, reversed intervals, breakpoints, and invalid
pairings have executable tests. Evidence is indexed in
[](../../60-validation/validation.md).

## Where the claim stops

Fixed quadrature does not estimate error, choose an order, diagnose divergence,
or certify interchange of differentiation and integration. A converged-looking
order sweep is evidence for the tested sequence, not a universal guarantee.
Adaptive Gauss-Kronrod, adaptive Clenshaw-Curtis, adaptive tanh-sinh, and
Romberg controllers are Phase A2 work.

## Connected ideas

:::{seealso}
Connect measures to
[](../../10-foundations/mathematical-objects/probability-and-distributions.md),
units to [](../../30-representations/units-quantities/quantities.md), the public
owner to [](../../50-api/approximation-integration/quad.md), sampled-data rules
to [](./cumulative-trapz.md), and the planned controller layer to
[](./adaptive-quadrature.md).
:::
