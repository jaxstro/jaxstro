---
title: Distribution kernels
description: >-
  Support-aware log densities, CDFs, and inverse CDFs with a smooth finite
  power-law limit.
---

## The question this method answers

How can scientific code evaluate a probability law, accumulate probability up
to a threshold, or map a unit-uniform value into that law? Jaxstro supplies
small distribution kernels without owning probabilistic-program syntax, priors,
samplers, or inference policy.

:::{important}
A normalized formula is a mathematical probability law only on its stated
support and for valid parameters. A finite runtime value does not certify those
conditions.
:::

## Before computation: what should be true?

Choose a valid parameter domain: positive normal scale, positive lognormal
coordinates, ordered truncation bounds, and `0 < xmin < xmax` for a finite power
law. Pass probabilities $u$ in $[0,1]$ to inverse CDFs. Decide whether endpoint
infinities for unbounded distributions are acceptable.

:::{warning}
The distribution kernels do not validate parameter domains. Invalid scale,
bounds, support, or probability values can return `NaN`, infinity, or a value
outside the intended law in both eager and traced execution.
:::

## Define the mathematical objects

The support $\mathcal{S}$ is the set of allowed values. A probability density
$p(x)$ is nonnegative and normalized:

```{math}
\int_{\mathcal{S}}p(x)\,dx=1.
```

The cumulative distribution function (CDF) is the probability at or below a
threshold:

```{math}
F(x)=\int_{-\infty}^{x}p(t)\,dt.
```

The percent-point function (PPF), or inverse CDF, returns a quantile $x$ whose
cumulative probability is $u$. For a continuous strictly increasing CDF,
$F(F^{-1}(u))=u$ for $0<u<1$.

## Derive the method

Probability normalization and cumulative mass are

```{math}
:label: eq-density-normalization
\int_{\mathcal{S}}p(x)\,dx=1.
```

```{math}
:label: eq-cdf-definition
F(x)=\int_{-\infty}^{x}p(t)\,dt.
```

For an interior probability on a continuous strictly increasing branch, the
inverse relation to audit is

```{math}
F(F^{-1}(u))=u.
```

### A finite power law through the logarithmic limit

For $p(x)\propto x^\alpha$ on $[x_{\min},x_{\max}]$, let
$e=\alpha+1$ and $D=\log(x_{\mathrm{hi}})-\log(x_{\mathrm{lo}})$. A stable
segment integral is

```{math}
:label: eq-powerlaw-integral
I(x_{\mathrm{lo}},x_{\mathrm{hi}},e)
=x_{\mathrm{lo}}^eD\,\phi(eD),
\qquad
\phi(z)=\frac{\operatorname{expm1}(z)}{z}.
```

At $z=0$, $\phi(z)=1+z/2+z^2/6+O(z^3)$. The inverse uses
$\psi(z)=\log(1+z)/z=1-z/2+z^2/3+O(z^3)$:

```{math}
x=\exp\!\left[\log(x_{\mathrm{lo}})+s\,\psi(es)\right],
\qquad
s=t x_{\mathrm{lo}}^{-e}.
```

The normalizer is $I(x_{\min},x_{\max},e)^{-1}$, the CDF is a partial-to-total
integral ratio, and the PPF applies the inverse at $t=uI$. The Taylor branches
sanitize the dangerous denominator before selection, so the value and parameter
derivative remain smooth through $\alpha=-1$.

At the exact limit, with $A=\log x_{\min}$, $B=\log x_{\max}$,
$L=B-A$, $\ell=\log(x/x_{\min})$, and $x_u=x_{\min}\exp(uL)$,

```{math}
\frac{\partial\log p(x)}{\partial\alpha}=\log x-\frac{A+B}{2},
\quad
\frac{\partial F(x)}{\partial\alpha}=\frac{\ell(\ell-L)}{2L},
\quad
\frac{\partial F^{-1}(u)}{\partial\alpha}=\frac{x_uL^2u(1-u)}{2}.
```

## What the algorithm actually does

The module includes normal, lognormal, finite power-law, and truncated-normal
`logpdf`, `cdf`, and `ppf` families. Array inputs broadcast with parameters.
Lognormal and power-law log densities return negative infinity outside support;
their unsafe logarithm operands are replaced before `where` selection. Their
CDFs clamp below and above finite support. PPF functions assume, but do not
check, $u\in[0,1]$.

The truncated-normal normalizer is a difference of two normal CDF values. Very
narrow or tail truncations can lose precision or yield a zero denominator. The
current kernel has no specialized log-difference fallback for that regime.

## What JAX differentiates

Inside a smooth support region with valid parameters, JAX differentiates the
executed log-density, CDF, and PPF formulas. The finite power-law implementation
supports smooth derivatives through $\alpha=-1$.

Support masks, CDF clipping, and endpoint choices are branch boundaries. The
log-density derivative with respect to $x$ is not meaningful at a hard support
edge, and normal/lognormal PPF sensitivities diverge near $u=0$ or $1$.
CDF/PPF round-trip parity checks values; it does not by itself validate an AD
claim at boundaries.

## Using it in Jaxstro

```python
import jax
import jax.numpy as jnp

from jaxstro.numerics.distributions import (
    powerlaw_cdf,
    powerlaw_logpdf,
    powerlaw_ppf,
)

x = jnp.array([1.0, 2.0, 4.0])
u = jnp.array([0.1, 0.5, 0.9])
log_density = powerlaw_logpdf(x, alpha=-1.0, xmin=1.0, xmax=4.0)
quantiles = powerlaw_ppf(u, alpha=-1.0, xmin=1.0, xmax=4.0)
round_trip = powerlaw_cdf(quantiles, alpha=-1.0, xmin=1.0, xmax=4.0)
alpha_gradient = jax.grad(
    lambda alpha: powerlaw_logpdf(2.0, alpha=alpha, xmin=1.0, xmax=4.0)
)(-1.0)

assert jnp.all(jnp.isfinite(log_density))
assert jnp.isneginf(powerlaw_logpdf(0.5, xmin=1.0, xmax=4.0))
assert jnp.allclose(round_trip, u)
assert jnp.isfinite(alpha_gradient)
```

## How to audit the result

1. Numerically integrate each density over its support and compare with one.
2. Check support values, CDF monotonicity, and endpoint behavior.
3. Evaluate `cdf(ppf(u))` on interior probabilities, including near tails.
4. Compare the power-law value and analytic derivatives at $\alpha=-1$.
5. Compare AD with independent central differences on both sides of the limit.
6. Record dtype, grid resolution, integration error, and maximum round-trip error.

:::{tip}
Separate a normalization audit from a round-trip audit. Two mutually consistent
formulas can round-trip while sharing the same wrong normalization.
:::

For `xmin=2`, `xmax=5`, `x=3`, `u=0.3`, and float64 central-difference step
`1e-5`, the established evidence records maximum CDF/PPF round-trip error
$2.22\times10^{-16}$ and numerical normalization error
$1.60\times10^{-10}$. These are fixture measurements, not universal bounds.

## Where the claim stops

These kernels do not validate data-generating assumptions, fit parameters,
construct priors, supply random draws, or establish Monte Carlo accuracy.
Support behavior does not make a masked boundary differentiable. Truncated-tail
accuracy and parameter domains remain caller responsibilities.

## Connected ideas

:::{seealso}
Build probability vocabulary in
[](../../10-foundations/mathematical-objects/probability-and-distributions.md),
connect probability to model state in
[](../../30-representations/uncertainty/what-uncertainty-represents.md), and use
the removable-limit audit pattern in
[](../../40-workflows/investigations/powerlaw-removable-limit.md).
The grouped API is [](../../50-api/randomness/distributions.md), and numerical
evidence belongs in [](../../60-validation/validation.md).
:::
