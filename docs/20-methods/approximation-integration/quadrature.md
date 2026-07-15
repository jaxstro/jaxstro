---
title: Fixed-node quadrature
description: >-
  Gaussian and Clenshaw-Curtis rules with explicit polynomial exactness,
  standard-normal convention, and fixed-node AD boundaries.
---

## The question this method answers

How can an integral be approximated by evaluating an integrand at a small,
carefully chosen set of fixed nodes? Fixed-node quadrature replaces integration
against a known weight function by a weighted sum. Jaxstro owns deterministic
node factories and Hermite expansion helpers, not adaptive error control.

:::{tip}
Match the rule to the domain and weight: Legendre for unit weight on $[-1,1]$,
Laguerre for $e^{-x}$ on $[0,\infty)$, and Jaxstro's probabilists' Hermite rule
for expectations under a standard normal density.
:::

## Before computation: what should be true?

Define the integration domain, weight $\omega(x)$, integrand units, and required
accuracy. Choose a concrete positive node count. Determine whether polynomial
exactness or empirical convergence on a non-polynomial integrand is the relevant
audit. Treat nodes and weights as setup constants, not inferred parameters.

:::{important}
The statement "exact through degree $2n-1$" applies to the matched Gaussian
weight and polynomial class. It is not an error estimate for an arbitrary
integrand.
:::

Probability-weighted integrals connect to
[](../../10-foundations/mathematical-objects/probability-and-distributions.md)
and coordinate/quantity semantics to
[](../../30-representations/units-quantities/quantities.md).

## Define the mathematical objects

Let $I[f]=\int_D f(x)\omega(x)\,dx$. A rule contains nodes
$x_i\in D$ and scalar weights $w_i$. Its approximation is
$Q_n[f]=\sum_{i=1}^n w_i f(x_i)$. Polynomial exactness of degree $q$ means
$Q_n[p]=I[p]$ for every polynomial $p$ with degree at most $q$.

Gaussian nodes are roots of the orthogonal polynomial associated with
$\omega$. Clenshaw-Curtis instead uses Chebyshev-Lobatto nodes including the
endpoints of $[-1,1]$.

## Derive the method

An $n$-node Gaussian rule chooses $2n$ node and weight degrees of freedom so
that the first $2n$ weighted moments match. The resulting exactness statement is

```{math}
:label: eq-gaussian-exactness
\sum_{i=1}^n w_i p(x_i)
=\int_D p(x)\omega(x)\,dx,
\qquad \deg p\le2n-1.
```

For a general integrand the same nodes define the approximation

```{math}
:label: eq-fixed-node-quadrature
I[f]=\int_D f(x)\omega(x)\,dx
\approx Q_n[f]=\sum_{i=1}^n w_i f(x_i).
```

NumPy supplies the physicists' Hermite rule
$\int e^{-z^2}f(z)\,dz\approx\sum_i w_i f(z_i)$. Substituting
$g=\sqrt{2}z$ and normalizing the Gaussian density gives Jaxstro's standard-
normal rule:

```{math}
:label: eq-standard-normal-hermite
g_i=\sqrt{2}\,z_i,\qquad
\widetilde{w}_i=\frac{w_i}{\sqrt{\pi}},\qquad
\mathbb{E}_{g\sim\mathcal{N}(0,1)}[f(g)]
\approx\sum_i\widetilde{w}_i f(g_i).
```

Thus the returned Hermite weights sum to one and reproduce standard-normal
moments through degree $2n-1$.

Clenshaw-Curtis uses
$x_i=\cos(i\pi/(n-1))$ and cosine-series weights. Its exactness pattern differs
from Gaussian quadrature, so convergence must be tested rather than inferred
from [](#eq-gaussian-exactness).

## What the algorithm actually does

`gauss_legendre_nodes`, `gauss_laguerre_nodes`, and `gauss_hermite_nodes` call
NumPy polynomial factories on the host and freeze the one-dimensional results
as JAX arrays. Laguerre and Clenshaw-Curtis explicitly reject `n < 1`; invalid
Legendre and Hermite orders propagate their NumPy factory errors.

`clenshaw_curtis_nodes(1)` returns node zero with weight two. Larger rules return
nodes ordered from one to minus one. `hermite_e_basis(g, n_max)` uses the
probabilists' recurrence
$He_{n+1}(g)=gHe_n(g)-nHe_{n-1}(g)$ and returns shape
`(n_max + 1, q)`. `hermite_coefficients(map_fn, n_max, n_quad=256)` returns
$c_n=\mathbb{E}[map\_fn(g)He_n(g)]$ for $n=0,\ldots,n_{\max}$.

Node count and expansion order are concrete Python integers. No factory adapts
order, returns a runtime error estimate, or owns a stopping policy.

## What JAX differentiates

Node generation is host-side setup and is neither traced nor differentiated.
Once nodes and weights exist, JAX differentiates the weighted sum through
integrand values and any parameters closed over by the integrand.
`hermite_e_basis` is pure JAX arithmetic and differentiable with respect to its
input points. `hermite_coefficients` carries derivatives through `map_fn` values,
not through `n_quad` or node construction.

:::{warning}
Fixed nodes do not make a nonsmooth or singular integrand smooth. A finite AD
result is the derivative of the fixed weighted sum, which approximates the
derivative of the integral only when differentiation and integration may be
interchanged and quadrature error is controlled for that derivative integrand.
:::

## Using it in Jaxstro

```python
import jax.numpy as jnp

from jaxstro import quad

legendre_x, legendre_w = quad.gauss_legendre_nodes(8)
poly_integral = jnp.sum(legendre_w * legendre_x**6)

normal_x, normal_w = quad.gauss_hermite_nodes(8)
normal_second_moment = jnp.sum(normal_w * normal_x**2)

assert jnp.allclose(poly_integral, 2.0 / 7.0)
assert jnp.allclose(jnp.sum(normal_w), 1.0)
assert jnp.allclose(normal_second_moment, 1.0)
```

The returned node and weight arrays both have shape `(n,)`. Their default dtype
follows the active JAX precision policy when NumPy results are converted.

## How to audit the result

For Gaussian rules, verify every moment from degree zero through $2n-1$ against
an analytic value and confirm the first degree beyond the guarantee is not
silently described as exact. Check weight sums: two for Legendre and Clenshaw-
Curtis on $[-1,1]$, one for Laguerre's $e^{-x}$ measure, and one for the normalized
Hermite rule. For a non-polynomial integrand, compare increasing node counts and
an independent high-accuracy reference. Audit parameter gradients against
central finite differences while holding nodes fixed.

Executable evidence is indexed in [](../../60-validation/validation.md).

## Where the claim stops

These factories do not detect singularities, transform arbitrary domains,
estimate error, choose order, or adapt nodes. Polynomial moment tests validate
the rule convention and implementation; they do not establish convergence for
a new integrand. Hermite expansion coefficients do not by themselves establish
that a truncated expansion is scientifically adequate.

## Connected ideas

:::{seealso}
Connect weighted integrals to
[](../../10-foundations/mathematical-objects/probability-and-distributions.md),
units to [](../../30-representations/units-quantities/quantities.md), owner
signatures to [](../../50-api/approximation-integration/quad.md), and evidence
to [](../../60-validation/validation.md). The
[legacy fixed-quadrature page](../../50-api/approximation-integration/quadrature.md)
records the temporary import-name mapping. Sampled Newton-Cotes rules are in
[](./cumulative-trapz.md); delegated adaptive methods are described in
[](./adaptive-quadrature.md).
:::
