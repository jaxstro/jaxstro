---
title: Fixed-node quadrature
description: >-
  Gaussian, Clenshaw-Curtis, and cumulative Simpson rules with explicit AD
  contracts for differentiable scientific integration.
---

Quadrature rules turn an integral into a weighted sum:

```{math}
\int f(x)\,\omega(x)\,dx \approx \sum_i w_i f(x_i).
```

jaxstro treats the nodes and weights as setup constants. They may be generated on
the host, but the differentiable part is the same JAX expression every time:
evaluate the integrand at fixed nodes and sum weighted values. This is why
gradients flow through `f(x_i)` and not through node construction.

## Gaussian Rules

The Gaussian rules are exact for polynomials through degree `2n - 1` when using
`n` nodes:

- `gauss_legendre_nodes(n)` integrates on `[-1, 1]` with unit weight.
- `gauss_laguerre_nodes(n)` integrates on `[0, \infty)` with weight `exp(-x)`.
- `gauss_hermite_nodes(n)` computes expectations under a standard normal
  density, using the probabilists' Hermite convention.

The nodes and weights are generated with NumPy's polynomial routines using the
classical Golub-Welsch construction, then frozen as JAX arrays. They are not
model parameters.

## Clenshaw-Curtis

`clenshaw_curtis_nodes(n)` places nodes at Chebyshev-Lobatto points:

```{math}
x_i = \cos\left(\frac{i\pi}{n-1}\right).
```

The weights come from the standard cosine-series construction. Clenshaw-Curtis is
often competitive with Gaussian quadrature for smooth functions and has useful
endpoint behavior because it includes `-1` and `1`.

## Cumulative Simpson

`cumulative_simpson(y, x=None, dx=1, axis=-1)` lives with the integration helpers
because it integrates sampled values on a uniform grid. Its shape contract is
explicit: it returns cumulative values only at Simpson panel endpoints. If the
input has `n` samples along the integration axis, `n` must be odd and the output
axis has length `(n + 1) // 2`.

That contract avoids pretending Simpson's three-point panel gives trustworthy
cumulative values at every individual sample. Use `cumulative_trapz` when every
sample needs a cumulative value.

## Differentiability

For fixed nodes, all quadrature outputs are linear combinations of integrand
values. For fixed sampled grids, Simpson and cumulative Simpson are linear
combinations of `y`. The validation suite checks finite-difference versus AD
gradients through those values.

Node generation and shape choices are intentionally static. Adaptive quadrature
and convergence-loop integration are deferred because their data-dependent
control flow needs a separate AD policy.
