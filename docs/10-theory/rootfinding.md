---
title: Root-finding
description: >-
  Why jaxstro's solvers run a fixed number of lax.scan steps, why bisect's
  gradient with respect to bracket endpoints is structurally zero, and when to
  reach for newton or newton_ppf instead.
---

You have a scalar equation $f(x) = 0$ and you want the root — and you want to
differentiate that root with respect to whatever parameters $f$ closes over. This
page explains the three solvers jaxstro ships, what each one's gradient actually
does, and the one trap that catches everyone.

This is principle [2](./index.md#p2-fixed-iteration) made concrete: a solver that
loops "until converged" cannot be cleanly differentiated, so every solver here
runs a **fixed** number of steps under `lax.scan`.

## `lax.scan` with a fixed count, never `while_loop`

A convergence loop asks "are we close enough yet?" and stops when the answer is
yes. The trip count then depends on the input values, which JAX cannot trace
through for differentiation, and `lax.while_loop` is not reverse-mode
differentiable at all. The alternative is to pick a step count that
**over-converges** and run exactly that many `lax.scan` iterations every time.

You pay for the wasted iterations and you buy a computation that is `jit`-, `vmap`-,
and `grad`-safe. The numbers are forgiving. Bisection halves the bracket each step,
so 50 steps reach $2^{-50} \approx 8.9\times10^{-16}$ — full float64 precision.
Newton converges quadratically near a smooth root, so 20–30 steps over-converge
from a reasonable guess.

## `bisect` — robust value, structurally zero gradient w.r.t. the bracket

`bisect(f, a, b)` brackets a sign change in $[a, b]$ and halves it 50 times. It is
the robust choice: it cannot diverge, and it needs no derivative of $f$. It selects
the half containing the root branchlessly with `jnp.where`, so the forward pass is
clean.

But here is the trap. **The gradient of the bisection result with respect to the
bracket endpoints `a` and `b` is structurally zero.** Each step replaces an
endpoint with the midpoint $\tfrac12(a+b)$, and the *comparison* that decides which
half to keep — `sign(f_a) * sign(f_m) <= 0` — is a non-differentiable predicate
(principle [6](./index.md#p6-non-diff-ops)). The control flow that depends on `a`
and `b` carries no derivative, so $\partial x^\star/\partial a$ and
$\partial x^\star/\partial b$ come back as zero even though the true root plainly
depends on the bracket. The value is right; the gradient is a lie.

:::{warning} Do not differentiate a bisection root w.r.t. its bracket or its parameters
If you need $\partial x^\star/\partial\theta$ for a parameter $\theta$ inside $f$,
`bisect` will hand you zeros, not an error. Use `newton` or `newton_ppf` instead —
their iterates are smooth functions of the inputs, so the gradient flows. Reserve
`bisect` for the forward solve, or for a robust *bracketing* step whose output you
do not differentiate.
:::

## `newton` and `newton_with_grad` — smooth iterates, real gradients

`newton(f, x0)` runs the update $x_{k+1} = x_k - f(x_k)/f'(x_k)$ for a fixed 30
steps, taking $f'$ from `jax.grad(f)` automatically. Because every step is a smooth
arithmetic function of $x_k$ and of any parameters captured in $f$, the gradient
flows through the whole iteration — Newton's root *is* differentiable with respect
to those parameters, exactly where `bisect` fails.

The one hazard is division by a zero derivative. jaxstro guards the **operand**,
not the result (principle [3](./index.md#p3-guard-singularities)): it replaces a
zero $f'$ with $1$ before dividing, so no `inf`/`NaN` enters the backward pass.
Supply an analytic derivative with `newton_with_grad(f, df, x0)` when autodiff of
$f$ is expensive or $f$ is not directly differentiable.

The price of that smoothness is robustness: Newton needs a decent starting guess
and a non-vanishing derivative along the path. Bracket first with knowledge of the
problem, then refine.

(newton-ppf)=
## `newton_ppf` — the inverse-CDF solver, and the clip-to-support caveat

Reparameterized sampling needs the percent-point function (the inverse CDF): given
a uniform draw $u \in (0,1)$, return the quantile $x = F^{-1}(u)$ solving
$F(x) = u$. `newton_ppf` applies Newton to the residual $g(x) = F(x) - u$, whose
derivative is exactly the density, $g'(x) = F'(x) = \mathrm{pdf}(x)$:

```{math}
:label: eq-ppf-step
x_{k+1} = x_k - \frac{F(x_k) - u}{\mathrm{pdf}(x_k)}.
```

It is deliberately distribution-agnostic — you pass your own `cdf` (closing over the
distribution's parameters), an initial guess, and the support bounds `[lo, hi]`. The
PDF comes from your `pdf` argument or, failing that, from `jax.grad(cdf)`. The
result is differentiable **both** with respect to $u$ **and** with respect to the
parameters inside `cdf`, which is what makes it usable inside an inference loop.

Two guards deserve a note, and they connect back to the principles:

- **A density floor, not a branch.** The denominator carries an additive
  `pdf_floor` (default $10^{-30}$) so a near-zero density in a flat-CDF region
  cannot produce a `NaN`. This is the safe-operand pattern again
  ([principle 3](./index.md#p3-guard-singularities)): additive, never a `where` on
  the result.
- **Clipping to the support saturates the gradient there.** Each iterate is clipped
  to `[lo, hi]` so it never leaves the support — but a quantile pinned at a bound
  has zero gradient with respect to your parameters at that bound
  ([principle 4](./index.md#p4-saturation)). For interior quantiles this never
  fires; if you are fitting parameters that push a quantile against `lo` or `hi`,
  that is where the gradient quietly dies.

The PPF is validated against the analytic exponential inverse,
$F^{-1}(u) = -\ln(1-u)/\lambda$, by an FD-vs-AD grad-check — value and gradient both
([](../60-validation/index.md)).

## What to reach for

```{list-table} Choosing a 1-D solver
:header-rows: 1
:label: tbl-solver-choice

* - You have…
  - Use
  - Differentiable w.r.t. parameters?
* - a sign-bracketed root, robustness matters
  - `bisect`
  - **No** — structurally zero gradient
* - a smooth $f$ and a good guess
  - `newton`
  - Yes
* - a smooth $f$ and a cheap analytic $f'$
  - `newton_with_grad`
  - Yes
* - an inverse-CDF / quantile to sample
  - `newton_ppf`
  - Yes (w.r.t. $u$ and CDF parameters)
```

Signatures and defaults are in [](../40-api/index.md); the design rationale for
hoisting the generic Newton-PPF into the foundation is
[](../30-decisions/0009-jaxstro-params-selective-inference.md).
