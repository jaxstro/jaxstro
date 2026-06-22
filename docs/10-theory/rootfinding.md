---
title: Root-finding
description: >-
  Why jaxstro's solvers run a fixed number of lax.scan steps, how bracketing
  and inverse interpolation behave under AD, and when to reach for newton or
  newton_ppf instead.
---

You have a scalar equation $f(x) = 0$ and you want the root — and you want to
differentiate that root with respect to whatever parameters $f$ closes over. This
page explains the solvers jaxstro ships, what each one's gradient actually does,
and the one trap that catches everyone.

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

## `bracket_expand` — fixed-count sign discovery

`bracket_expand(f, x0, step=1, growth=2, max_steps=32)` expands a symmetric
interval around an initial point:

```{math}
[x_0 - s g^k,\; x_0 + s g^k],
```

for a fixed number of scan steps. It returns `(lo, hi, found)`, where `found` is
a boolean mask indicating whether a sign-changing bracket was found. If no
bracket is found, the endpoints are the last expanded interval. That makes the
failure mode explicit and transform-friendly: callers can decide whether to
refine, mask, or fail closed.

This is a **forward bracketing utility**, not a differentiable solver. The
selection of the first valid bracket depends on sign predicates, so treat the
returned bracket as value evidence, not as a smooth function of model
parameters.

## `bisect` and `bisect_many` — robust values, branchy gradients

`bisect(f, a, b)` assumes a sign change in $[a, b]$ and halves it 50 times. It is
the robust choice: it cannot diverge after a valid bracket, and it needs no
derivative of $f$. It selects the half containing the root branchlessly with
`jnp.where`, so the forward pass is clean. `bisect_many(...)` is the explicit
array-shaped wrapper for independent brackets; it exists to make vectorized use
readable when each element carries its own bracket.

But here is the trap. **The gradient of the bisection result with respect to the
function parameters is structurally zero.** The *comparison* that decides which
half to keep — `sign(f_a) * sign(f_m) <= 0` — is a non-differentiable predicate
(principle [6](./index.md#p6-non-diff-ops)). Parameters captured inside `f` only
affect those sign decisions, so $\partial x^\star/\partial\theta$ comes back as
zero even though the true root plainly depends on $\theta$. The value is right;
that gradient is a lie.

The bracket endpoints enter the arithmetic midpoints directly, so a truncated
fixed-count bisection can have endpoint sensitivities. Those are implementation
sensitivities of the finite iteration, not a scientifically meaningful implicit
root derivative.

:::{warning} Do not differentiate a bisection root w.r.t. function parameters
If you need $\partial x^\star/\partial\theta$ for a parameter $\theta$ inside $f$,
`bisect` will hand you zeros, not an error. Use `newton` or `newton_ppf` instead —
their iterates are smooth functions of the residual and its derivative, so the
gradient flows. Reserve `bracket_expand` and `bisect` for forward solve
reliability, initialization, and validation masks.
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

## `monotone_inverse_interp` — inverse lookup for table-defined CDFs

`monotone_inverse_interp(x, y, y_new)` is the table-first inverse path. It assumes
`x` and `y` are one-dimensional, same-length, strictly increasing arrays, then
interpolates the inverse table $x(y)$ linearly. Queries below `y[0]` clamp to
`x[0]`; queries above `y[-1]` clamp to `x[-1]`.

This is useful for CDF-like tables when the distribution is known only on a grid
or when the caller wants a deterministic, inexpensive inverse before a smoother
solver is justified. Inside the tabulated domain, gradients with respect to
`y_new` are the reciprocal local slope. At clamped endpoints the gradient
saturates to zero, matching the library's broader clamp policy.

The function validates concrete tables eagerly. Under `jit`, value-dependent
validation cannot raise on tracers, so production callers should build tables
once and keep the monotonicity contract explicit.

## Brent-like solvers: deferred until the AD policy is honest

Hybrid Brent-style methods mix interpolation, bisection, and convergence-driven
branching. They are excellent forward solvers, but their branch history is not a
smooth mathematical map. jaxstro does not ship a Brent wrapper in this slice. If
one is added later, it should be documented as value-first unless it carries a
specific implicit-differentiation or custom-VJP policy.

## What to reach for

```{list-table} Choosing a 1-D solver
:header-rows: 1
:label: tbl-solver-choice

* - You have…
  - Use
  - Differentiable w.r.t. parameters?
* - a point near an unknown sign-changing root
  - `bracket_expand`
  - No — sign-discovery utility
* - a sign-bracketed root, robustness matters
  - `bisect`
  - No w.r.t. function parameters
* - many independent sign-bracketed roots
  - `bisect_many`
  - No w.r.t. function parameters
* - a smooth $f$ and a good guess
  - `newton`
  - Yes
* - a smooth $f$ and a cheap analytic $f'$
  - `newton_with_grad`
  - Yes
* - an inverse-CDF / quantile to sample
  - `newton_ppf`
  - Yes (w.r.t. $u$ and CDF parameters)
* - a monotone table that should be inverted by lookup
  - `monotone_inverse_interp`
  - Yes inside table cells w.r.t. query values
```

Signatures and defaults are in [](../40-api/index.md); the design rationale for
hoisting the generic Newton-PPF into the foundation is
[](../30-decisions/0009-jaxstro-params-selective-inference.md).
