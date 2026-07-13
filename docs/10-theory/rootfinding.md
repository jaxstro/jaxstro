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

## Learning objectives

After this chapter, you should be able to distinguish a sign-bracket guarantee
from a plausible root estimate, interpret fixed-shape solver telemetry, and
decide when a branch-selected value map or certified implicit derivative answers
the scientific question.

## `lax.scan` with a fixed count, never `while_loop`

A convergence loop asks "are we close enough yet?" and stops when the answer is
yes. The trip count then depends on the input values, which JAX cannot trace
through for differentiation, and `lax.while_loop` is not reverse-mode
differentiable at all. The alternative is to pick a static maximum and run
exactly that many `lax.scan` slots. A solver may still mask inactive slots with
`lax.cond`, so a fixed trace shape does not require repeated expensive function
evaluations after convergence.

You pay for the static trace and buy a computation that composes with `jit` and
`vmap`. That transformability is not itself a gradient claim; each solver's AD
contract below remains authoritative. The numbers are forgiving. Bisection halves the bracket each step,
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
`bisect` can hand you zeros, not an error. Use `newton` only when you want the
sensitivity of its smooth finite executed iteration. Use
`implicit_bracketed_root` when the scientific target is the derivative of a
unique smooth mathematical root and its validation gates pass. `newton_ppf` is
a specialized inverse-CDF construction, not a generic implicit-root
certificate. Reserve `bracket_expand` and `bisect` for forward solve
reliability, initialization, and validation masks.
:::

## Safeguarded bracket primitives — the downstream integration surface

`initialize_bracket(lo, hi, f_lo, f_hi)` constructs a `BracketState` from
already-evaluated endpoint evidence. A bracket is verified only when its
endpoints and residuals are finite, `lo <= hi`, and the residuals have opposite
sign bits or an endpoint is exactly zero. Sign products are deliberately
avoided because they can overflow or underflow even for finite residuals.

`update_bracket(state, x, fx, valid=True)` replaces exactly one endpoint when
`x` is finite and inside the verified bracket and `fx` is finite. An exact
interior root collapses both endpoints to `x`. `valid=False` returns every field
unchanged, so a downstream solver can exclude an externally inadmissible trial
without corrupting root evidence or moving its own expensive trial state into
Jaxstro.

`BracketedRootState` pairs true `BracketState` endpoint evidence with
`BracketHistory` used only for interpolation. `propose_bracketed` tries
inverse-quadratic interpolation when three distinct residual points exist;
otherwise the endpoint secant is selected. The selected interpolant must pass
finite-denominator, strict-bracket, safeguard-band, and previous-step progress
guards. A rejected selected interpolant uses the overflow-safe midpoint.
`advance_bracketed_root` consumes one evaluation; `valid=False` preserves the
bracket, history, and status exactly.

```{list-table} Proposal-kind telemetry
:header-rows: 1

* - Identifier
  - Value
  - Meaning
* - `PROPOSAL_NONE`
  - `0`
  - Masked slot or missing bracket
* - `PROPOSAL_SECANT`
  - `1`
  - Secant interpolation passed every guard; `safeguarded=False`
* - `PROPOSAL_MIDPOINT`
  - `2`
  - Selected interpolant rejected; deterministic midpoint used and `safeguarded=True`
* - `PROPOSAL_LO_ENDPOINT`
  - `3`
  - Exact root at the lower endpoint
* - `PROPOSAL_HI_ENDPOINT`
  - `4`
  - Exact root at the upper endpoint
* - `PROPOSAL_INVERSE_QUADRATIC`
  - `5`
  - Three-point inverse-quadratic interpolation passed every guard
```

## `safeguarded_bracketed_root` — fixed trace, guarded evaluations

The high-level wrapper evaluates both endpoints, verifies the bracket, and runs
a fixed-length scan. Each active slot evaluates `f` inside `lax.cond`; missing-
bracket, converged, and later masked slots do not execute `f`. Convergence means
an exact residual or a full bracket width no larger than

```{math}
\mathrm{atol} + \mathrm{rtol}\,|x_{\mathrm{best}}|.
```

This is a root-coordinate tolerance: `atol` has the units of `x`, and `rtol` is
dimensionless. The returned point is always an evaluated endpoint in the final
verified bracket, so the full-width test certifies its coordinate error for a
continuous sign-changing residual. Exhaustion returns the endpoint with smaller
absolute residual but keeps `converged=False`. A nonfinite interior evaluation is recorded as executed but
inadmissible and exhausts that lane immediately, avoiding repeated expensive
calls at the same proposal. A missing bracket returns `root=NaN`,
`residual=NaN`, and `bracketed=False`; it never fabricates a root.

`RootTrace` records fixed-length arrays for proposal and signed residual, the
post-update `lo`, `hi`, `f_lo`, and `f_hi`, proposal kind, and `executed`,
`admissible`, per-slot `converged`, and terminal `status`. Unused floating slots are NaN,
unused kinds are `PROPOSAL_NONE`, and unused masks are false.
`BracketedRootResult` also returns terminal status, initial residual scale, the
final true bracket, function-evaluation count, and trace.

The no-extra-evaluation guarantee and `n_evaluations` count describe scalar
execution. `vmap` preserves values and fixed shapes, but JAX may batch scalar
`lax.cond` into select-style execution that computes both branches. When a batch
contains expensive residuals and physical per-lane skipping matters, use
`map_safeguarded_bracketed_root`, which owns an explicit `lax.map` boundary.

```python
from jaxstro.numerics import safeguarded_bracketed_root

result = safeguarded_bracketed_root(
    lambda x: x**2 - 2.0,
    0.0,
    2.0,
    max_steps=64,
    atol=1.0e-12,
    rtol=1.0e-12,
    safeguard_fraction=0.1,
)
assert result.converged
```

:::{figure} ./figures/rootfinding-safeguards.webp
:name: fig-rootfinding-safeguards
:alt: Two-panel safeguarded root trace showing circle IQI, square secant, and triangle midpoint proposals on a quadratic residual and solid lower endpoint, dashed upper endpoint, and dotted bracket width across executed iterations

Both panels are computed from the public solver and its fixed-shape trace. The
left panel shows which selected proposals were evaluated; the right shows the
stored interval endpoints and width contracting. The opposite-sign endpoint
invariant is checked from the traced endpoint residuals in the figure test, but
is not encoded as another curve. This fixture demonstrates auditable interval
telemetry, not universal speed for every residual.
:::

The reproducible evaluation-count and warm-timing comparison with fixed-count
bisection is available as a
[human-readable evidence report](../validation/rootfinding-performance.md) and
[machine-readable envelope](../validation/rootfinding-performance.json). The
benchmark treats function-evaluation count as the primary algorithmic cost and
does not impose a hardware-dependent wall-time threshold. The ratified gate
requires the hybrid not to exceed bisection evaluations on any of five cases and
to reduce evaluations by at least 25% on three.

:::{warning} Value-first means no implicit-root derivative claim
The executed IQI/secant/midpoint branch history is a numerical map, not an
implicit-root derivative. No gradient claim is made for parameters captured by
`f`, even if `jax.grad` returns a finite number for a particular execution. This
API does not use `lax.custom_root` and does not implement an IFT derivative.

Use the separate `implicit_bracketed_root(f, args, ...)` only when an
implicit function theorem (IFT)
derivative is required. It uses `lax.custom_root` and requires caller assertions
of a unique root and smooth selected branch plus runtime gates for primal
convergence, finite evidence, residual, final-bracket width, and slope
conditioning. Rejection returns NaN for both the derivative-facing value and an
attempted gradient while retaining the nested primal diagnostics.
:::

### Predict → compute → audit: which derivative are you asking for?

**Predict.** Decide whether the desired quantity is the sensitivity of the
finite executed algorithm or of a unique, smooth mathematical root. Write down
the expected derivative and the conditions under which it exists.

**Compute.** Run the value solver and, separately, the certified implicit API.
Keep the branch trace, signed residual, final bracket, local slope, and every
certificate predicate.

**Audit.** Compare AD, the analytic derivative, and an independent central
finite difference. Reject the claim when uniqueness, smoothness, residual,
width, finiteness, or conditioning evidence fails.

:::{figure} ./figures/rootfinding-value-versus-ift.webp
:name: fig-rootfinding-value-versus-ift
:alt: Two-panel comparison of a branch-selected quadratic root trace with analytic, certified implicit-function AD, and central finite-difference sensitivities; certification includes uniqueness and smoothness assertions plus convergence, finiteness, residual, width, and slope gates, while a flat-root certificate is rejected

The value-first trace answers “what numerical map executed?” The derivative
panel answers the different question “how does the certified root relation move
with its parameter?” The flat-root annotation is deliberately shown as a
rejection, not as a derivative estimate.
:::

```{list-table} Measured quadratic implicit-root evidence
:header-rows: 1

* - Metric identity
  - Symbol
  - Value
  - Units
* - Certified root
  - $x^\star$
  - `1.414213562373095`
  - coordinate units
* - Final absolute residual
  - $|G(x^\star)|$
  - `4.440892098500626e-16`
  - function units
* - Final bracket width
  - $\Delta x$
  - `1.7763568394002505e-14`
  - coordinate units
* - Implicit-function AD sensitivity
  - $dx^\star/d\theta|_{\mathrm{AD}}$
  - `0.3535533905932738`
  - coordinate units per parameter unit
* - Central-FD root sensitivity
  - $dx^\star/d\theta|_{\mathrm{FD}}$
  - `0.35355339059739416`
  - coordinate units per parameter unit
```

These values are copied from the
[machine-readable envelope](../validation/implicit-root-gradients.json), with a
[human-readable evidence report](../validation/implicit-root-gradients.md), both
generated and freshness-checked by `scripts/benchmark_implicit_root.py`. A
documentation test requires the table to match that artifact; the JSON and
numerical tests remain the primary executable evidence.

## `newton` and `newton_with_grad` — smooth iterates, finite-map gradients

`newton(f, x0)` runs the update $x_{k+1} = x_k - f(x_k)/f'(x_k)$ for a fixed 30
steps, taking $f'$ from `jax.grad(f)` automatically. Because every step is a smooth
arithmetic function of $x_k$ and of any parameters captured in $f$, AD can expose
the finite executed-map sensitivity with respect to those parameters. This is a
derivative of the fixed iteration that ran, not automatically the derivative of
the ideal mathematical root. Use `implicit_bracketed_root` when the scientific
target is a certified mathematical-root sensitivity.

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
finite executed iteration can carry sensitivities **both** with respect to $u$
**and** with respect to parameters inside `cdf`. This supports inference through
that executed numerical map; it is not a generic implicit-root certificate.

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
* - an expensive sign-bracketed root with auditable failure state
  - `safeguarded_bracketed_root`
  - No implicit-root derivative claim; value-first
* - a downstream solver that owns trial admissibility and expensive state
  - `initialize_bracket` + `propose_bracketed` + `update_bracket`
  - No implicit-root derivative claim; value-first
* - many independent sign-bracketed roots
  - `bisect_many`
  - No w.r.t. function parameters
* - a smooth $f$ and a good guess
  - `newton`
  - Finite executed-map sensitivity; not an implicit-root certificate
* - a smooth $f$ and a cheap analytic $f'$
  - `newton_with_grad`
  - Finite executed-map sensitivity; not an implicit-root certificate
* - an inverse-CDF / quantile to sample
  - `newton_ppf`
  - Finite executed-map sensitivity w.r.t. $u$ and CDF parameters
* - a monotone table that should be inverted by lookup
  - `monotone_inverse_interp`
  - Yes inside table cells w.r.t. query values
```

Signatures and defaults are in [](../40-api/index.md); the design rationale for
hoisting the generic Newton-PPF into the foundation is
[](../30-decisions/0009-jaxstro-params-selective-inference.md).
