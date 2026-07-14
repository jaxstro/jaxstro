---
title: Root-finding
description: >-
  Sign brackets, safeguarded scalar solves, fixed executed maps, and certified
  implicit derivatives with explicit evidence boundaries.
---

## The question this method answers

Given a scalar relation $f(x;\theta)=0$, what value of $x$ satisfies it, and
which derivative of that answer is scientifically meaningful? A robust root
value, the sensitivity of a finite executed solver, and the sensitivity of a
unique smooth mathematical root are different objects.

The derivative distinction begins in
[](../../10-foundations/mathematical-objects/what-is-a-derivative.md). Parameter
and state representations are connected in
[](../../30-representations/parameters-state/parameters-and-transforms.md).

:::{tip}
Choose the derivative target before the solver. Use a bracketed method for
reliable values, Newton for the sensitivity of its smooth finite executed iteration,
and `implicit_bracketed_root` only for a certified mathematical-root sensitivity.
:::

## Before computation: what should be true?

Define the scalar variable, residual units, parameters, expected root branch,
and admissible interval. For a sign-bracketed solve, $f$ must be continuous on
the interval and its endpoints must have opposite signs or contain an exact
root. For an implicit derivative, the selected root must also be unique and
smooth in the parameter, with a finite nonzero local slope $\partial f/\partial x$.

:::{important}
A sign change certifies an odd number of crossings for a continuous residual;
it does not by itself certify uniqueness. The caller owns continuity, branch
meaning, units, and admissibility. Jaxstro can preserve endpoint evidence and
check numerical certificate gates, but it cannot infer those scientific facts.
:::

Pick tolerances in root-coordinate units. `atol` has the units of $x$ and
`rtol` is dimensionless. Enable sufficient precision before requesting a
tolerance near float64 limits.

## Define the mathematical objects

A root is $x^\star$ such that $f(x^\star;\theta)=0$. A bracket at iteration $k$
is an ordered interval $[a_k,b_k]$ with evaluated endpoint residuals. The best
endpoint is the one with smaller absolute residual. A proposal is a candidate
inside the bracket, produced by bisection, secant interpolation, or
inverse-quadratic interpolation (IQI).

A fixed trace stores the proposal, its residual, updated endpoint evidence,
proposal kind, execution mask, admissibility, convergence state, and terminal
status for every allocated scan slot. Unused floating entries are NaN and
unused mask entries are false; fixed shape is an execution contract, not a
convergence claim.

For a parameterized root relation, define
$f_x=\partial f/\partial x$ and $f_\theta=\partial f/\partial\theta$ on the
selected smooth branch. These local derivatives belong to the mathematical
relation, not to the branch history of a numerical solver.

## Derive the method

### Bracket preservation and bisection

A verified bracket preserves the opposite-sign endpoint invariant

```{math}
:label: eq-root-bracket-invariant
a_k\le b_k,\qquad f(a_k)\,f(b_k)\le 0.
```

Sign bits or exact endpoint zeros are safer in floating point than multiplying
large or tiny residuals. Bisection evaluates
$m_k=a_k+(b_k-a_k)/2$ and replaces the endpoint that has the same sign as
$f(m_k)$. Therefore $b_{k+1}-a_{k+1}=(b_k-a_k)/2$. After $N$ steps,

```{math}
b_N-a_N=2^{-N}(b_0-a_0).
```

This is a value-error certificate for a continuous sign-changing residual; it
does not create a useful derivative with respect to parameters inside $f$.

### Safeguarded interpolation

Secant interpolation uses the line through the endpoint pairs. IQI fits the
inverse relation $x(f)$ through three distinct residual points. The current
proposal code tries inverse-quadratic interpolation when three distinct
residual points exist; otherwise the endpoint secant is selected. The selected
interpolant must be finite, strictly inside the bracket, make sufficient
progress, and stay inside the safeguard band

```{math}
:label: eq-root-safeguard-band
a_k+\sigma(b_k-a_k)
\le x_{\mathrm{trial}}\le
b_k-\sigma(b_k-a_k),\qquad 0\le\sigma<\frac12.
```

The safeguard band is inclusive; a proposal on either band edge is admissible
when it also satisfies the separate finite, strict-interior, and progress
checks. A rejected selected interpolant falls back to the overflow-safe
midpoint. The bracket update then restores [](#eq-root-bracket-invariant).

### Newton and the implicit derivative

Newton linearizes the residual about $x_k$:
$f(x_k+\Delta x)\approx f(x_k)+f'(x_k)\Delta x$. Setting the approximation to
zero gives

```{math}
x_{k+1}=x_k-\frac{f(x_k)}{f'(x_k)}.
```

For a unique smooth mathematical root, differentiate
$f(x^\star(\theta);\theta)=0$:

```{math}
:label: eq-root-implicit-derivative
\frac{d x^\star}{d\theta}
=-
\frac{\partial f/\partial\theta}
{\partial f/\partial x}
\bigg|_{x=x^\star}.
```

This implicit function theorem (IFT) result requires a nonzero denominator and
the stated uniqueness and smooth-branch assumptions. It is not obtained by
differentiating bisection decisions.

(newton-ppf)=
### Inverse CDFs

For a cumulative distribution $F$, the quantile $x=F^{-1}(u)$ solves
$F(x)-u=0$ and $F'(x)=\operatorname{pdf}(x)$. Newton therefore becomes

```{math}
:label: eq-ppf-step
x_{k+1}=x_k-\frac{F(x_k)-u}{\operatorname{pdf}(x_k)}.
```

This finite inverse-CDF construction is not a generic implicit-root
certificate.

## What the algorithm actually does

`bracket_expand(f, x0, step=1, growth=2, max_steps=32)` searches symmetric
intervals with a fixed scan and returns `(lo, hi, found)`. If discovery fails,
the returned endpoints are the last expanded interval and `found=False`.
`bisect(..., max_steps=50)` performs a caller-selected fixed number of
halvings, with default `max_steps=50`; `bisect_many` is the explicit
array-shaped wrapper for independent brackets.

`initialize_bracket` constructs true endpoint evidence from already evaluated
residuals. `update_bracket(..., valid=False)` leaves every field unchanged.
`BracketedRootState` pairs `BracketState` with interpolation-only
`BracketHistory`; `propose_bracketed` proposes without evaluating, and
`advance_bracketed_root` consumes one externally supplied evaluation.

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
  - Endpoint secant accepted
* - `PROPOSAL_MIDPOINT`
  - `2`
  - Selected interpolant rejected; deterministic midpoint used
* - `PROPOSAL_LO_ENDPOINT`
  - `3`
  - Exact lower-endpoint root
* - `PROPOSAL_HI_ENDPOINT`
  - `4`
  - Exact upper-endpoint root
* - `PROPOSAL_INVERSE_QUADRATIC`
  - `5`
  - Three-point IQI accepted
```

`safeguarded_bracketed_root` evaluates both endpoints and runs `max_steps`
fixed `lax.scan` slots. A scalar `lax.cond` prevents residual evaluation after
convergence or terminal failure. Convergence is an exact residual or a full
bracket width no larger than
$\mathrm{atol}+\mathrm{rtol}|x_{\mathrm{best}}|$. Exhaustion returns the
evaluated endpoint with smaller absolute residual and `converged=False`.
A missing bracket returns NaN root and residual with `bracketed=False`; a
nonfinite interior evaluation terminates that lane.

`RootTrace` and `BracketedRootResult` retain the fixed trace, status, final
bracket, initial residual scale, and function-evaluation count. The no-extra-
evaluation guarantee is scalar. `vmap` preserves values and shapes but can
lower `lax.cond` to select-style execution; `map_safeguarded_bracketed_root`
owns an explicit `lax.map` boundary when physical per-lane skipping matters.

## What JAX differentiates

`bisect`, bracket discovery, and safeguarded proposals select intervals through
sign and acceptance predicates. Their parameter gradients are branch-selected
finite-program artifacts, not root sensitivities; bisection is structurally
zero with respect to parameters captured only inside $f$.

`newton` and `newton_with_grad` use smooth iterates, finite-map gradients.
Their caller-selected fixed update count has default `max_steps=30`. JAX can
expose a finite executed-map sensitivity through those updates, but that is not
automatically [](#eq-root-implicit-derivative). The zero-derivative operand is
replaced by one before division to avoid dead-branch poisoning; this guard
preserves finiteness, not convergence.

`implicit_bracketed_root(f, args, ...)` is separate. It uses `lax.custom_root`
and exposes a certified mathematical-root sensitivity only after caller
assertions of uniqueness and a smooth branch plus convergence, finiteness,
residual, bracket-width, and slope-conditioning gates. Rejection returns NaN
for the derivative-facing value and attempted gradient while retaining the
nested primal diagnostics.

`newton_ppf` adds `pdf_floor` to its denominator and clips every iterate to
`[lo, hi]`. Interior iterates can carry finite executed-map sensitivity with
respect to $u$ and parameters in the CDF. At a clipped support boundary the
gradient saturates to zero. `monotone_inverse_interp` is linear inside table
cells and clamps outside the tabulated range.

:::{warning}
Do not infer an implicit derivative from a finite gradient. The value-first
hybrid makes no implicit-root derivative claim. Only the fail-closed implicit
API claims [](#eq-root-implicit-derivative), and only when every certificate
gate passes.
:::

## Using it in Jaxstro

Use the actual owner path and inspect the typed result rather than extracting a
number without its status:

```python
from jaxstro.jaxconfig import enable_high_precision

enable_high_precision()  # before creating JAX arrays

from jaxstro.numerics.rootfinding import safeguarded_bracketed_root


result = safeguarded_bracketed_root(
    lambda x: x**2 - 2.0,
    0.0,
    2.0,
    max_steps=64,
    atol=1.0e-12,
    rtol=1.0e-12,
    safeguard_fraction=0.1,
)
assert result.bracketed
assert result.converged
```

The safeguarded interface is scalar and returns fixed-length trace arrays.
Batched callers map scalar solves explicitly. The function and iteration count
are static when wrapped in `jax.jit`.

```{list-table} Choosing a one-dimensional solver
:header-rows: 1
:label: tbl-solver-choice

* - Need
  - Use
  - Derivative meaning
* - Discover a sign change
  - `bracket_expand`
  - Value evidence only
* - Simple robust bracketed value
  - `bisect` or `bisect_many`
  - No parameter-root derivative claim
* - Auditable expensive bracketed value
  - `safeguarded_bracketed_root`
  - Value-first branch-selected map
* - Downstream-owned trial evaluation
  - bracket primitives
  - Value-first checkpoint surface
* - Smooth residual and good guess
  - `newton` or `newton_with_grad`
  - Finite executed-map sensitivity
* - Unique smooth certified root
  - `implicit_bracketed_root`
  - Certified mathematical-root sensitivity
* - Smooth inverse CDF
  - `newton_ppf`
  - Finite executed-map sensitivity
* - Monotone inverse table
  - `monotone_inverse_interp`
  - Cell-local query sensitivity
```

## How to audit the result

### Predict -> compute -> audit: which derivative are you asking for?

**Predict.** State whether the target is a root value, the finite executed
algorithm, or a unique smooth mathematical root. Derive the expected sensitivity.

**Compute.** Retain signed endpoint residuals, trace, status, final bracket,
local slope, and every certificate predicate.

**Audit.** Check the opposite-sign endpoint invariant is checked at every
executed update. Compare the certified AD derivative with the analytic result
and an independently recomputed central finite difference. Reject the claim if
uniqueness, smoothness, residual, width, finiteness, or conditioning fails.

:::{figure} ../../10-theory/figures/rootfinding-safeguards.webp
:name: fig-rootfinding-safeguards
:alt: Two-panel safeguarded root trace showing circle IQI, square secant, and triangle midpoint proposals on a quadratic residual and solid lower endpoint, dashed upper endpoint, and dotted bracket width across executed iterations

The public trace supplies every plotted point. The figure demonstrates interval
telemetry and the opposite-sign endpoint invariant; it does not claim universal
speed.
:::

The reproducible evaluation-count report is
[](../../60-validation/numerical/rootfinding-performance.md). Its primary cost
is function-evaluation count, not a hardware-dependent timing threshold.
[](#fig-rootfinding-safeguards) shows the public trace quantities behind that
value-first audit.

:::{figure} ../../10-theory/figures/rootfinding-value-versus-ift.webp
:name: fig-rootfinding-value-versus-ift
:alt: Two-panel comparison of a branch-selected quadratic root trace with analytic, certified implicit-function AD, and central finite-difference sensitivities; certification includes uniqueness and smoothness assertions plus convergence, finiteness, residual, width, and slope gates, while a flat-root certificate is rejected

The left panel asks what numerical map executed. The right asks how the
certified root relation moves with its parameter. The flat-root case is a
rejection, not an estimate.
:::

[](#fig-rootfinding-value-versus-ift) keeps the executed-map and certified-root
derivative questions visually separate.

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

The machine-readable envelope and human report are generated and freshness-
checked together; see [](../../60-validation/numerical/implicit-root-gradients.md).

## Where the claim stops

A valid bracket does not prove uniqueness. A small residual alone can hide an
ill-conditioned root. A narrow bracket certifies coordinate location only under
the continuity and sign-change assumptions. Successful JIT or VMAP execution
does not add derivative meaning, and a warm benchmark does not establish
performance for a different residual cost or batch structure.

The exponential inverse-CDF check validates one smooth interior case. It does
not certify all distributions, clipped quantiles, traced invalid tables, or
model-specific branch semantics.

## Connected ideas

:::{seealso}
Connect roots to conditioning in
[](../../10-foundations/models-and-computation/sensitivity-conditioning-identifiability.md),
parameters to [](../../30-representations/parameters-state/parameters-and-transforms.md),
exact signatures and statuses to [](../../50-api/change-constraints/rootfinding.md),
and executable evidence to [](../../60-validation/validation.md). The generic
fixed-iteration principles are in [](../methods.md#p2-fixed-iteration).
:::
