---
title: Writing AD-safe scientific numerics
short_title: Theory
description: >-
  Ten principles for numerics that survive jax.grad — the design thesis behind
  every function in jaxstro, each bridging to the method page that exemplifies it.
---

A function can return the right number and still be wrong. If its gradient is
silently zero, or `NaN`, or quietly detached from the parameter you care about,
then the value "worked" and the science is broken — and nobody finds out until an
optimizer stalls or a Fisher matrix comes back singular. *It ran* is not *it is
correct*. Elegant nonsense is still nonsense.

jaxstro exists so that the bottom of the dependency graph never produces that kind
of failure. Every primitive here is built to a single constraint: **it must
survive `jax.grad`, and its gradient must be checked against finite differences.**
This is why the numerics layer is useful beyond astronomy: the discipline is not
about a single domain, but about making differentiable scientific methods
auditable. This page is the thesis — ten principles for writing numerics that
differentiate cleanly. Each principle ends with a bridge to the method page that
shows it at work, so you can read the idea and then read the code that obeys it.

:::{tip} Already fluent in differentiable programming?
Skip to the principle that bites you most often — most people's is
[](#p3-guard-singularities) (the `where`-trap) or [](#p4-saturation) (the silent
gradient killer) — or go straight to the method pages:
[](./rootfinding.md), [](./cumulative-trapz.md), [](./quadrature.md),
[](./interpolation.md), [](./regular-grid.md), and [](./bsplines.md).
The dense helper layer for small fits and covariance diagnostics is
[](./linear-algebra.md), objective helpers live in [](./optimization.md),
fixed-step ODE helpers live in [](./ode.md), and generic special-function kernels
live in [](./special-functions.md). Generic distribution kernels live in
[](./distributions.md), vector geometry lives in [](./geometry.md), and
matrix-free algebra helpers live in
[](./operators.md). Autodiff product helpers live in [](./autodiff.md). Grid
construction, conservative rebinning, and stratified uniforms are in
[](./grids.md), while explicit PRNG streams and resampling helpers are in
[](./random.md).
:::

(p1-differentiability)=
## 1. Differentiability is a design constraint, not an afterthought

Decide up front that every public primitive will be differentiated, then design
backward from that. This rules out whole categories of "convenient" code before
you write them. The contract is concrete: for every differentiable function we
compute both the autodiff gradient and a finite-difference estimate and require
them to agree. If they disagree, the function is not done — see
[](../60-validation/index.md) for how that audit is run.

(p2-fixed-iteration)=
## 2. Fixed iteration, not convergence loops

A `while_loop` that runs "until converged" has a data-dependent trip count, and
JAX cannot differentiate through that cleanly. The fix is to run a **fixed** number
of `lax.scan` steps chosen to over-converge for the problem at hand. You trade a
few wasted iterations for a bounded, fully differentiable computation. Bisection
at 50 steps reaches $2^{-50}\approx10^{-15}$; Newton at 20–30 steps over-converges
for smooth functions. This is the load-bearing choice behind the root-finders.

→ [](./rootfinding.md) — why `bisect`, `newton`, and `newton_ppf` use `lax.scan`.

(p3-guard-singularities)=
## 3. Guard singularities without killing the gradient — the `where`-trap

The natural way to avoid a division by zero is `jnp.where(d == 0, fallback, a/d)`.
It returns the right value and a *wrong* gradient. JAX evaluates **both branches**
to differentiate the select, so `a/d` is still computed at $d=0$, producing an
`inf`, and `inf * 0` in the backward pass becomes `NaN` that propagates through the
"safe" branch. The discipline is to guard the **operand**, not the result: make the
denominator safe *before* dividing, e.g. `a / (d + eps)` or a double-`where` that
sanitizes $d$ first. jaxstro's `safe_div` and `safe_log` are exactly these guards.

(p4-saturation)=
## 4. Saturation is a silent gradient killer

`clip`, `min`, `max`, and `floor` set the gradient to zero wherever they
saturate. Sometimes that is what you want (a hard bound). Often it is a bug: a
parameter pinned at a clip boundary receives no gradient and never moves, and the
optimizer reports "converged" while sitting on a wall. Know which case you are in.
When you clip iterates to a support (as `newton_ppf` does to `[lo, hi]`), confirm
the optimum is interior, or the gradient you wanted is gone.

→ [](./rootfinding.md#newton-ppf) discusses the clip-to-support trade-off.

(p5-floating-point)=
## 5. Floating point is part of the math

Catastrophic cancellation, overflow in `exp`, and underflow in `log` are not
edge cases — they are the common case in likelihood code. Work in the log domain,
use `log1p`/`expm1` near zero, and sum in the order that minimizes error. jaxstro
provides `stable_log1p`, `stable_expm1`, `safe_log`, `safe_exp`, and Neumaier
compensated summation for reductions where the ordinary `sum` loses digits. And
turn on float64 first ([](#p8-precision)).

(p6-non-diff-ops)=
## 6. Non-differentiable operations are forbidden in the differentiable graph

`argmax`, `argsort`, `sort`, integer casts, and data-dependent shapes have no
useful gradient. They are not banned from the package — the spatial module needs
them — but they must be **isolated** from any path you will differentiate. Build
the Morton codes and neighbor lists once, as discrete preprocessing; keep the
differentiable physics downstream of them.

(p7-quadrature)=
## 7. Quadrature and sampling differentiate through the values, not the nodes

A Gaussian quadrature rule has fixed nodes and weights; an inverse-CDF sampler has
a fixed grid. Differentiate through the **integrand evaluated at the nodes** or the
**values being interpolated**, never through the node positions. This is why the
quadrature factory generates nodes once on the host with numpy and freezes them to
constants: the gradient flows through `f(x_i)`, and the constant $x_i$ contributes
nothing it should not.

→ [](./cumulative-trapz.md) — Newton–Cotes integration over a grid of values.

→ [](./quadrature.md) — fixed-node Gaussian, Clenshaw-Curtis, and cumulative
Simpson rules differentiate through values rather than node generation.

→ [](./interpolation.md) — PCHIP-style interpolation differentiates inside
stable limiter branches and avoids inventing monotone-table overshoot.

→ [](./regular-grid.md) — multilinear interpolation differentiates inside grid
cells while making out-of-domain policy explicit.

→ [](./bsplines.md) — B-spline evaluation differentiates cleanly through
coefficients and interior coordinates for fixed knots.

(p8-precision)=
## 8. Precision discipline

Float32 carries about 7 decimal digits; one bad subtraction can spend all of them.
Enable float64 with `jaxconfig.enable_high_precision()` before creating any array,
and request the highest matmul precision so reductions are not silently downcast on
accelerators. This is cheap insurance and the default posture for everything here.

(p9-correctness)=
## 9. Correctness over comfort

Every constant cites its source — CODATA 2018, IAU 2015, Oke & Gunn 1983 — so a
reader can audit the number, not trust it. Every method is validated against an
analytic result or a known answer. "It converged" and "it's elegant" are not
evidence. The radiation constant in this package is $a = 7.565733250\times10^{-15}\,
\erg\,\mathrm{cm}^{-3}\,\mathrm{K}^{-4}$ precisely because it is derived as
$4\sigma_\mathrm{SB}/c$ from the CODATA values, not rounded independently
(see [](../95-release/index.md)).

(p10-vectorize)=
## 10. Vectorize and compose

Prefer `vmap` over Python loops, pure functions over mutable state, and immutable
PyTrees (equinox modules) over in-place updates. Composition is what lets a
foundation stay small: a handful of well-behaved primitives, combined, cover the
ecosystem's needs without each package reinventing them.

## What we just established

These ten principles are not style preferences; they are the difference between a
gradient you can trust and one that lies to you. The rest of the theory section
shows them in specific methods. Read on:

- [](./rootfinding.md) — fixed-iteration solvers, and the `bisect` zero-gradient
  caveat (principles [2](#p2-fixed-iteration), [3](#p3-guard-singularities),
  [4](#p4-saturation)).
- [](./cumulative-trapz.md) — Newton–Cotes integration and the dx-outside ordering
  (principles [5](#p5-floating-point), [7](#p7-quadrature)).
- [](./quadrature.md) — fixed-node Gaussian and Clenshaw-Curtis quadrature plus
  cumulative Simpson panel sums (principles [7](#p7-quadrature),
  [10](#p10-vectorize)).
- [](./interpolation.md) — cubic Hermite and PCHIP-style interpolation for
  smooth table evaluation without overshoot (principles [3](#p3-guard-singularities),
  [4](#p4-saturation), [7](#p7-quadrature)).
- [](./regular-grid.md) — static-rank multilinear interpolation for gridded
  tables with explicit boundary policy (principles [4](#p4-saturation),
  [7](#p7-quadrature), [10](#p10-vectorize)).
- [](./bsplines.md) — local smooth basis functions for AD-friendly tabulated
  functions (principles [3](#p3-guard-singularities), [7](#p7-quadrature),
  [10](#p10-vectorize)).
- [](./linear-algebra.md) — weighted fits, solve wrappers, covariance helpers,
  and positive-definite jitter for small dense problems (principles
  [3](#p3-guard-singularities), [8](#p8-precision), [9](#p9-correctness)).
- [](./autodiff.md) — JVP, VJP, HVP, Gauss-Newton, and empirical Fisher-style
  products as named JAX-native helpers (principles [1](#p1-differentiability),
  [9](#p9-correctness), [10](#p10-vectorize)).
- [](./geometry.md) — vector normalization, angular distances, rotations,
  quaternions, rigid transforms, and explicit composition helpers (principles
  [1](#p1-differentiability), [9](#p9-correctness), [10](#p10-vectorize)).
- [](./distributions.md) — logpdf, CDF, and inverse-CDF kernels for normal,
  lognormal, finite power-law, and truncated-normal families (principles
  [3](#p3-guard-singularities), [5](#p5-floating-point), [7](#p7-quadrature)).
- [](./optimization.md) — robust residual losses, objective summaries,
  fixed-iteration line search, and convergence diagnostics (principles
  [1](#p1-differentiability), [2](#p2-fixed-iteration), [10](#p10-vectorize)).
- [](./ode.md) — fixed-step Euler, midpoint/RK2, RK4, and velocity-Verlet
  integration with scan-friendly gradient flow (principles
  [1](#p1-differentiability), [2](#p2-fixed-iteration), [10](#p10-vectorize)).
- [](./operators.md) — dense, diagonal, scaled, summed, composed, transposed,
  and block-diagonal linear operators as PyTrees (principles
  [1](#p1-differentiability), [9](#p9-correctness), [10](#p10-vectorize)).
- [](./special-functions.md) — stable Planck kernels, normalized log weights,
  and orthogonal polynomial bases (principles [3](#p3-guard-singularities),
  [5](#p5-floating-point), [9](#p9-correctness)).
- [](./random.md) — explicit key streams, deterministic seed manifests, and
  systematic/stratified/residual resampling (principles [6](#p6-non-diff-ops),
  [9](#p9-correctness), [10](#p10-vectorize)).
- [](./grids.md) — log grids, conservative binning, and stratified uniforms
  (principles [7](#p7-quadrature), [9](#p9-correctness), [10](#p10-vectorize)).

Then map principles to call signatures in [](../40-api/index.md), and the design
*choices* behind them in [](../30-decisions/index.md).
