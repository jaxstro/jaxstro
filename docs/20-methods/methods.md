---
title: Numerical methods
short_title: Methods
description: >-
  A method atlas for turning scientific questions into auditable JAX
  computations while preserving explicit transform and claim boundaries.
---

Use this page when you need to choose a numerical method from the scientific
question, the mathematical structure, and the evidence the result will require.

Methods connect mathematical representations to executed computations. The
families below group the current Jaxstro methods by the kind of question they
answer; the shared transform-contract principles that follow apply across those
families.

::::{grid} 1 2 2 2

:::{card} Change, constraints, and evolution
:link: ./change-constraints-evolution/rootfinding.md

Differentiate programs, solve scalar constraints, optimize objectives, and
advance fixed-step dynamical systems.
:::

:::{card} Approximation from finite information
:link: ./approximation-integration/interpolation.md

Interpolate tabulated values, construct smooth bases, and accumulate integrals
with explicit spacing and boundary contracts.
:::

:::{card} Linear structure and reusable operators
:link: ./linear-structure/linear-algebra.md

Work with dense linear structure, matrix-free operators, and stable reusable
function bases.
:::

:::{card} Randomness as a computational object
:link: ./probability-sampling/random.md

Represent probability laws, own PRNG state explicitly, and separate random
computation from discrete sampling and resampling decisions.
:::

:::{card} Discrete worlds: grids, meshes, and neighborhoods
:link: ./discrete-space/grids.md

Construct discrete domains, conservative remaps, and neighborhood candidates
without inventing smooth derivatives for discrete identity operations.
:::

:::{card} Signals as sampled evidence
:link: ./signals/signal-axes.md

Define sample and frequency axes, window finite records, estimate power, and
interpret cross-spectral phase and delay. These guides describe a planned
scientific convention layer; they do not claim an implemented `jaxstro.signal`
runtime.
:::
::::

:::{note}
A method choice is only one link in the research chain. Connect it to the
representation that gives the inputs meaning, the executed JAX program, an
independent audit, and the narrowest warranted scientific claim.
:::

## Shared numerical contracts

A function can return the right number and still be wrong. When a gradient is
part of its contract, a silently zero, `NaN`, or detached derivative can break
the science even though the value "worked." When the operation is deliberately
discrete, pretending it has a useful gradient is equally wrong. *It ran* is not
*it is correct*. Elegant nonsense is still nonsense.

jaxstro therefore requires every public numerical path to name its transform
contract. Smooth pathwise gradients receive independent finite-difference
checks. Expected-zero, blocked, surrogate, validation-only, and discrete paths
state their narrower claim instead of borrowing the language of smooth
inference. This page is the thesis: ten principles for making those scientific
contracts explicit, testable, and teachable.

If you are starting from a research question rather than a method name, use
[](../40-workflows/differentiable-research/science-patterns.md) to route events, equilibria, integrals, tabulated
models, limiting distributions, spatial interactions, and provenance to the
relevant module and evidence.

:::{tip} Already fluent in differentiable programming?
Skip to the principle that bites you most often - most people's is
[](#p3-guard-singularities) (the `where`-trap) or [](#p4-saturation) (the silent
gradient killer) - or go straight to the method pages:
[](./change-constraints-evolution/rootfinding.md), [](./approximation-integration/cumulative-trapz.md), [](./approximation-integration/quadrature.md),
[](./approximation-integration/interpolation.md), [](./approximation-integration/regular-grid.md), and [](./approximation-integration/bsplines.md).
The dense helper layer for small fits and covariance diagnostics is
[](./linear-structure/linear-algebra.md), objective helpers live in [](./change-constraints-evolution/optimization.md),
fixed-step ODE helpers live in [](./change-constraints-evolution/ode.md), and generic special-function kernels
live in [](./linear-structure/special-functions.md). Generic distribution kernels live in
[](./probability-sampling/distributions.md), vector geometry lives in
[](../30-representations/geometry-coordinates/geometry.md), and
matrix-free algebra helpers live in
[](./linear-structure/operators.md). Autodiff product helpers live in [](./change-constraints-evolution/autodiff.md). Grid
construction, conservative rebinning, and stratified uniforms are in
[](./discrete-space/grids.md), structured 1D mesh helpers are in [](./discrete-space/meshes.md), and explicit
PRNG streams are in [](./probability-sampling/random.md). Sampling
and resampling decisions are in
[](./probability-sampling/sampling.md). Quantity semantics
and boundary conversion live in
[](../30-representations/units-quantities/quantities.md). Spatial indexing,
candidate recall, and exact fixed-radius pairs are in [](./discrete-space/spatial.md).
:::

## Gradient contracts

The first question is not "does `jax.grad` run?" It is "what derivative claim is
scientifically valid here?" `jaxstro.testing` exposes five live contracts, and
the audit gate interprets each one differently.

```{list-table} Gradient contracts
:header-rows: 1
:label: tbl-gradient-contracts

* - Gradient contract
  - AD expectation
  - FD role
  - Inference / claim boundary
* - `smooth_pathwise`
  - Finite, nonzero AD agrees with the local smooth derivative.
  - Required: AD and central FD must agree within the declared tolerance.
  - Only a clean `smooth_pathwise` result is inference-ready.
* - `known_zero`
  - AD is intentionally zero because the output is locally insensitive.
  - Required: FD must also be zero; an appearing derivative is a contract change.
  - Documents insensitivity, not a usable inference direction.
* - `known_blocked`
  - Gradient flow is intentionally stopped or unavailable; the audited result
    must remain finite.
  - AD-FD equality is not part of this gate.
  - Never inference-ready; callers must not describe it as a physical gradient.
* - `surrogate`
  - A live, nonzero surrogate sensitivity is required.
  - FD equality to the underlying physical model is not claimed.
  - May support an explicitly named surrogate claim; never silently substitutes
    for a physical derivative.
* - `validation_only`
  - Any derivative is diagnostic evidence for a bounded validation question.
  - FD comparison is optional and must be stated by the validation itself.
  - Not an inference, Fisher, or OED gradient.
```

(p1-differentiability)=
## 1. Classify the transform contract first

Before implementation, classify the transform contract first. A smooth kernel,
an expected-zero sensitivity, a stopped gradient, a surrogate, and a discrete
index builder require different evidence. For `smooth_pathwise` and
`known_zero`, the audit computes both AD and an independent finite-difference
estimate. Other contracts fail closed for inference and carry only the narrower
claim they name. See [](../60-validation/index.md) for the measured audit.

(p2-fixed-iteration)=
## 2. Fixed iteration is necessary, not sufficient

Fixed scan lengths give JAX a static computation and avoid reverse-mode limits
around data-dependent convergence loops. They do not make the update map smooth.
A fixed-step solver can still contain branch-selected intervals, clips, or
singular derivatives.

For a smooth function with a nonzero derivative and a parameter-independent
initial guess, Newton can carry a `smooth_pathwise` contract after AD-FD
verification. In contrast, bisection is a branch-selected forward solve: it can
deliver an accurate root value without providing the smooth inverse sensitivity
needed for inference. Iteration count and gradient contract are separate facts.

-> [](./change-constraints-evolution/rootfinding.md) - the distinct contracts of `bisect`, `newton`, and
`newton_ppf`.

(p3-guard-singularities)=
## 3. Guard singularities without killing the gradient - the `where`-trap

The natural way to avoid a division by zero is
`jnp.where(d == 0, fallback, a / d)`. The selected forward value may be finite,
but both branch expressions are traced. If the unselected expression creates an
`inf` or `NaN`, its reverse-mode cotangent can meet a zero multiplier and the
inactive branch can still poison a derivative. The discipline is to guard the
**operand**, not only the selected result: sanitize the denominator before
division, then select the intended value. `safe_div` and `safe_log` implement
that policy for their documented domains.

(p4-saturation)=
## 4. Saturation is a silent gradient killer

`clip`, `min`, `max`, and `floor` are piecewise or discrete operations. They can
zero, route, or make a gradient convention-dependent at their boundaries.
Sometimes that is the intended hard-bound contract. Sometimes it pins a
parameter on a wall while an optimizer reports convergence. Name which case you
intend. When `newton_ppf` clips iterates to `[lo, hi]`, for example, its smooth
pathwise claim applies to an interior solution, not to saturation at the support.

-> [](./change-constraints-evolution/rootfinding.md#newton-ppf) discusses the clip-to-support trade-off.

(p5-floating-point)=
## 5. Floating point is part of the math

Catastrophic cancellation, overflow in `exp`, and underflow in `log` are not
edge cases - they are the common case in likelihood code. Work in the log domain,
use `log1p`/`expm1` near zero, and sum in the order that minimizes error. jaxstro
provides `stable_log1p`, `stable_expm1`, `safe_log`, `safe_exp`, and Neumaier
compensated summation for reductions where the ordinary `sum` loses digits. And
turn on float64 first ([](#p8-precision)).

(p6-non-diff-ops)=
## 6. Non-differentiable operations are forbidden in the differentiable graph

`argmax`, `argsort`, `sort`, integer casts, and data-dependent shapes have no
useful gradient. They are not banned from the package - the spatial module needs
them - but they must be **isolated** from any path you will differentiate. Build
the Morton codes and neighbor lists once, as discrete preprocessing; keep the
differentiable physics downstream of them.

-> [](./discrete-space/spatial.md) - fixed-capacity cells, candidate recall, exact-pair overflow,
and the boundary between discrete identity and downstream differentiable values.

(p7-quadrature)=
## 7. Quadrature and sampling differentiate through the values, not the nodes

A Gaussian quadrature rule has fixed nodes and weights; an inverse-CDF sampler has
a fixed grid. Differentiate through the **integrand evaluated at the nodes** or the
**values being interpolated**, never through the node positions. This is why the
quadrature factory generates nodes once on the host with numpy and freezes them to
constants: the gradient flows through `f(x_i)`, and the constant $x_i$ contributes
nothing it should not.

-> [](./approximation-integration/cumulative-trapz.md) - Newton-Cotes integration over a grid of values.

-> [](./approximation-integration/quadrature.md) - fixed-node Gaussian, Clenshaw-Curtis, and cumulative
Simpson rules differentiate through values rather than node generation.

-> [](./approximation-integration/interpolation.md) - PCHIP-style interpolation differentiates inside
stable limiter branches and avoids inventing monotone-table overshoot.

-> [](./approximation-integration/regular-grid.md) - multilinear interpolation differentiates inside grid
cells while making out-of-domain policy explicit.

-> [](./approximation-integration/bsplines.md) - B-spline evaluation differentiates cleanly through
coefficients and interior coordinates for fixed knots.

(p8-precision)=
## 8. Precision discipline

Float32 carries about 7 decimal digits; one bad subtraction can spend all of them.
Enable float64 with `jaxconfig.enable_high_precision()` before creating any array,
and request the highest matmul precision so reductions are not silently downcast on
accelerators. This is cheap insurance and the default posture for everything here.

(p9-correctness)=
## 9. Correctness over comfort

Every constant cites its source - CODATA 2018, IAU 2015, Oke & Gunn 1983 - so a
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

These ten principles are not style preferences. They separate gradients that can
support inference from expected-zero, blocked, surrogate, validation-only, and
discrete paths with narrower claims. The rest of the theory section shows those
boundaries in specific methods. Read on:

- [](./change-constraints-evolution/rootfinding.md) - fixed-iteration solvers, and the `bisect` zero-gradient
  caveat (principles [2](#p2-fixed-iteration), [3](#p3-guard-singularities),
  [4](#p4-saturation)).
- [](./approximation-integration/cumulative-trapz.md) - Newton-Cotes integration and the dx-outside ordering
  (principles [5](#p5-floating-point), [7](#p7-quadrature)).
- [](./approximation-integration/quadrature.md) - fixed-node Gaussian and Clenshaw-Curtis quadrature plus
  cumulative Simpson panel sums (principles [7](#p7-quadrature),
  [10](#p10-vectorize)).
- [](./approximation-integration/interpolation.md) - cubic Hermite and PCHIP-style interpolation for
  smooth table evaluation without overshoot (principles [3](#p3-guard-singularities),
  [4](#p4-saturation), [7](#p7-quadrature)).
- [](./approximation-integration/regular-grid.md) - static-rank multilinear interpolation for gridded
  tables with explicit boundary policy (principles [4](#p4-saturation),
  [7](#p7-quadrature), [10](#p10-vectorize)).
- [](./approximation-integration/bsplines.md) - local smooth basis functions for AD-friendly tabulated
  functions (principles [3](#p3-guard-singularities), [7](#p7-quadrature),
  [10](#p10-vectorize)).
- [](./linear-structure/linear-algebra.md) - weighted fits, solve wrappers, covariance helpers,
  and positive-definite jitter for small dense problems (principles
  [3](#p3-guard-singularities), [8](#p8-precision), [9](#p9-correctness)).
- [](./change-constraints-evolution/autodiff.md) - JVP, VJP, HVP, Gauss-Newton, and empirical Fisher-style
  products as named JAX-native helpers (principles [1](#p1-differentiability),
  [9](#p9-correctness), [10](#p10-vectorize)).
- [](../30-representations/geometry-coordinates/geometry.md) - vector normalization,
  angular distances, rotations,
  quaternions, rigid transforms, and explicit composition helpers (principles
  [1](#p1-differentiability), [9](#p9-correctness), [10](#p10-vectorize)).
- [](./discrete-space/spatial.md) - Morton and linear cells, capacity/overflow, approximate
  candidates, and exact fixed-radius neighbors as discrete preprocessing
  (principles [1](#p1-differentiability), [6](#p6-non-diff-ops),
  [9](#p9-correctness)).
- [](./probability-sampling/distributions.md) - logpdf, CDF, and inverse-CDF kernels for normal,
  lognormal, finite power-law, and truncated-normal families (principles
  [3](#p3-guard-singularities), [5](#p5-floating-point), [7](#p7-quadrature)).
- [](./change-constraints-evolution/optimization.md) - robust residual losses, objective summaries,
  fixed-iteration line search, and convergence diagnostics (principles
  [1](#p1-differentiability), [2](#p2-fixed-iteration), [10](#p10-vectorize)).
- [](./change-constraints-evolution/ode.md) - fixed-step Euler, midpoint/RK2, RK4, and velocity-Verlet
  integration with scan-friendly gradient flow (principles
  [1](#p1-differentiability), [2](#p2-fixed-iteration), [10](#p10-vectorize)).
- [](./linear-structure/operators.md) - dense, diagonal, scaled, summed, composed, transposed,
  and block-diagonal linear operators as PyTrees (principles
  [1](#p1-differentiability), [9](#p9-correctness), [10](#p10-vectorize)).
- [](./linear-structure/special-functions.md) - stable Planck kernels, normalized log weights,
  and orthogonal polynomial bases (principles [3](#p3-guard-singularities),
  [5](#p5-floating-point), [9](#p9-correctness)).
- [](./probability-sampling/random.md) - explicit key streams and
  deterministic seed manifests (principles [6](#p6-non-diff-ops),
  [9](#p9-correctness), [10](#p10-vectorize)).
- [](./probability-sampling/sampling.md) -
  systematic/stratified/residual resampling (principles [6](#p6-non-diff-ops),
  [9](#p9-correctness), [10](#p10-vectorize)).
- [](./discrete-space/grids.md) - log grids, conservative binning, and stratified uniforms
  (principles [7](#p7-quadrature), [9](#p9-correctness), [10](#p10-vectorize)).
- [](./discrete-space/meshes.md) - structured 1D cell/face geometry, finite-volume stencils,
  and conservative cell-average remapping (principles [7](#p7-quadrature),
  [9](#p9-correctness), [10](#p10-vectorize)).
- [](../30-representations/units-quantities/quantities.md) - exact dimensions, JAX
  PyTree quantities, parser
  canonicalization, bases, constants, equivalencies, and the raw-array boundary
  pattern (principles [1](#p1-differentiability), [9](#p9-correctness),
  [10](#p10-vectorize)).

Then map principles to call signatures in [](../50-api/api.md), and the design
*choices* behind them in [](../30-decisions/index.md).
