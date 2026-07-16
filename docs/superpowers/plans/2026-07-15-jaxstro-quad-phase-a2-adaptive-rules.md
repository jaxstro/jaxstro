# Jaxstro Quad Phase A2 Adaptive Rules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a complete primal-only, one-dimensional adaptive quadrature layer behind `jaxstro.quad.integrate`, including every approved Gauss-Kronrod pair, adaptive Clenshaw-Curtis, adaptive tanh-sinh, Romberg, and Romberg-tanh-sinh with fixed-capacity JAX control, deterministic typed failure states, exact work accounting, and researcher-first documentation.

**Architecture:** All locally adaptive methods operate on one normalized reference partition. A method-specific local estimator produces a high-order value and payload-shaped nonnegative error evidence; one shared controller owns region priority, bisection, capacity accounting, global tolerance, and status precedence. Standard Romberg uses a fixed-capacity Richardson table. Romberg-tanh-sinh uses a separate nested double-exponential level sequence without unjustified classical Richardson columns. All engines use fixed-shape arrays and JAX loop control. Phase A2 exposes only `gradient="stop"`; replay derivatives and moving-bound differentiation remain Phase A3 work.

**Tech Stack:** Python 3.11+, JAX, `jax.numpy`, jaxtyping, pytest, Ruff, MyPy, MyST. Netlib QUADPACK is the primary source for Gauss-Kronrod constants and estimator formulas. Analytic references are the primary validation oracles; SciPy and Quadax may be used only as secondary comparison checks and never become runtime dependencies.

## Global constraints and frozen decisions

- Follow `CLAUDE.md`, `AGENTS.md`, the approved capability design in `docs/superpowers/specs/2026-07-15-jaxstro-quad-capability-program-design.md`, and the verified A1 contracts recorded in `STATUS.md`.
- Use test-driven development. Every behavioral change begins with a failing regression.
- Add no runtime dependency and no second adaptive implementation lane.
- Accept raw arrays only. Quantity normalization remains Phase A3.
- Implement primal execution only. `integrate(..., gradient="stop")` is the only accepted A2 gradient policy, and every `QuadResult` leaf is explicitly stopped. Phase A3 will add replay without differentiating this controller.
- Do not differentiate a `jax.lax.while_loop`, sorting, region choice, bisection, stopping, capacity logic, or status.
- Method type and parameters, payload shape, breakpoint count, capacities, error norm, and gradient policy are static under JIT. Bounds, breakpoint values, tolerances, and explicit `args` leaves may be dynamic arrays.
- No Python loop advances regions, refinement levels, nodes, evaluation batches, or Richardson columns at runtime. Static Python dispatch may select a method or construct fixed literal rule data.
- Every adaptive result is a `QuadResult`; every error estimate is payload-shaped, real, and nonnegative, including for complex values.
- `QuadError.kind` is `EMBEDDED_RULE` for Gauss-Kronrod and `REFINEMENT_DIFFERENCE` for Clenshaw-Curtis, tanh-sinh, and both Romberg variants. `confidence_level` is `nan` for all A2 methods.
- `QuadWork.evaluations` counts integrand array entries logically executed, not rule calls or unique floating-point coordinates. For one unbatched solve this is also the physically active lane count. Under ordinary `jax.vmap`, batched loop lowering may execute masked lanes while another batch member remains active; work remains the logical per-lane count. Cost-sensitive independent batches should use `jax.lax.map`.
- A fixed rule difference or refinement difference is an indicator, not a universal error bound. Documentation and tests must not claim otherwise.
- `LebesgueMeasure` and `WeightedMeasure` are the A2 adaptive measure surface. Classical measures remain matched fixed-Gaussian declarations in A2 rather than being silently reinterpreted.
- Gauss-Kronrod and adaptive Clenshaw-Curtis accept finite `Interval` domains, including breakpoints. Adaptive tanh-sinh accepts every Phase A domain, including finite breakpoints. Romberg accepts a finite `Interval` without breakpoints. Romberg-tanh-sinh accepts every Phase A domain without breakpoints.
- Preserve every A1 callable, legacy object identity, and the byte-identical probabilists' Hermite compatibility helper. Phase A2 explicitly permits one scientifically necessary A1 numerical correction: harden the private tanh-sinh node/weight construction against fixed-cutoff false convergence while preserving the public `TanhSinhRule` and `fixed` signatures and returned integral payload shapes; the quadrature-owned leading node-batch length may change.
- Do not implement replay derivatives, quantities, comparisons claiming superiority, sibling migrations, deprecations, publication, push, Pages runs, or live-site changes.

## Public interfaces

```python
GaussKronrod(pair=21)
AdaptiveClenshawCurtis(initial_order=17)
AdaptiveTanhSinh(initial_level=3)
Romberg(initial_level=1)
RombergTanhSinh(initial_level=1)

integrate(
    fun,
    domain,
    *,
    args=(),
    method,
    measure=None,
    epsabs,
    epsrel,
    max_evaluations,
    max_regions,
    error_norm=MaxNorm(),
    gradient="stop",
) -> QuadResult
```

The A2 default is deliberately `gradient="stop"`. Phase A3 changes the default only when replay is implemented and independently verified.

## Reference-space controller contract

All h-adaptive methods store active regions as intervals `[u_i, v_i]` inside the normalized reference interval `[-1, 1]`. A local rule coordinate `z` maps to

```{math}
:label: eq-a2-local-reference-map
t_i(z)=c_i+h_i z,
\qquad
c_i=\frac{u_i+v_i}{2},
\qquad
h_i=\frac{v_i-u_i}{2}.
```

The existing domain map then sends `t_i` to the physical coordinate. The local Jacobian is the product of `h_i` and the domain-map Jacobian. This keeps accepted partitions independent of physical bounds and prepares the correct A3 replay boundary.

For a finite interval, physical breakpoints are converted to normalized reference values, sorted in increasing reference order, and stopped in derivatives. The requested integral orientation remains a separate sign from the existing domain map.

Each active region stores:

- reference lower and upper endpoints;
- the high-order payload value;
- the payload-shaped error estimate;
- the scalar priority obtained from the selected error norm; and
- active and nonfinite flags.

Every nested local pair computes the high value, low value, and an
absolute-integral proxy from the same evaluated node union. The `ErrorKind`
records whether the underlying comparison is an embedded rule or a refinement
difference. Gauss-Kronrod alone uses the calibrated QUADPACK rescaling in
[](#eq-a2-gk-rescaled-error). Clenshaw-Curtis and tanh-sinh use their raw
refinement difference plus the sum of separate low- and high-reduction floors,
with $\gamma_n=n\epsilon/(1-n\epsilon)$ for each active summand count.
Tanh-sinh additionally carries a separate outer-tail indicator. Code utilities
may be shared; an estimator calibration is never generalized to a different
method without evidence.

Initialization evaluates all declared breakpoint segments. Each refinement selects the lowest-index region among ties for maximal priority, bisects it, replaces the parent with its left child, and appends its right child. A split costs two local high-rule node counts and increases `active_regions` and `refinements` by one.

The global value is the sum of active region values. The global payload error is the sum of active payload errors. Its selected norm is compared with

```{math}
:label: eq-a2-controller-tolerance
\tau=\max\!\left(\epsilon_{\mathrm{abs}},
\epsilon_{\mathrm{rel}}\lVert\widehat{I}\rVert\right).
```

## Status and validation policy

Static structural failures raise eagerly:

- unsupported method, domain, or measure combinations;
- unsupported Gauss-Kronrod pair;
- invalid static order or level;
- noninteger or nonpositive capacities;
- capacities smaller than the static initialization cost;
- non-scalar or non-real tolerance inputs;
- payloads without the leading node axis; and
- any gradient policy other than `"stop"`.

Dynamic numerical outcomes use this fixed precedence:

1. `INVALID_INPUT` for nonfinite bounds, invalid traced breakpoints, or negative/nonfinite tolerances;
2. `NONFINITE_INTEGRAND` when an evaluated integrand, density, mapped node contribution, or local estimate is nonfinite;
3. `CONVERGED` when the named estimator meets the tolerance;
4. `ROUNDOFF_LIMITED` when a selected reference midpoint equals an endpoint;
   when deterministic selected-region/global stagnation thresholds are met
   while tolerance remains unmet; or when global `RombergTanhSinh` has reached
   the dtype's final strictly interior mapped node and its nonvanishing tail
   indicator remains above tolerance;
5. `MAX_EVALUATIONS` when the next complete refinement would exceed the evaluation budget;
6. `MAX_REGIONS` when the next refinement would exceed region capacity;
7. `DIVERGENCE_SUSPECTED`; and
8. `ERROR_ESTIMATE_UNAVAILABLE`.

The final two codes remain reserved in A2. A2 does not invent a divergence heuristic or return an unavailable estimate for methods that all have named indicators.

The h-adaptive stagnation counters are global, consecutive counters updated
only from the selected split. Let $V_p,E_p$ be the selected parent value and
scalar error priority, and let $V_c=V_l+V_r$, $E_c=E_l+E_r$. With the selected
error norm, define

```{math}
:label: eq-a2-stagnation-scale
\delta_V=\lVert V_c-V_p\rVert,
\qquad
s_V=32\epsilon\max\!\left(\lVert V_p\rVert,\lVert V_c\rVert,
u\right),
```

where $u$ is the smallest positive normal value. The no-improvement counter
increments exactly when $\delta_V\le s_V$ and $E_c\ge0.99E_p$; otherwise it
resets to zero. Beginning after ten completed refinements, the growth counter
increments exactly when $E_c>1.01E_p$; otherwise it resets to zero. Six
consecutive no-improvement hits or five consecutive growth hits produce
`ROUNDOFF_LIMITED` while tolerance remains unmet. Tests cover the state one hit
below and exactly at both thresholds. A local floor flag alone never abandons
other refinable regions.

The zero-width fast path runs only after dynamic input validity. A finite zero
width with finite bounds and valid real tolerances returns exact zero
value/error/tolerance-compatible evidence, `CONVERGED`, and zero work after
static payload-shape inference without numerically evaluating the integrand.
Nonfinite equal bounds or invalid tolerances return `INVALID_INPUT` instead.

## Method-specific contracts

### Gauss-Kronrod

Support pairs 15, 21, 31, 41, 51, and 61. Literal abscissae and weights come from Netlib QUADPACK `dqk15.f`, `dqk21.f`, `dqk31.f`, `dqk41.f`, `dqk51.f`, and `dqk61.f`. Store one symmetric Kronrod node array, Kronrod weights, and an aligned embedded-Gauss weight array per pair.

For each payload component, compute the Kronrod value, embedded Gauss value,
absolute integral proxy, and mean-deviation proxy. Begin with
$e=e_0=|Q_K-Q_G|$. Rescale only where both $R_{\mathrm{asc}}\ne0$ and
$e_0\ne0$:

```{math}
:label: eq-a2-gk-rescaled-error
e_0=\left|Q_K-Q_G\right|,
\qquad
e=R_{\mathrm{asc}}
\min\!\left[1,
\left(\frac{200e_0}{R_{\mathrm{asc}}}\right)^{3/2}\right],
```

Then apply

```{math}
:label: eq-a2-gk-roundoff-floor
e\leftarrow\max\!\left(e,50\epsilon R_{\mathrm{abs}}\right)
```

only under QUADPACK's safe-underflow condition
$R_{\mathrm{abs}}>u/(50\epsilon)$, where $u$ is the smallest positive normal
value. Tests cover the zero, constant, complex, tiny-magnitude, float32, and
float64 branches against an independent scalar translation. The estimator
remains an indicator despite this stabilization.

The embedded Gauss degrees are 13, 19, 29, 39, 49, and 59. The corresponding Kronrod degrees are 23, 31, 47, 61, 77, and 91.

### Adaptive Clenshaw-Curtis

Require `initial_order = 2^k + 1` with `k >= 2`. A local estimate constructs the high rule of order `2 * initial_order - 1` and derives the low estimate from the high-node even-index subset. The integrand is evaluated only at the high nodes. This preserves nesting and gives exact node-level work counts.

Construct both low- and high-order A1 rule data. Evaluate the integrand only at
the high nodes, verify that low nodes equal the even-index high subset, reduce
the high values with high weights, and reduce `high_values[::2]` with the
separately constructed low weights. Nested nodes do not share weights.

The payload error is the componentwise sum of `abs(high - low)` and the two
reduction floors

```{math}
:label: eq-a2-cc-summation-floor
E_{\mathrm{sum}}
=\gamma_{n_h}R_{\mathrm{abs},h}
+\gamma_{n_l}R_{\mathrm{abs},l},
\qquad
\gamma_n=\frac{n\epsilon}{1-n\epsilon},
```

where $n_h$ and $n_l$ count active summands in their respective reductions;
$\gamma_n=\infty$ when $n\epsilon\ge1$. Endpoint values are evaluated, so this method is
documented for finite integrands at interval endpoints. Endpoint singularities
route to adaptive tanh-sinh.

### Adaptive tanh-sinh

A prerequisite replaces A1's fixed `$|s|=3$` lattice with the following frozen
representability-aware construction. For active real dtype $d$, let

```{math}
:label: eq-a2-ts-map
x(t)=\tanh\!\left(\frac{\pi}{2}\sinh t\right),
\qquad
\omega(t)=\frac{\pi}{2}
\frac{\cosh t}{\cosh^2\!\left(\frac{\pi}{2}\sinh t\right)},
```

and let $x_d^-=\operatorname{nextafter}_d(1,0)$. The deterministic positive
candidate cap at level $\ell$ is

```{math}
:label: eq-a2-ts-cap
h_\ell=2^{-\ell},
\qquad
T_d=\operatorname{asinh}\!\left(
\frac{2}{\pi}\operatorname{atanh}(x_d^-)
\right),
\qquad
K_\ell^{\mathrm{cap}}=\left\lceil\frac{T_d}{h_\ell}\right\rceil+1.
```

For $\ell>0$, replace this raw cap by
$\max(K_\ell^{\mathrm{cap}},2\max A_{\ell-1})$ so every mandatory mapped
coarse index is representable in the static candidate array even when rounding
the analytic cap downward would otherwise exclude it.

Let $A_\ell\subseteq\{0,\ldots,K_\ell^{\mathrm{cap}}\}$ be the retained
nonnegative index set. At the base level, reserve $0$ and scan every remaining
candidate through the cap in increasing index order, retaining it only when
$x(kh_\ell)$ is finite, strictly interior, strictly greater than the previous
retained node, and $h_\ell\omega(kh_\ell)$ is finite and strictly positive.
Do not terminate the scan on a collision.

Construct each finer level recursively. First reserve the mandatory mapped
coarse set $2A_\ell$. Then scan every unreserved fine candidate through the cap
in increasing index order. Retain it only when the same finite/interior/
positive-weight predicates hold and its node lies strictly between its nearest
retained neighbors (or strictly beyond the final retained neighbor). Reserved
coarse nodes always win a collision with a new odd or outer candidate. This
defines the sole deterministic collision rule and guarantees

```{math}
:label: eq-a2-ts-nesting
2A_\ell\subseteq A_{\ell+1},
\qquad
N_\ell=2\lvert A_\ell\rvert-1.
```

Reflect retained nonzero indices exactly to create the negative half. Fine
indices outside $2A_\ell$ are new; the active set need not be contiguous.

The public `tanh_sinh_rule_data` returns only the compact $N_\ell$ active unique
nodes and weights. Compact construction is a cached host-side static setup
operation keyed by `(level, active JAX precision policy)`: NumPy/Python computes
the cap and recursive retained-index tuples outside tracing, then converts the
immutable tuples to JAX constants. This is permitted static rule-data setup,
not a runtime quadrature loop. The private adaptive constructor
also exposes padded candidates through $K_\ell^{\mathrm{cap}}$, active masks,
the explicit coarse-to-fine map, and terminal metadata. Padded inactive
coordinates use a finite safe sentinel and zero weight, but are never part of
the public fixed-rule array.

A local estimate forms the union needed by levels `initial_level` and
`initial_level + 1`, physically skips inactive lanes for an unbatched solve,
and evaluates each active high/outer entry once. Each local reference region is
composed with the existing finite, semi-infinite, or full-line domain map.

For a mapped integrand, let $g_{\ell,k}$ be its transformed payload density,
including $\omega(kh_\ell)$ and every domain, local-region, orientation, and
measure factor but excluding the mesh spacing, and let
$c_{\ell,k}=h_\ell g_{\ell,k}$. At adjacent levels define

```{math}
:label: eq-a2-ts-error
E_{\mathrm{disc}}=\left|Q_{\ell+1}-Q_\ell\right|,
\qquad
E_{\mathrm{sum}}
=\gamma_{N_{\ell+1}}R_{\mathrm{abs},\ell+1}
+\gamma_{N_\ell}R_{\mathrm{abs},\ell},
```

where $N_\ell=2\lvert A_\ell\rvert-1$. Let
$K_\ell=\max A_\ell$. On the signed reflected active set, the newly exposed
outer shell is
$S_{\ell+1}=\{k:k\in\widetilde A_{\ell+1},\ 2K_\ell<|k|<K_{\ell+1}\}$ and

```{math}
:label: eq-a2-ts-tail
E_{\mathrm{tail}}
=\sum_{k\in S_{\ell+1}}\left|c_{\ell+1,k}\right|
+\left|g_{\ell+1,-K_{\ell+1}}\right|
+\left|g_{\ell+1,K_{\ell+1}}\right|,
\qquad
E=E_{\mathrm{disc}}+E_{\mathrm{sum}}+E_{\mathrm{tail}}.
```

Every operation is payload-componentwise before the selected scalar error norm.
The terminal terms deliberately exclude $h_{\ell+1}$: they remain unchanged
when a finer mesh retains the same terminal parameter coordinate, shrink when
the representable transformed extent genuinely advances into a decaying tail,
and inherit the local physical-region scale under h-adaptive bisection. They
are dimensionally transformed-density indicators over the dimensionless
parameter scale, not claimed tail bounds.
Thus the payload error combines three independently visible terms:

- the raw low/high discretization difference;
- the sum of the low- and high-reduction floors; and
- an outer-shell indicator from newly exposed tail nodes plus the terminal
  transformed contribution at each representable end.

The outer-shell indicator is not called a bound. Representable extent is
exhausted when no candidate after $K_{\ell+1}$ through
$K_{\ell+1}^{\mathrm{cap}}$ satisfies the frozen predicates against the
retained set. This level-local fact is not global representability exhaustion:
several dyadic levels may share a terminal coordinate before a finer level
exposes a new one. For h-adaptive `AdaptiveTanhSinh`, exhaustion never
terminates the controller by itself: $E_{\mathrm{tail}}$ remains the region
priority and physical bisection continues until convergence, capacity
exhaustion, midpoint collapse, or the frozen stagnation rule. For global
`RombergTanhSinh`, dtype exhaustion is true only when its terminal mapped node
equals $x_d^-=\operatorname{nextafter}_d(1,0)$. If dtype exhaustion is true,
$\lVert E_{\mathrm{disc}}+E_{\mathrm{sum}}\rVert\le\tau$, and
$\lVert E_{\mathrm{tail}}\rVert>\tau$, the result is `ROUNDOFF_LIMITED`.
An unchanged terminal coordinate alone never triggers this status; refinement
continues subject to the evaluation budget.
Nonfinite active mapped contributions produce
`NONFINITE_INTEGRAND`. An unbatched `lax.map` solve physically skips inactive
candidates. Ordinary `vmap` may lower conditions to masked execution, so all
inactive integrand, density, Jacobian, contribution, and nonfinite-flag values
are replaced by finite neutral values before reduction; correctness does not
depend on physical skipping.

### Romberg and Romberg-tanh-sinh

Standard Romberg uses one fixed-capacity Richardson table and one
`jax.lax.while_loop`. It starts with the composite trapezoid hierarchy. Base
level zero evaluates two endpoints. Constructing through finest zero-based
level $n$ creates $n+1$ levels, uses exactly $2^n+1$ logical evaluations, and
records `refinements=n`. `initial_level=L>=1` requires the complete static
initialization cost $2^L+1$.

At level `n`, standard Romberg reuses the previous trapezoid estimate:

```{math}
:label: eq-a2-romberg-refinement
R_{n,0}
=\frac{1}{2}R_{n-1,0}
+
h_n\sum_{j=1}^{2^{n-1}}
f\!\left(a+(2j-1)h_n\right).
```

Richardson extrapolation uses

```{math}
:label: eq-a2-romberg-richardson
R_{n,m}
=R_{n,m-1}
+\frac{R_{n,m-1}-R_{n-1,m-1}}{4^m-1}.
```

The number of new standard nodes is dynamic in the level but bounded by the
static budget. Allocate maximum lane arrays once. A `jax.lax.map` body receives
a one-node leading axis and uses a per-lane `jax.lax.cond` to physically skip
inactive lanes in an unbatched solve. Richardson columns use
`jax.lax.fori_loop`, never a Python loop.

`RombergTanhSinh` deliberately does **not** apply the classical $4^m-1$
Richardson table. Its representability-aware double-exponential sequence adds
outer nodes as well as interior odd nodes, so nesting alone does not justify the
Euler-Maclaurin exponent. It uses a fixed-capacity nested level controller,
reuses every active coarse contribution, and compares successive complete
level estimates plus outer-tail evidence. Through finest level $n$, its exact
logical work is the active-node count $N_n$ from the nested mask, not the raw
candidate capacity.

For standard Romberg, the value is the newest diagonal entry. Each base
trapezoid entry carries $F_{n,0}=\gamma_{N_n}R_{\mathrm{abs},n}$. Writing
$q=4^m$, propagate the payload-shaped roundoff proxy through every Richardson
column as

```{math}
:label: eq-a2-romberg-roundoff
F_{n,m}
=\frac{qF_{n,m-1}+F_{n-1,m-1}}{q-1}
+\gamma_3\frac{q\left|R_{n,m-1}\right|
+\left|R_{n-1,m-1}\right|}{q-1}.
```

Its payload error is the absolute difference between successive diagonal
entries plus $F_{n,n}+F_{n-1,n-1}$. Cancellation-heavy tests ratchet that this
propagated floor, rather than a base-rule floor alone, controls stopping. For
Romberg-tanh-sinh, the value is the newest level estimate
and the payload error combines successive-level, summation-floor, and tail
indicators. Both record `work.levels=n+1`, `work.refinements=n`, and
`work.active_regions=1`. Both variants require `initial_level>=1` and reject
breakpoints in A2.

## File structure

- `src/jaxstro/quad/methods.py`: frozen adaptive method configurations.
- `src/jaxstro/quad/_gk.py`: QUADPACK Gauss-Kronrod data and local estimator.
- `src/jaxstro/quad/_adaptive.py`: reference-region records, transformed integrand, local paired-rule adapters, and shared h-adaptive controller.
- `src/jaxstro/quad/_romberg.py`: shared standard/tanh-sinh nested refinement engine.
- `src/jaxstro/quad/adaptive.py`: public validation, method dispatch, result assembly, stop-gradient boundary, and `integrate`.
- `src/jaxstro/quad/result.py`: only if a helper is needed for stopped result assembly or eager status formatting; public record fields remain unchanged.
- `src/jaxstro/quad/__init__.py`: public adaptive exports.
- `tests/unit/quad/`: method records, rule data, local estimators, controller, statuses, work, and Romberg tests.
- `tests/integration/test_quad_adaptive_transforms.py`: JIT, VMAP, stop-gradient, payload, support-matrix, and trace-structure contracts.
- `tests/validation/test_quad_adaptive_reference.py`: analytic and independent high-precision benchmarks.
- `docs/20-methods/approximation-integration/adaptive-quadrature.md`: current researcher-first method guide.
- `docs/50-api/approximation-integration/quad.md`: complete adaptive API and support matrix.
- project roadmaps, SOTA/scorecard, contract registry, evidence index, route manifest, and status surfaces as required by repository ratchets.

---

### Task 0: Harden the A1 tanh-sinh lattice before adaptive reuse

**Files:**
- Modify: `src/jaxstro/quad/_tanh_sinh.py`
- Modify: `src/jaxstro/quad/fixed.py` only if inactive-lane safety requires it
- Modify: `docs/20-methods/approximation-integration/quadrature.md`
- Modify: `tests/unit/quad/test_tanh_sinh.py`
- Modify: `tests/unit/quad/test_fixed.py`

**Interfaces:**
- Preserves `TanhSinhRule(level)` and `tanh_sinh_rule_data(rule) -> FixedRuleData`.
- Preserves returned integral payload shapes; the quadrature-owned leading
  integrand node-batch length may increase from the former
  `6 * 2**level + 1` construction.
- Adds an internal representability-aware constructor that exposes candidate
  indices, active masks, coarse-to-fine mappings, parameter coordinates, and
  terminal-tail metadata for A2.

- [x] **Step 1: Write failing truncation, nesting, and representability tests**

Under float32 and float64, require active nodes to be finite, strictly interior,
unique, symmetric, and nested across adjacent levels, with finite strictly
positive weights. Require inactive private candidates to have zero weight and
a finite safe coordinate. Test exact active counts and shell indices from
checked expected fixtures, plus explicit coarse-to-fine mappings when the outer
index count changes. Add regression
sequences for endpoint exponents `0.5`, `0.9`, and `0.99`, plus right-infinite
and full-line tails. Demonstrate RED from the old false-convergence sequence:
shrinking mesh differences must not be the only visible evidence while endpoint
mass remains unresolved. Add a parameter-gradient regression proving that a
nonfinite derivative at the private inactive sentinel cannot enter the compact
fixed-rule path. Ratchet several levels with the same terminal coordinate:
terminal evidence and status must not converge merely because $h_\ell$ halves.
In particular, float64 levels one through three must prove that unchanged
terminals at levels one and two do not block the farther node exposed at level
three.

- [x] **Step 2: Verify RED**

Run: `JAX_ENABLE_X64=1 env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad/test_tanh_sinh.py tests/unit/quad/test_fixed.py -k 'tanh_sinh or representable or truncation'`

- [x] **Step 3: Implement the nested active lattice**

Use cached host-side NumPy/Python construction for compact public rule constants
and static candidate arrays plus JAX array operations in the adaptive hot path.
Grow the parameter extent toward the dtype-derived strictly interior limit.
Preserve every active coarse even-index coordinate in the fine active set;
continue scanning after rejecting or masking a colliding new coordinate rather
than clipping or terminating. Expose the active-node count and terminal
transformed Jacobian data internally. Return only compact active arrays from the
public fixed constructor; padded masks and sentinels belong exclusively to the
private adaptive constructor. Add a JIT/JAXPR regression proving compact setup
introduces no dynamic-shape primitive into the traced evaluator.

- [x] **Step 4: Re-run complete A1 gates and document the correction**

Update the fixed-method page to explain representability masks, outer-tail
limitations, and why fixed level agreement alone is not an error certificate.
Run all A1 tanh-sinh, fixed, compatibility, transform, and reference tests under
both active precision policies.

- [x] **Step 5: Dispatch a focused read-only numerical reviewer**

The reviewer must inspect nesting, unique active nodes, endpoint behavior,
float32/float64 masks, fixed-rule numerical changes, and whether the new
metadata is sufficient for A2 tail evidence. Resolve all Critical and Important
findings before Task 1.

- [x] **Step 6: Commit**

```bash
git add src/jaxstro/quad/_tanh_sinh.py src/jaxstro/quad/fixed.py docs/20-methods/approximation-integration/quadrature.md tests/unit/quad/test_tanh_sinh.py tests/unit/quad/test_fixed.py
git commit -m "fix(quad): harden tanh-sinh representability"
```

### Task 1: Freeze adaptive method configurations and the public A2 surface

**Files:**
- Create: `src/jaxstro/quad/methods.py`
- Modify: `src/jaxstro/quad/__init__.py`
- Test: `tests/unit/quad/test_methods.py`
- Test: `tests/unit/quad/test_import_surface.py`

**Interfaces:**
- Produces the five frozen configuration types with static PyTree metadata.
- Extends the canonical public inventory without adding a legacy alias.

- [x] **Step 1: Write failing constructor, PyTree, validation, and inventory tests**

Test accepted defaults, all Gauss-Kronrod pairs, rejected unsupported pairs,
Clenshaw-Curtis `2^k + 1` structure, nonnegative adaptive-tanh-sinh levels,
Romberg and Romberg-tanh-sinh levels at least one, keyword signatures,
immutability, empty dynamic leaves, static metadata, and complete exports.

- [x] **Step 2: Verify RED**

Run: `JAX_ENABLE_X64=1 env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad/test_methods.py tests/unit/quad/test_import_surface.py`

Expected: adaptive configuration imports fail.

- [x] **Step 3: Implement minimal frozen records**

Use registered dataclass PyTrees. Reject booleans where integers are required.
Validate `GaussKronrod.pair in {15, 21, 31, 41, 51, 61}`,
`AdaptiveClenshawCurtis.initial_order = 2^k + 1 >= 5`, a nonnegative level
for adaptive tanh-sinh, and `initial_level >= 1` for both Romberg types.

- [x] **Step 4: Run focused tests and static checks**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad/test_methods.py tests/unit/quad/test_import_surface.py`

Run: `env -u VIRTUAL_ENV uv run --no-sync ruff check src/jaxstro/quad tests/unit/quad && env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro/quad`

- [x] **Step 5: Commit**

```bash
git add src/jaxstro/quad/methods.py src/jaxstro/quad/__init__.py tests/unit/quad/test_methods.py tests/unit/quad/test_import_surface.py
git commit -m "feat(quad): add adaptive method configurations"
```

### Task 2: Add all canonical Gauss-Kronrod pairs and the stabilized local estimator

**Files:**
- Create: `src/jaxstro/quad/_gk.py`
- Create: `src/jaxstro/quad/_gk_data.py` through the fixture owner script
- Create: `tests/unit/quad/test_gk.py`
- Create: `tests/validation/test_quad_gk_tables.py`
- Create: `tests/fixtures/quadpack/dqk15.f`
- Create: `tests/fixtures/quadpack/dqk21.f`
- Create: `tests/fixtures/quadpack/dqk31.f`
- Create: `tests/fixtures/quadpack/dqk41.f`
- Create: `tests/fixtures/quadpack/dqk51.f`
- Create: `tests/fixtures/quadpack/dqk61.f`
- Create: `tests/fixtures/quadpack/gk-reference.json`
- Create: `scripts/build_quadpack_gk_fixture.py`

**Interfaces:**
- Produces `gauss_kronrod_data(method)` and a payload-capable local embedded-rule estimate.
- Sources every constant and rescaling coefficient from named Netlib QUADPACK files.

- [x] **Step 1: Capture primary-source provenance before constants**

Check the exact Netlib source files into `tests/fixtures/quadpack/` with source
URLs, retrieval date, and SHA-256 digests. Add a deterministic parser/emit-check
script that extracts the data statements into an offline validation artifact.
Do not add a runtime downloader or a second hand-copied expected table.

Run: `env -u VIRTUAL_ENV uv run --no-sync python scripts/build_quadpack_gk_fixture.py --emit`

Run: `env -u VIRTUAL_ENV uv run --no-sync python scripts/build_quadpack_gk_fixture.py --check`

- [x] **Step 2: Write failing table-invariant and exactness tests**

For all six pairs, test node count, symmetry, strict ordering, positive Kronrod weights, aligned embedded weights, mass two, Gauss moments through degrees 13/19/29/39/49/59, and Kronrod moments through degrees 23/31/47/61/77/91 under float64 tolerances declared per pair.

- [x] **Step 3: Verify RED**

Run: `JAX_ENABLE_X64=1 env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad/test_gk.py tests/validation/test_quad_gk_tables.py`

Expected: `_gk` imports fail.

- [x] **Step 4: Implement literal data and the QUADPACK-style estimator**

Store immutable Python tuple literals and convert them to the active JAX dtype. Expand symmetry with array operations. Evaluate all nodes in one integrand call. Return high value, payload error, absolute-integral proxy, mean-deviation proxy, nonfinite flag, and roundoff-floor flag. Use componentwise magnitudes for real and complex payloads.

- [x] **Step 5: Validate independently and compile**

Compare runtime nodes and weights with the deterministically parsed checked-in
Netlib provenance fixture and, secondarily, SciPy's current fixtures where
available. Analytic moments, not the parsed fixture, are the independent
numerical oracle. Add a freshness regression proving a mutated source fixture
or generated JSON fails `--check`.
Test the exact QUADPACK rescaling branches for zero, constant, complex,
tiny-magnitude, float32, and float64 payloads against an independent scalar
translation. Test JIT and no order-dependent Python-loop unrolling.

Run: `JAX_ENABLE_X64=1 env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad/test_gk.py tests/validation/test_quad_gk_tables.py`

- [x] **Step 6: Commit**

```bash
git add src/jaxstro/quad/_gk.py tests/unit/quad/test_gk.py tests/validation/test_quad_gk_tables.py tests/fixtures/quadpack scripts/build_quadpack_gk_fixture.py
git commit -m "feat(quad): add Gauss-Kronrod pair family"
```

### Task 3: Build the shared transformed-integrand and reference-partition substrate

**Files:**
- Create: `src/jaxstro/quad/_adaptive.py`
- Create: `src/jaxstro/quad/_integrand.py` for behavior-identical shared node-axis helpers
- Create: `tests/unit/quad/test_adaptive_substrate.py`
- Modify: `src/jaxstro/quad/transforms.py`
- Modify: `src/jaxstro/quad/fixed.py` only to extract genuinely shared, behavior-identical integrand helpers if needed

**Interfaces:**
- Produces normalized reference regions, finite-breakpoint conversion, transformed-node evaluation, density application, and fixed-shape local records.
- Reuses A1 domain maps and weighted-density semantics without duplicating them.

- [x] **Step 1: Write failing reference-map, density, orientation, and payload tests**

Test finite/reversed intervals, finite breakpoints, all three improper domain types, weighted density exactly once, explicit `args`, scalar/vector/complex payloads, invalid breakpoint masks, and zero-width payload inference.

- [x] **Step 2: Verify RED**

Run: `JAX_ENABLE_X64=1 env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad/test_adaptive_substrate.py`

- [x] **Step 3: Implement one transformed-integrand owner**

Convert physical finite breakpoints to increasing normalized reference coordinates. Compose local reference, global domain, local Jacobian, global Jacobian, orientation, and optional general density. Preserve the leading node axis and fixed payload shape. Return explicit validity and nonfinite flags rather than raising from traced values.

- [x] **Step 4: Prove A1 did not drift**

Run: `JAX_ENABLE_X64=1 env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad/test_adaptive_substrate.py tests/unit/quad/test_fixed.py tests/integration/test_quad_fixed_transforms.py`

- [x] **Step 5: Commit**

```bash
git add src/jaxstro/quad/_adaptive.py src/jaxstro/quad/_integrand.py src/jaxstro/quad/transforms.py src/jaxstro/quad/fixed.py tests/unit/quad/test_adaptive_substrate.py
git commit -m "feat(quad): add adaptive reference substrate"
```

### Task 4: Implement the fixed-capacity h-adaptive controller

**Files:**
- Modify: `src/jaxstro/quad/_adaptive.py`
- Create: `tests/unit/quad/test_adaptive_controller.py`

**Interfaces:**
- Consumes any static local estimator with a fixed node cost.
- Produces deterministic region selection, global value/error accounting, status, and exact work evidence.

- [x] **Step 1: Write failing controller-state and status tests**

Use a synthetic local estimator with known values/errors. Test initial segment accounting, lowest-index tie breaking, parent replacement plus child append, exact refinements/evaluations/active regions, convergence, evaluation exhaustion, region exhaustion, nonfinite precedence, invalid input precedence, midpoint collapse, payload error summation, and error-norm selection.

- [x] **Step 2: Verify RED**

Run: `JAX_ENABLE_X64=1 env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad/test_adaptive_controller.py`

- [x] **Step 3: Implement one `jax.lax.while_loop` controller**

Allocate all region arrays from static `max_regions`. Batch initial segments and
two children with JAX transforms. Before every split, check the complete node
cost and region slot. After an accepted split, recompute the global value and
payload error from the masked active-region arrays; do not use
`global - parent + children`, which can permanently lose a small untouched
region that was rounded out of an earlier sum. Select priorities with inactive
entries set to negative infinity and preserve deterministic first-index ties.

- [x] **Step 4: Prove the trace is bounded structurally**

Compare serialized JAXPR primitive counts across small and large capacities. Array shapes may change; the number of controller-body copies and integrand primitives must not grow with `max_regions` or `max_evaluations`.

- [x] **Step 5: Run focused tests and commit**

Run: `JAX_ENABLE_X64=1 env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad/test_adaptive_controller.py`

```bash
git add src/jaxstro/quad/_adaptive.py tests/unit/quad/test_adaptive_controller.py
git commit -m "feat(quad): add fixed-capacity adaptive controller"
```

### Task 5: Wire adaptive Gauss-Kronrod through `quad.integrate`

**Files:**
- Create: `src/jaxstro/quad/adaptive.py`
- Modify: `src/jaxstro/quad/__init__.py`
- Create: `tests/unit/quad/test_integrate_gk.py`
- Create: `tests/integration/test_quad_adaptive_transforms.py`

**Interfaces:**
- Produces the public `integrate` evaluator for all six Gauss-Kronrod methods.
- Freezes public validation and stopped-result assembly reused by later methods.

- [x] **Step 1: Write failing public scalar, payload, breakpoint, and status tests**

Cover polynomial/exponential integrals, reversed orientation, multiple
breakpoints, weighted measures, explicit args, complex payloads, tolerance
evidence, every work field, every supported pair, structural pairing errors,
complex/nonscalar tolerance rejection, dynamic invalid input, nonfinite
integrands, both capacity statuses, deterministic stagnation and adjacent-endpoint
roundoff, and valid-versus-invalid zero-width behavior.

- [x] **Step 2: Write failing JIT, VMAP, and stop-gradient tests**

Test compiled bounds/args/tolerances, vmap over explicit parameter batches, and forward/reverse gradients equal to exact zero for every result leaf under `gradient="stop"`. Reject `gradient="replay"` eagerly with an A3-directed message.

- [x] **Step 3: Verify RED**

Run: `JAX_ENABLE_X64=1 env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad/test_integrate_gk.py tests/integration/test_quad_adaptive_transforms.py`

- [x] **Step 4: Implement public validation, dispatch, and result assembly**

Use static Python dispatch on the method type and call the shared controller. Validate structural capacities before allocating. Construct `QuadError`, scalar tolerance, scalar status code, and `QuadWork`; explicitly stop all leaves. Do not catch or relabel user Python exceptions that occur eagerly outside traced numerical evaluation.

- [x] **Step 5: Run the first adaptive checkpoint**

Run all Tasks 1-5 tests plus A1 fixed and compatibility tests. Dispatch a fresh read-only subagent to inspect table correctness, controller invariants, status precedence, work counts, transform contracts, and absence of runtime Python loops. Resolve all Critical and Important findings from failing regressions.

- [x] **Step 6: Commit**

```bash
git add src/jaxstro/quad/adaptive.py src/jaxstro/quad/__init__.py tests/unit/quad/test_integrate_gk.py tests/integration/test_quad_adaptive_transforms.py
git commit -m "feat(quad): add adaptive Gauss-Kronrod integration"
```

### Task 6: Add nested adaptive Clenshaw-Curtis

**Files:**
- Modify: `src/jaxstro/quad/_adaptive.py`
- Modify: `src/jaxstro/quad/adaptive.py`
- Create: `tests/unit/quad/test_adaptive_clenshaw_curtis.py`

**Interfaces:**
- Reuses the A1 stable cosine construction and the shared h-adaptive controller.
- Counts only high-rule nodes because the low rule is an exact subset.

- [x] **Step 1: Write failing nesting, work, and endpoint tests**

Test high/low node identity, explicit inequality of low weights and subset high
weights, polynomial exactness, no duplicated low evaluations, raw refinement
difference and summation-floor behavior, smooth finite integrals, breakpoint
localization, vector payloads, endpoint-nonfinite failure, capacity states, and
`REFINEMENT_DIFFERENCE` evidence.

- [x] **Step 2: Verify RED**

Run: `JAX_ENABLE_X64=1 env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad/test_adaptive_clenshaw_curtis.py`

- [x] **Step 3: Implement the local pair adapter and dispatch**

Construct both low and high A1 `ClenshawCurtisRule` data through the same
cosine owner. Evaluate once at the high nodes. Reduce high values with high
weights and even-index values with the separately constructed low weights. Do
not create a second cosine implementation.

- [x] **Step 4: Run focused and A1 regression tests**

Run: `JAX_ENABLE_X64=1 env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad/test_adaptive_clenshaw_curtis.py tests/unit/quad/test_chebyshev_rules.py tests/integration/test_quad_adaptive_transforms.py`

- [x] **Step 5: Commit**

```bash
git add src/jaxstro/quad/_adaptive.py src/jaxstro/quad/adaptive.py tests/unit/quad/test_adaptive_clenshaw_curtis.py
git commit -m "feat(quad): add adaptive Clenshaw-Curtis"
```

### Task 7: Add adaptive tanh-sinh on every A2 domain

**Files:**
- Modify: `src/jaxstro/quad/_adaptive.py`
- Modify: `src/jaxstro/quad/adaptive.py`
- Create: `tests/unit/quad/test_adaptive_tanh_sinh.py`

**Interfaces:**
- Reuses A1 tanh-sinh nesting and every A1 domain map.
- Uses the same local reference partition for finite and improper domains.

- [x] **Step 1: Write failing nested-work and domain-family tests**

Cover finite smooth and endpoint-singular integrals with exponents `0.5`, `0.9`,
and `0.99`, finite breakpoints,
right-infinite exponential tails, left-infinite tails, full-line Gaussian tails,
reversed finite orientation, weighted measures, payloads, discretization versus
outer-tail evidence, false-convergence prevention, nonfinite active mapped
contributions, capacity states, and exact active-entry work counts.
Under float64, predeclare observed-error convergence envelopes for exponents
`0.5` and `0.9`; exponent `0.99` must either meet its declared envelope or
return `ROUNDOFF_LIMITED` with tail norm above tolerance. Under float32,
exponents `0.9` and `0.99` may take that same honest roundoff path, but may not
pass merely because adjacent levels agree. Ratchet the exact error aggregation,
active counts, shell indices, and terminal transition.

- [x] **Step 2: Verify RED**

Run: `JAX_ENABLE_X64=1 env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad/test_adaptive_tanh_sinh.py`

- [x] **Step 3: Implement one high-level nested adapter**

Use the Task 0 active-lattice owner to construct the adjacent-level union,
explicit coarse-to-fine mapping, active masks, and tail metadata. Physically
skip inactive entries for an unbatched solve and count only active logical
entries. Sanitize every inactive integrand, density, mapped-Jacobian,
contribution, and nonfinite-flag lane before selection/reduction so VMAP
lowering cannot poison a result. Compose each local region with `map_domain`;
never split a physical improper interval directly. Treat exhausted local
reference extent as priority evidence, not an immediate roundoff status; allow
the shared controller to bisect the reference region.

- [x] **Step 4: Run focused transform and fixed-rule regressions**

Run: `JAX_ENABLE_X64=1 env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad/test_adaptive_tanh_sinh.py tests/unit/quad/test_tanh_sinh.py tests/integration/test_quad_adaptive_transforms.py`

- [x] **Step 5: Commit**

```bash
git add src/jaxstro/quad/_adaptive.py src/jaxstro/quad/adaptive.py tests/unit/quad/test_adaptive_tanh_sinh.py
git commit -m "feat(quad): add adaptive tanh-sinh"
```

### Task 8: Add fixed-capacity Romberg and nested tanh-sinh refinement

**Files:**
- Create: `src/jaxstro/quad/_romberg.py`
- Modify: `src/jaxstro/quad/adaptive.py`
- Create: `tests/unit/quad/test_romberg.py`

**Interfaces:**
- Produces a standard Richardson engine and a separate nested tanh-sinh level
  engine behind one public dispatch boundary.
- Reuses active node values and evaluates each level through fixed maximum lane
  arrays with explicit masks.
- Documents `RombergTanhSinh` as Romberg-style global level refinement with no
  Richardson extrapolation, distinct from h-adaptive `AdaptiveTanhSinh`.

- [x] **Step 1: Write failing Richardson, reuse, and work tests**

Test standard table identities on polynomials, the mutation-resistant addition
in the base recurrence, exact `2^n + 1` standard work, explicit zero-based level
semantics, exact active-node tanh-sinh work, fixed-capacity lane masking,
successive-diagonal standard error, successive-level plus tail tanh-sinh error,
`levels`/`refinements` semantics, scalar/vector/complex payloads, tolerance
convergence, max-evaluation exit, nonfinite exit, stagnation/representability
roundoff, unsupported breakpoints, and improper-domain support only for the
tanh-sinh variant.

- [x] **Step 2: Verify RED**

Run: `JAX_ENABLE_X64=1 env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad/test_romberg.py`

- [x] **Step 3: Implement the fixed-capacity refinement engines**

Derive the static maximum level from `max_evaluations`. Allocate the complete
standard table, maximum lane arrays, and tanh-sinh active masks from that bound.
Use `jax.lax.map` plus per-lane `jax.lax.cond` to skip inactive standard and
tanh-sinh lanes physically in an unbatched solve. Advance levels with
`jax.lax.while_loop`; use `jax.lax.fori_loop` only for standard Richardson
columns. Stop before a level whose complete active logical node set exceeds the
budget. Do not apply Richardson columns to Romberg-tanh-sinh.

- [x] **Step 4: Prove structural trace bounds and compile behavior**

JAXPR tests must show one outer loop body, one fixed lane map, and, for standard
Romberg only, one Richardson loop regardless of static evaluation budget. Test
JIT and VMAP on both variants, stopped gradients, logical per-lane work under
VMAP, and direct `lax.map` guidance for cost-sensitive batches.

- [x] **Step 5: Run focused tests and commit**

Run: `JAX_ENABLE_X64=1 env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad/test_romberg.py tests/integration/test_quad_adaptive_transforms.py`

```bash
git add src/jaxstro/quad/_romberg.py src/jaxstro/quad/adaptive.py tests/unit/quad/test_romberg.py tests/integration/test_quad_adaptive_transforms.py
git commit -m "feat(quad): add Romberg refinement family"
```

### Task 9: Build independent adaptive validation and estimator-calibration evidence

**Files:**
- Create: `tests/validation/test_quad_adaptive_reference.py`
- Modify: `docs/60-validation/validation.md`
- Modify generated evidence sources only through their owner scripts

**Interfaces:**
- Establishes an explicit numerical envelope without making a universal error-bound or superiority claim.

- [x] **Step 1: Predeclare float64 thresholds and benchmark meanings**

Record per-family tolerances before running comparisons. Include analytic polynomials/exponentials, peaked Lorentzian profiles, declared discontinuities, endpoint algebraic singularities, semi-infinite exponential tails, full-line Gaussians, complex oscillation over a modest finite interval, and vector payloads. State which methods are expected to succeed and why.

- [x] **Step 2: Write independent analytic and secondary comparison tests**

Use analytic values as the independent acceptance oracle. SciPy and Quadax may
be secondary comparison checks but cannot be the only oracle for copied
QUADPACK data, controller behavior, or error calibration. Do not add mpmath or
another development dependency in A2.

- [x] **Step 3: Add tolerance sweeps**

For each method, sweep at least three tolerances and record requested tolerance,
reported indicator norm, observed error, status, evaluations, refinements,
regions, and levels. Before execution, assign every benchmark either a numeric
observed-error envelope when `CONVERGED` is expected or an explicit acceptable
nonconverged status set. The indicator need not upper-bound every observed
error, but a benchmark cannot pass on “stable convergence” alone.

- [x] **Step 4: Add failure-envelope tests**

Include a missed narrow feature without a breakpoint, a nonintegrable or
nonfinite case, exhausted budgets, adjacent representable endpoints, and an
endpoint singularity sent deliberately to Clenshaw-Curtis. For the missed
feature, explicitly permit and ratchet false estimator convergence: preserve
`CONVERGED` if the named estimator passes while asserting that observed error
exceeds tolerance and that the limitation is documented. Do not require a
failure heuristic the controller does not own.

- [x] **Step 5: Run validation and commit**

Run: `JAX_ENABLE_X64=1 env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/validation/test_quad_gk_tables.py tests/validation/test_quad_adaptive_reference.py`

```bash
git add tests/validation/test_quad_adaptive_reference.py docs/60-validation/validation.md docs/validation
git commit -m "test(quad): validate adaptive quadrature envelope"
```

### Task 10: Publish current adaptive method and API documentation

**Files:**
- Rewrite: `docs/20-methods/approximation-integration/adaptive-quadrature.md`
- Modify: `docs/20-methods/approximation-integration/quadrature.md`
- Modify: `docs/50-api/approximation-integration/quad.md`
- Modify: `docs/70-project/development/future-capabilities-roadmap.md`
- Modify: `docs/70-project/development/numerical-methods-roadmap.md`
- Modify: `docs/70-project/development/sota-assessment.md`
- Modify: `docs/70-project/development/package-assessment-scorecard.md`
- Modify: `src/jaxstro/contracts/registry.py`
- Modify generated contract/evidence artifacts through owner scripts
- Test: `tests/integration/test_method_page_contract.py`
- Test: `tests/integration/test_grouped_api_reference.py`
- Test: `tests/unit/test_contract_manifests.py`

**Interfaces:**
- Makes the adaptive page current and Jaxstro-owned.
- Keeps Quadax as a comparator rather than a delegated owner.

- [ ] **Step 1: Write failing current-capability and claim-boundary ratchets**

Require every method/configuration name, `quad.integrate`, support matrix, status precedence, work definitions, error-kind semantics, primal-only stop policy, exact-evidence disclaimer, cost model, validation links, and absence of the stale “Quadax owns” claim.

- [ ] **Step 2: Verify RED**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/integration/test_method_page_contract.py tests/integration/test_grouped_api_reference.py tests/unit/test_contract_manifests.py`

- [ ] **Step 3: Write the researcher-first adaptive guide**

Follow the ten-stage method-page sequence. Include labeled LaTeX derivations for the local/global error account, tolerance, Gauss-Kronrod embedding and stabilization, nested Clenshaw-Curtis, double-exponential refinement, Romberg recurrence, and work scaling. Use MyST `important`, `warning`, `note`, `tip`, and `seealso` only for their intended semantic roles. Include executable raw-array examples for each method family and a method-choice table for new researchers.

- [ ] **Step 4: Update API, roadmaps, SOTA, scorecard, contracts, and evidence together**

Mark A2 runtime methods current while keeping replay derivatives, quantities, formal comparisons, sibling migration, and later-dimensional methods planned. Register a callable contract for `jaxstro.quad.integrate` with exact evidence links and limitations.

- [ ] **Step 5: Regenerate and verify**

Run: `env -u VIRTUAL_ENV uv run --no-sync python scripts/build_contract_registry.py --emit`

Run: `env -u VIRTUAL_ENV uv run --no-sync python scripts/build_evidence_index.py --emit`

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/integration/test_method_page_contract.py tests/integration/test_grouped_api_reference.py tests/unit/test_contract_manifests.py`

Run: `DOCS_APP_PORT=4381 DOCS_SERVER_PORT=4382 bash scripts/check_docs.sh`

- [ ] **Step 6: Commit**

```bash
git add src/jaxstro/contracts/registry.py docs tests/integration/test_method_page_contract.py tests/integration/test_grouped_api_reference.py tests/unit/test_contract_manifests.py
git commit -m "docs(quad): publish adaptive quadrature contracts"
```

### Task 11: Complete A2 checkpoint review and verification

**Files:**
- Modify: `STATUS.md`
- Modify: this plan's checkboxes

**Interfaces:**
- Produces a verified A2 state suitable for a fresh A3 plan, but does not begin A3.

- [ ] **Step 1: Run the exact focused A2 gate**

Run: `JAX_ENABLE_X64=1 env -u VIRTUAL_ENV uv run --no-sync pytest -q tests/unit/quad tests/integration/test_quad_adaptive_transforms.py tests/integration/test_quad_fixed_transforms.py tests/integration/test_quad_compatibility.py tests/validation/test_quad_gk_tables.py tests/validation/test_quad_adaptive_reference.py`

Expected: all focused tests pass.

- [ ] **Step 2: Run static and freshness checks**

Run Ruff check/format over `src` and `tests`, MyPy over `src`, all four generated
registry freshness checks, the QUADPACK fixture freshness check, and
`git diff --check`.

Run: `env -u VIRTUAL_ENV uv run --no-sync python scripts/build_quadpack_gk_fixture.py --check`

- [ ] **Step 3: Dispatch a fresh read-only complete A2 reviewer**

The reviewer must inspect primary-source constants, numerical estimators, exact work counts, controller invariants, status precedence, domain/measure pairings, JAX trace structure, stopped AD, validation independence, documentation equations, and public claim calibration. Resolve every Critical and Important finding from a failing regression. Record Minor findings explicitly.

- [ ] **Step 4: Run full repository and strict rendered-documentation gates**

Run: `env -u VIRTUAL_ENV uv run --no-sync pytest -q`

Run: `DOCS_APP_PORT=4381 DOCS_SERVER_PORT=4382 bash scripts/check_docs.sh`

- [ ] **Step 5: Update status and commit verification evidence**

Record exact commits, focused/full counts, checkpoint dispositions, supported methods, reference sources, route count, and exclusions.

```bash
git add STATUS.md docs/superpowers/plans/2026-07-15-jaxstro-quad-phase-a2-adaptive-rules.md
git commit -m "docs(quad): record Phase A2 verification"
```

## Stop conditions

Stop without declaring A2 complete if:

- any Netlib table or declared exactness degree is unverified;
- any controller work counter differs from executed node counts;
- any status depends on non-deterministic tie selection or ambiguous precedence;
- a Python loop advances regions, levels, nodes, evaluation batches, or Richardson columns;
- a nonfinite contribution is silently dropped or converted to convergence;
- a result error is described as a proof of true error;
- reverse- or forward-mode AD differentiates the adaptive loop in A2;
- any A1 compatibility identity or byte-level Hermite contract changes;
- quantities, replay derivatives, sibling migration, deprecation, publication, or live-site state enters the diff; or
- any Critical or Important checkpoint finding remains unresolved.

## Phase A3 handoff

Only after Task 11 passes, write a fresh Phase A3 plan against the verified A2 primal interfaces. A3 owns normalized-reference replay, `gradient="replay"`, moving-bound and explicit-parameter derivative validation, stopped evidence leaves, raw/quantity normalization, result-unit propagation, and separate derivative evidence. Comparisons, sibling migrations, deprecations, publication, and multidimensional methods remain outside A3.
