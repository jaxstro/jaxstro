---
title: Jaxstro.quad Phase B multidimensional integration design
description: Approved architecture for hyperrectangles, deterministic cubature, sparse grids, randomized quasi-Monte Carlo, replay derivatives, quantities, and validation.
---

# Jaxstro.quad Phase B multidimensional integration design

**Status:** Approved section by section on 2026-07-17; awaiting final written-spec
review before implementation planning.

**Scope:** One umbrella architecture delivered through separately planned and
reviewed B0 through B4 stages.

## Decision summary

Phase B extends the stable Phase A one-dimensional contracts to general
multidimensional integration over finite, static-dimensional hyperrectangles.
It adds four distinct method families:

1. fixed tensor-product quadrature;
2. p-adaptive tensor Clenshaw-Curtis and h-adaptive Genz-Malik cubature;
3. isotropic, anisotropic, and dimension-adaptive Smolyak sparse grids; and
4. deterministic and randomized Sobol integration.

The primary public workflow remains `quad.integrate`. It is a thin dispatcher
over separate family controllers and contains no numerical algorithm.

The design preserves one fixed-shape `QuadResult`, but it does not collapse
different evidence meanings. Embedded cubature error, tensor refinement
difference, sparse-grid surplus, replicate standard error, and randomized
confidence half-width remain different `ErrorKind` values.

Phase B supports replay differentiation of the accepted numerical formula.
Controller choices, region topology, sparse-index selection, randomization,
sample counts, statuses, and error evidence remain stopped.

Raw multidimensional integration continues to use coordinate-last JAX arrays.
Opt-in quantity mode adds explicit per-axis units through `Axis` and
`CoordinatePoint`; it does not require or imply downstream quantity adoption.

## Goals

Phase B must:

- provide one coherent multidimensional integration API;
- preserve method-specific mathematical and statistical semantics;
- support JIT, VMAP, and first-order replay differentiation where claimed;
- retain fixed-shape, capacity-bounded JAX execution;
- support scalar, vector, array, real, and documented complex outputs in
  deterministic families, with method-specific restrictions stated explicitly;
- support heterogeneous physical coordinate dimensions in opt-in quantity mode;
- provide deterministic work accounting and reproducible randomization;
- validate mathematical truth before comparing performance;
- benchmark only matched or explicitly labeled comparison lanes; and
- teach new researchers how to choose, run, audit, and interpret each method.

## Non-goals

Phase B does not:

- add simplex, spherical, manifold, oscillatory, Hankel, or line-of-sight
  specialization APIs;
- implement general posterior inference, experimental-design policy, or
  scientific acceptance policy;
- migrate Progenax, Gravax, Startrax, Informax, or another sibling package;
- promote quantity support beyond alpha and opt-in;
- differentiate controller decisions, random keys, or uncertainty diagnostics;
- claim that an estimator threshold certifies the true integration error;
- claim universal performance superiority;
- add Quadax, SciPy, Tasmanian, Torchquad, or another comparator as a runtime
  dependency;
- push, publish, deploy, or change live documentation without separate approval;
  or
- implement all B0 through B4 stages in one monolithic change.

## Program structure

```text
B0: shared multidimensional contracts
 |
 +-- B1: tensor rules and adaptive cubature
 |
 +-- B2: Smolyak sparse grids
 |
 +-- B3: Sobol and randomized QMC
 |
 `-- B4: replay AD, quantities, comparisons, docs, and hardening
```

Every stage receives a separate implementation plan, test-first execution, and
independent review checkpoint. A later stage may extend an earlier private
owner, but it may not silently change an approved public semantic contract.

## Public workflow and ownership

The primary call is:

```python
result = quad.integrate(
    fun,
    domain,
    method=method,
    args=args,
    epsabs=epsabs,
    epsrel=epsrel,
    error_norm=error_norm,
    gradient="replay",
    **method_controls,
)
```

Randomized methods additionally require an explicit JAX key:

```python
result = quad.integrate(
    fun,
    domain,
    method=ScrambledSobol(...),
    key=key,
)
```

The facade owns:

- eager structural validation;
- raw-versus-quantity mode selection;
- shared domain normalization;
- integrand and argument normalization;
- result restoration; and
- dispatch to one static method family.

It does not own:

- region subdivision;
- tensor-level growth;
- sparse-index admissibility;
- Sobol point construction;
- randomization;
- statistical intervals; or
- method-specific stopping.

Expert family functions may remain importable, but documentation teaches
`quad.integrate` first.

# B0: shared multidimensional contracts

## Integrand convention

Raw numerical integrands use:

```python
def fun(x, args):
    ...
```

with:

```text
x.shape == (..., dimension)
```

The final axis always represents coordinates. Leading axes represent point,
replicate, parameter-batch, or other evaluator-owned batch dimensions.

The integrand may return a scalar or an array-valued payload. Output payload
axes are distinct from the coordinate axis. The evaluator records the point
axis explicitly and never guesses it from output rank.

`args` remains the only supported generic differentiable parameter container.
Closing over a differentiable parameter is unsupported.

## Hyperrectangle

The only native Phase B domain is:

```python
Hyperrectangle(lower, upper)
```

with:

```text
lower.shape == upper.shape == (dimension,)
dimension is static
```

The normalized reference domain is $[0,1]^d$. For axis $i$:

```{math}
x_i(t_i)
=
a_i + (b_i-a_i)t_i.
```

The Jacobian determinant is:

```{math}
J
=
\prod_{i=1}^{d}(b_i-a_i).
```

Per-axis orientation is preserved by the sign of $b_i-a_i$. Total orientation
is the product of all axis signs. If any axis has coincident bounds, the
integral has zero volume and returns the exact zero payload with zero work after
the output shape is established. Replay derivatives at a coincident bound are
outside the Phase B derivative contract because the zero-work primal shortcut
does not contain the lower-dimensional boundary integral needed for its
tangent. Replay fails closed with `INVALID_INPUT` and a nonfinite value tangent
at that boundary; `gradient="stop"` retains the exact zero shortcut.

Phase B requires finite bounds. Improper multidimensional domains, simplexes,
spheres, and other geometries remain Phase C work.

## Unified result

All Phase B methods return the existing fixed-shape:

```python
QuadResult(
    value=...,
    error=QuadError(
        estimate=...,
        norm=...,
        kind=...,
        confidence_level=...,
    ),
    tolerance=...,
    status=...,
    work=QuadWork(
        evaluations=...,
        refinements=...,
        active_regions=...,
        levels=...,
        replicates=...,
    ),
)
```

Evidence meanings are:

| Family | `ErrorKind` |
| --- | --- |
| Fixed tensor product | `UNAVAILABLE` |
| p-adaptive tensor product | `REFINEMENT_DIFFERENCE` |
| h-adaptive cubature | `EMBEDDED_RULE` |
| Sparse grid | `SPARSE_GRID_SURPLUS` |
| Fixed deterministic QMC | `UNAVAILABLE` |
| Fixed randomized QMC | `CONFIDENCE_INTERVAL_HALF_WIDTH` |
| Bounded adaptive randomized QMC | `CONFIDENCE_INTERVAL_HALF_WIDTH` |

A fixed tensor formula or fixed deterministic QMC estimate has no runtime error
estimator and returns `ERROR_ESTIMATE_UNAVAILABLE`, not `CONVERGED`. Its
`tolerance` records
$\max(\mathtt{epsabs},\mathtt{epsrel}\lVert I\rVert)$ for result-shape
consistency but does not imply that the threshold was tested. Fixed methods
accept and validate the shared tolerance arguments; they do not use them for
stopping. Their `confidence_level` is `NaN` because no confidence statement
applies.

`CONVERGED` always means that the named method-specific estimator met its
declared stopping policy. It never means that the true integral is certified
correct.

`MAX_INDICES` is appended to `QuadStatus`; existing numeric status values remain
unchanged.

## Work semantics

### Fixed tensor

```text
evaluations    = tensor-node count
refinements    = 0
active_regions = 0
levels         = 0
replicates     = 0
```

### Adaptive tensor

```text
evaluations    = unique logical point evaluations, including frontier evidence
refinements    = accepted per-axis level increments
active_regions = 0
levels         = maximum accepted per-axis level
replicates     = 0
```

### Adaptive cubature

```text
evaluations    = logical point evaluations
refinements    = accepted region splits
active_regions = final leaf-region count
levels         = deepest active subdivision depth
replicates     = 0
```

### Sparse grid

```text
evaluations    = unique coalesced nodes
refinements    = accepted multi-index additions
active_regions = 0
levels         = maximum accepted total multi-index level
replicates     = 0
```

### Deterministic QMC

```text
evaluations    = final points
refinements    = 0
active_regions = 0
levels         = log2(final points)
replicates     = 0
```

### Randomized QMC

```text
evaluations    = replicates * final points per replicate
refinements    = accepted point-level or replicate-count expansions
active_regions = 0
levels         = log2(final points per replicate)
replicates     = independent randomizations
```

Sobol prefixes are nested and reused, so adaptive expansion does not count
previous prefix points again.

# B1: deterministic tensor and cubature methods

## Method set

B1 implements:

1. `TensorProduct`;
2. `AdaptiveTensorClenshawCurtis`; and
3. `AdaptiveCubature(rule=GenzMalik())`.

The validated B1 dimensional envelope is 2 through 8. Dimension 1 uses the
existing Phase A methods. Dimensions above 8 use B2 or B3 in the initial public
contract.

## Fixed tensor products

`TensorProduct` accepts one replicated one-dimensional rule or one static rule
per coordinate axis. Compatible Phase A Gaussian, Clenshaw-Curtis, Fejer, and
tanh-sinh rules are reused rather than reimplemented.

For $n_i$ nodes on axis $i$:

```{math}
N_{\mathrm{tensor}}
=
\prod_{i=1}^{d} n_i.
```

The exact count is computed eagerly. A declaration exceeding
`max_evaluations` fails before node materialization or compilation.

## p-adaptive tensor Clenshaw-Curtis

The controller maintains an anisotropic level vector
$\boldsymbol{\ell}=(\ell_1,\ldots,\ell_d)$ and reuses all nested
Clenshaw-Curtis nodes. Each axis $i$ has a one-step frontier candidate with
directional evidence:

```{math}
D_i
=
\left\|
I_{\boldsymbol{\ell}+\mathbf{e}_i}
-
I_{\boldsymbol{\ell}}
\right\|.
```

Its profit is $D_i/\max(1,\Delta N_i)$, where $\Delta N_i$ is the number of new
tensor nodes after exact nested reuse. The largest-profit axis is accepted;
ties use the lowest axis index. After acceptance, all directional candidates
are updated against the new tensor value.

Stopping uses the conservative frontier aggregate:

```{math}
E_{\mathrm{tensor}}
=
\sum_{i=1}^{d}D_i.
```

The error kind is `REFINEMENT_DIFFERENCE`. Capacity exhaustion returns
`MAX_EVALUATIONS`. The controller is intended for smooth, globally resolved,
low-dimensional integrands.

## h-adaptive Genz-Malik cubature

Each active hyperrectangle receives a fully symmetric higher-degree estimate,
an embedded lower-degree estimate, and axis-direction smoothness evidence.

Each refinement:

1. selects the active region with the largest normalized local error;
2. selects its least-smooth axis;
3. bisects that axis at its midpoint;
4. evaluates both child regions;
5. updates the global value and error evidence; and
6. stops on tolerance or declared capacity.

Region and axis ties use deterministic lexicographic ordering. The controller
uses a fixed-capacity region store. Logical work excludes padded inactive
storage.

The algorithm follows the mathematical description of Genz and Malik rather
than copying an external implementation.

# B2: Smolyak sparse grids

## Hierarchical construction

Let $Q_{\ell}$ be the nested one-dimensional Clenshaw-Curtis rule and:

```{math}
\Delta_{\ell}
=
Q_{\ell}-Q_{\ell-1},
\qquad
Q_0=0.
```

For multi-index $\boldsymbol{\ell}$:

```{math}
\Delta_{\boldsymbol{\ell}}
=
\Delta_{\ell_1}\otimes\cdots\otimes\Delta_{\ell_d}.
```

For a downward-closed set $\mathcal{I}$:

```{math}
A_{\mathcal{I}}[f]
=
\sum_{\boldsymbol{\ell}\in\mathcal{I}}
\Delta_{\boldsymbol{\ell}}[f].
```

## Policies

### Isotropic

```{math}
\sum_{j=1}^{d}(\ell_j-1)
\leq q.
```

### Anisotropic

For static positive weights $w_j$:

```{math}
\sum_{j=1}^{d}w_j(\ell_j-1)
\leq q.
```

Weights are algorithm configuration and are stopped.

### Dimension adaptive

A forward candidate is admissible only when all valid immediate backward
neighbors are accepted.

The profit is:

```{math}
P_{\boldsymbol{\ell}}
=
\frac{
\left\|\Delta_{\boldsymbol{\ell}}[f]\right\|
}{
\max(1,\Delta N_{\boldsymbol{\ell}})
}.
```

$\Delta N_{\boldsymbol{\ell}}$ counts genuinely new nodes. The highest-profit
candidate is accepted; ties are lexicographic.

The stopping evidence is the active-frontier surplus sum:

```{math}
E_{\mathrm{frontier}}
=
\sum_{\boldsymbol{\ell}\in\mathcal{F}}
\left\|\Delta_{\boldsymbol{\ell}}[f]\right\|.
```

It is `SPARSE_GRID_SURPLUS`, not a universal absolute error bound.

## Exact node coalescing

Nested nodes are never deduplicated by floating-point comparison. Every
one-dimensional Clenshaw-Curtis node receives a canonical reduced dyadic-angle
identity derived from its integer level and index. A multidimensional identity
is the tuple of the canonical per-axis identities.

Reference identities are coalesced before physical coordinates are created.
This provides exact reuse and deterministic work counts.

## Capacities and dimension

The static capacities are:

```text
max_indices
max_frontier
max_nodes
```

Sparse grids have no arbitrary public dimension cutoff. Impossible declared
capacity combinations fail eagerly. Initial performance claims are restricted
to dimensions 2 through 16; larger supported calls receive no general
efficiency claim.

# B3: Sobol and randomized quasi-Monte Carlo

## Direction-number owner

Jaxstro vendors the Joe-Kuo `new-joe-kuo-6.21201` direction-number source data,
its BSD-style license, source metadata, and checksum. The runtime generator is
implemented in Jaxstro and has no SciPy dependency.

The generator supports the table's declared 21,201 dimensions. Performance and
quality claims remain limited to tested regimes.

Sobol points use Gray-code ordering and power-of-two prefixes:

```{math}
N=2^m.
```

The integration API does not skip, thin, or request arbitrary non-power-of-two
prefix lengths.

`Sobol` is the fixed deterministic integration method. It returns
`ErrorKind.UNAVAILABLE`, `QuadStatus.ERROR_ESTIMATE_UNAVAILABLE`, a `NaN`
confidence level, the declared tolerance, and deterministic-QMC work semantics.

## Precision

Digital arithmetic is integer based. Public distinct-coordinate limits are:

```text
float32: at most 24 output bits
float64: at most 53 output bits
```

The bit depth is static. A request beyond the selected dtype's distinct dyadic
representation fails eagerly.

## Randomization

### Linear matrix plus shift

`LinearMatrixScramble` applies one random nonsingular lower-triangular binary
matrix per coordinate followed by an independent digital shift. It is the
practical default.

### Nested Owen

`OwenScramble` applies prefix-dependent nested binary digit permutations. It is
the higher-cost reference randomization. A keyed, stateless prefix identity
determines each required permutation reproducibly.

### Digital shift

`DigitalShift` exists for research comparison but is not the recommended
integration default.

No LMS implementation is labeled as Owen scrambling.

## Key ownership

The user supplies one explicit JAX key. Replicate $r$ uses:

```python
jax.random.fold_in(key, r)
```

Existing replicate identities therefore remain stable if the replicate
capacity grows. Keys, randomization, direction numbers, bit depth, sample count,
and replicate count are stopped.

## Fixed-look randomized QMC

`ScrambledSobol` evaluates one predeclared level with at least eight independent
replicates. The calibrated Phase B confidence-interval contract is restricted
to real scalar integrals. Vector, array, and complex payloads require a future
simultaneous-region or explicitly multiplicity-controlled contract and are
rejected by randomized integration methods in Phase B.

For replicate estimates $\widehat I_r$:

```{math}
\overline I
=
\frac{1}{R}\sum_{r=1}^{R}\widehat I_r,
```

```{math}
\widehat{\operatorname{SE}}
=
\sqrt{
\frac{1}{R(R-1)}
\sum_{r=1}^{R}
\left(\widehat I_r-\overline I\right)^2
}.
```

The fixed-look Student-t half-width is:

```{math}
H
=
t_{1-\alpha/2,R-1}
\widehat{\operatorname{SE}}.
```

The interval is inspected once. It is a model-based fixed-look interval, not a
nonasymptotic guarantee. The result is `CONVERGED` when $H$ is no larger than
the declared tolerance and `MAX_EVALUATIONS` otherwise because the fixed
evaluation budget has been exhausted.

## Bounded sequential QMC

`AdaptiveScrambledSobol` requires certified finite bounds $[A,B]$ on each
replicate integral estimate. Jaxstro may derive those bounds from pointwise
integrand bounds only for a finite nonnegative measure with known total mass
after applying the absolute domain orientation and quantity conversion. Signed
or otherwise unbounded measures require direct, unit-compatible estimate bounds
and cannot infer them from pointwise extrema.

At each inspected level, the controller uses an empirical Bernstein half-width
of the form:

```{math}
H_k
=
\sqrt{
\frac{2 V_R \log(2/\alpha_k)}{R}
}
+
\frac{
7(B-A)\log(2/\alpha_k)
}{
3(R-1)
},
```

where $V_R$ is the unbiased replicate sample variance:

```{math}
V_R
=
\frac{1}{R-1}
\sum_{r=1}^{R}
\left(\widehat I_r-\overline I\right)^2,
```

and:

```{math}
\alpha_k
=
\alpha
\frac{6}{\pi^2(k+1)^2}.
```

Thus:

```{math}
\sum_{k=0}^{\infty}\alpha_k
\leq\alpha.
```

At each fixed level, replicate estimates must remain independent across
randomizations. The same prefix within a replicate is extended and reused
across levels. The alpha-spending union bound does not require independence
between inspection levels and protects the simultaneous repeated-look claim.

The inspected schedule is a static monotone sequence of pairs:

```{math}
(m_0,R_0), (m_1,R_1), \ldots, (m_K,R_K),
```

where $N_k=2^{m_k}$, $m_{k+1}\geq m_k$, $R_{k+1}\geq R_k$, and at least one
inequality is strict. The schedule must contain replicate growth
$R_K>R_0$ because increasing points alone cannot shrink the range term. New
replicates use stable `fold_in` identities and are evaluated on the current
complete prefix; existing replicates extend only the new suffix. Exhausting the
predeclared schedule before convergence returns `MAX_EVALUATIONS`.

Without declared bounds, repeated standard-error ladders are diagnostic only.
They cannot drive a calibrated early `CONVERGED` status.

# B4.1: replay differentiation

## General rule

Phase B differentiates the accepted numerical formula, not the procedure that
selected it.

Only `QuadResult.value` carries a scientific tangent. Error estimates, norms,
confidence levels, tolerance, status, and work are stopped.

The public modes remain:

```python
gradient="replay"
gradient="stop"
```

## Family replay

### Fixed tensor

Ordinary JAX differentiation applies to the fixed tensor formula, but the
public result restoration stops every diagnostic tangent and exposes a
scientific tangent only for `QuadResult.value`.

### p-adaptive tensor

Replay freezes the accepted tensor level vector and reconstructs nodes and
weights from live outer bounds and parameters.

### h-adaptive cubature

Replay freezes leaf topology, normalized leaf bounds, split decisions, activity
masks, and accepted rule declarations. Physical coordinates and Jacobians are
reconstructed from live outer bounds.

### Sparse grid

Replay freezes the accepted multi-index set, coalesced reference identities,
combination coefficients, and activity masks. Physical coordinates, Jacobians,
integrand values, and supported parameters remain live.

### Randomized QMC

Replay freezes the realized direction-number points, randomization, replicate
identities, and accepted sample level. It differentiates the realized
randomized estimator with respect to live bounds and explicit integrand
parameters.

The primal error estimate never certifies derivative error.

## Derivative boundary

First-order replay claims require a differentiable integrand, smooth supported
domain maps, finite realized formulas, explicit parameters, and justification
for interchanging differentiation and integration.

Controller changes, status changes, capacity changes, discontinuities, unit
representation changes, and randomization changes are nonsmooth boundaries.

Higher derivatives remain unsupported.

# B4.2: heterogeneous quantity coordinates

## Axis declarations

Quantity-aware domains use:

```python
Hyperrectangle.from_axes(
    (
        Axis(lower_0, upper_0),
        Axis(lower_1, upper_1),
        ...
    )
)
```

Each axis has compatible finite scalar bounds and an explicit unit, including
dimensionless axes. Units are static; numerical magnitudes may be live.

## CoordinatePoint

Quantity-mode integrands receive:

```python
CoordinatePoint(
    values=values,
    units=units,
)
```

`values.shape == (..., dimension)`. `units` is a static tuple.

The supported interface is:

```python
x.shape
x.dimension
x.values
x.units
x.axis(i)
x.as_quantity(unit)
```

`x.axis(i)` requires static $i$ and returns a scalar-axis `Quantity`.
`as_quantity(unit)` is available only when every axis is compatible with the
requested unit.

Raw mode continues to receive an ordinary coordinate-last array.

## Result dimensions

For axis units $U_i$, integrand unit $U_f$, and density unit $U_{\rho}$:

```{math}
U_I
=
U_f
\left(\prod_{i=1}^{d}U_i\right)
U_{\rho}.
```

`epsabs` must have units compatible with $U_I$; `epsrel` is dimensionless. The
initial Phase B quantity contract requires one common output dimension for every
array-valued result component.

## Measures

Phase B supports:

- `LebesgueMeasure`;
- `ProductMeasure` over independent per-axis measures; and
- multidimensional `WeightedMeasure` with an explicit density unit.

`LebesgueMeasure` evaluates a dimensionless unit density in physical coordinate
space. A `ProductMeasure` contains exactly one finite-domain
`LebesgueMeasure` or one-dimensional `WeightedMeasure` per axis. Each weighted
component receives the physical-axis `Quantity`; its component density units
multiply to form $U_{\rho}$. Infinite-domain classical measures are rejected
because Phase B domains are finite hyperrectangles.

A multidimensional `WeightedMeasure` receives the physical
`CoordinatePoint`, explicit `args`, and returns one density value per input
point with units compatible with its declared `density_unit`.

`normalized=True` is static metadata supplied by the measure owner. Phase B
does not infer, test, or renormalize a user density from that flag, and no
result status claims normalization. Built-in normalized measures must carry an
analytic normalization identity tested independently; user-weighted measures
remain declarations.

Quantity support remains alpha and opt-in. No sibling defaults change.

# Shared JAX architecture

## Data flow

```text
public integrate facade
        |
        v
eager structural validation
        |
        v
raw or quantity normalization
        |
        v
reference-space point/controller owner
        |
        v
shared coordinate-last evaluator
        |
        v
family accumulation and evidence
        |
        v
QuadResult
        |
        v
optional quantity restoration
```

## Module boundaries

The intended public and private owners are:

```text
domains.py             Hyperrectangle and Axis declarations
coordinates.py         CoordinatePoint and quantity normalization
tensor.py              tensor methods and public family evaluator
_tensor.py             tensor construction and nested reuse
cubature.py            adaptive cubature declarations and evaluator
_cubature.py           Genz-Malik rule and region controller
sparse.py              Smolyak declarations and evaluator
_sparse.py             index sets, admissibility, coalescing, controller
qmc.py                 QMC declarations and evaluator
_sobol.py              direction numbers and deterministic Sobol points
_scramble.py           LMS, Owen, and digital-shift owners
_qmc_interval.py       fixed-look and bounded sequential intervals
_multidim_replay.py    family replay dispatch
result.py              shared result, error, work, and status records
```

The exact private split may be refined to keep files focused. Algorithms do not
move between mathematical family owners.

## Static capacity rules

Adaptive methods use fixed-capacity arrays and masks. Python-side validation
computes structural bounds before tracing. Runtime loops use fixed-length scans
or statically bounded loops. Phase B does not introduce differentiated
`lax.while_loop` owners.

Inactive padded storage never contributes to logical work.

# Shared error handling

Static structural errors raise eagerly:

- incompatible shapes;
- dynamic or zero dimension;
- host-known nonfinite domain bounds;
- unsupported method-domain pairing;
- invalid rule order or level;
- impossible capacity declaration;
- incompatible units;
- invalid bit depth;
- non-power-of-two QMC request;
- insufficient randomized replicates;
- missing QMC key;
- invalid confidence level; and
- missing bounds for bounded sequential QMC.

Dynamic numerical outcomes return `QuadStatus`:

- `CONVERGED`;
- `MAX_EVALUATIONS`;
- `MAX_REGIONS`;
- `MAX_INDICES`;
- `NONFINITE_INTEGRAND`;
- `ROUNDOFF_LIMITED`;
- `DIVERGENCE_SUSPECTED`;
- `INVALID_INPUT` for traced numerical inputs that cannot be rejected eagerly;
  and
- `ERROR_ESTIMATE_UNAVAILABLE`.

Each family fixes status precedence in its implementation plan and tests it
with mutation-resistant cases.

# B4.3: validation, comparison, and documentation

## Mathematical truth

The deterministic suite includes:

- tensor polynomial moments;
- Gaussian and beta-product moments;
- separable exponential integrals;
- rotated smooth functions;
- the complete Genz oscillatory, product-peak, corner-peak, Gaussian, continuous,
  and discontinuous families;
- localized peaks;
- boundary layers; and
- independently certified high-accuracy references where no analytic result is
  available.

## Astrophysical applications

Public validation includes domain-neutral versions of:

- a finite-aperture Plummer-like volume integral;
- multivariate Gaussian velocity normalization and covariance;
- a mass-metallicity-age-distance population expectation; and
- a bounded survey-selection efficiency.

These demonstrate applicability without transferring physical model ownership
into Jaxstro.

## JAX matrix

Every claimed family covers:

```text
eager
jit
vmap
jvp
vjp
grad
jit(grad)
vmap(grad)
jit(vmap(grad))
```

For deterministic families, the matrix includes moving bounds, heterogeneous
accepted states, scalar and array payloads, real and documented complex
payloads, and float32/float64. Randomized integration covers real scalar
payloads only in Phase B so that its confidence statement remains
well-defined. Coincident bounds are tested separately as a zero-primal,
unsupported-replay boundary.

## Randomized calibration

RQMC campaigns predeclare:

- seed count;
- nominal confidence;
- binomial empirical-coverage acceptance bands;
- matched evaluation budgets;
- RMSE metric;
- interval-width metric;
- smoothness and effective-dimension families; and
- failure criteria.

The alpha-spending implementation must fail mutation tests that reuse the full
alpha at every level.

## External comparisons

Development-only comparison lanes are:

- SciPy `integrate.cubature` for family-matched Genz-Malik evidence;
- SciPy `stats.qmc.Sobol` for exact deterministic and LMS-plus-shift point
  construction where configurations match;
- ORNL Tasmanian for sparse-grid node, moment, and convergence evidence; and
- Torchquad for selected tensor, accelerator, and differentiable capability
  comparisons.

Every record is labeled `exact`, `strong-match`, `node-matched`,
`family-matched`, or `capability`. A label is attached to one record, not one
library globally.

Comparators never become runtime dependencies.

## Performance

Benchmark dimensions begin at 2, 4, 8, and 16, with larger QMC point-generation
cases where feasible.

Modes include compile, single solve, VMAP-16, VMAP-128 where feasible, JVP,
gradient, repeated same-domain calls, and repeated changing-parameter calls.

Metrics include truth error, estimator identity, logical evaluations, unique
nodes, regions or indices, replicates, Sobol level, compile time, warm runtime,
dispersion, compiler-cost proxy, memory proxy, gradient error, and empirical
coverage.

Optimization follows the Phase A protocol:

1. freeze controls;
2. preserve a reviewed baseline;
3. evaluate predeclared triggers;
4. profile the exact owner;
5. write an optimization addendum;
6. make the smallest owner-local change;
7. preserve the immutable baseline; and
8. require two independent optimized suites.

# Documentation architecture

The MyST table of contents adds:

```text
Methods
  Multidimensional integration
    Hyperrectangles and coordinate maps
    Tensor-product quadrature
    Adaptive cubature
    Sparse-grid integration
    Randomized quasi-Monte Carlo
    Differentiating multidimensional integrals
    Choosing a multidimensional method
```

API pages remain grouped by family even though `quad.integrate` is the primary
workflow.

Every method page includes:

- the scientific question;
- geometric interpretation;
- derivation;
- cost growth;
- estimator or uncertainty meaning;
- JAX behavior;
- quantity behavior;
- a worked astrophysical example;
- failure modes;
- an audit recipe; and
- a warranted claim boundary.

Admonitions distinguish mathematical assumptions, JAX constraints, statistical
warnings, capacity limits, and method-choice guidance.

# Delivery gates

## B0

- Hyperrectangle and coordinate-last evaluator contracts.
- Unified facade dispatch.
- Result/status extension.
- Raw JAX transform envelope.
- No numerical Phase B family yet.

## B1

- Fixed heterogeneous tensor rules.
- p-adaptive tensor Clenshaw-Curtis.
- h-adaptive Genz-Malik.
- Dimensional envelope 2 through 8.
- Deterministic analytic and comparison evidence.

## B2

- Isotropic Smolyak.
- Static anisotropic Smolyak.
- Dimension-adaptive admissible refinement.
- Exact dyadic node coalescing.
- Sparse surplus and index-capacity evidence.

## B3

- Vendored Joe-Kuo provenance.
- Deterministic Sobol.
- LMS plus shift.
- Nested Owen scrambling.
- Fixed-look Student-t randomized integration.
- Bounded sequential empirical-Bernstein integration.
- Coverage campaigns.

## B4

- First-order replay differentiation for all families.
- Heterogeneous `CoordinatePoint` quantity mode.
- External comparison artifacts.
- Performance and compiler-cost evidence.
- Researcher-facing derivations, API guides, and validation pages.
- Complete release gate and independent review.

# Phase B completion

Phase B is complete only when:

- B0 through B4 public contracts are implemented;
- every approved family is importable and documented;
- replay and stop modes pass their claimed envelopes;
- heterogeneous quantity coordinates pass representation invariance;
- randomized confidence claims pass predeclared coverage campaigns;
- generated contracts and evidence are fresh;
- external comparisons use calibrated labels;
- the complete local release gate passes;
- independent numerical, JAX, statistical, quantity, API, and documentation
  reviews have no remaining Critical or Important findings; and
- no sibling migration, publication, push, deployment, or Phase C
  implementation is included.

# References

- S. A. Smolyak, [Interpolation and quadrature formulas for the classes
  $W_s^\alpha$ and $E_s^\alpha$](https://www.mathnet.ru/eng/dan39981), 1960.
- A. C. Genz and A. A. Malik, [An adaptive algorithm for numerical integration
  over an N-dimensional rectangular region](https://doi.org/10.1016/0771-050X(80)90039-X),
  1980.
- T. Gerstner and M. Griebel, [Dimension-adaptive tensor-product
  quadrature](https://doi.org/10.1007/s00607-003-0015-5), 2003.
- S. Joe and F. Y. Kuo, [Constructing Sobol sequences with better
  two-dimensional projections](https://doi.org/10.1137/070709359), 2008.
- A. B. Owen, [Monte Carlo variance of scrambled net
  quadrature](https://doi.org/10.1137/S0036142994277468), 1997.
- A. B. Owen, [Scrambling Sobol and Niederreiter-Xing
  points](https://doi.org/10.1006/jcom.1998.0487), 1998.
- P. L'Ecuyer, M. K. Nakayama, A. B. Owen, and B. Tuffin, [Confidence intervals
  for randomized quasi-Monte Carlo
  estimators](https://doi.org/10.1109/WSC60868.2023.10408613), 2023.
- A. Jain, F. J. Hickernell, A. B. Owen, and A. Sorokin, [Empirical Bernstein
  and betting confidence intervals for randomized quasi-Monte
  Carlo](https://doi.org/10.1093/imaiai/iaag003), 2026.
