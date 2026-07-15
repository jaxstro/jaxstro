---
title: Jaxstro quad capability program design
date: 2026-07-15
status: approved in dialogue; written-spec review pending
---

# Jaxstro quad capability program design

## Purpose

Jaxstro will own a standalone, JAX-native integration library under
`jaxstro.quad`. The package will begin with complete one-dimensional fixed and
adaptive quadrature, then add deterministic multidimensional cubature, sparse
grids, randomized quasi-Monte Carlo, and scientific integration methods. It will
not depend on Quadax at runtime.

The package is intended for researchers who may be new to numerical integration,
JAX transformations, automatic differentiation, or dimensional quantities. Its
documentation therefore derives the methods, explains their failure modes, and
distinguishes a computed value from evidence that the value is accurate.

The program has three capability phases:

1. Phase A: complete one-dimensional integration.
2. Phase B: general multidimensional integration on hyperrectangles.
3. Phase C: native scientific geometries and specialized integral families.

This is a program-level architecture specification. Each phase is divided into
bounded implementation slices, and each phase after Phase A requires its own
detailed design and implementation plan. No empty Phase B or Phase C runtime
packages are created during Phase A.

## Decisions and ownership changes

This design changes the ownership recorded by the current roadmap, SOTA
assessment, adaptive-quadrature guide, and their contract tests. Those surfaces
currently delegate adaptive quadrature to Quadax and place quasi-Monte Carlo in
`jaxstro.numerics.qmc`. After the first supported `jaxstro.quad` slice exists,
they must instead record:

- `jaxstro.quad` owns Jaxstro's integration rules, controllers, domains,
  measures, results, derivative contracts, and numerical evidence.
- Quadax remains an independent comparison implementation and optional
  benchmark reference, not a runtime dependency or delegated owner.
- `jaxstro.quad` owns future quasi-Monte Carlo integration. It does not own
  generic Monte Carlo inference, posterior computation, or experimental design.
- Existing `jaxstro.numerics.integration` and
  `jaxstro.numerics.quadrature` paths remain temporary compatibility surfaces
  backed by the canonical implementation.
- Startrax, Gravax, Progenax, and other sibling packages are not migrated by
  this program until their separately approved consumer migrations begin.

The package name `quad` is canonical. Documentation teaches only
`jaxstro.quad`; compatibility paths appear only in migration notes.

## Design principles

### Mathematical honesty

An integral is represented as

```{math}
:label: eq-quad-measure-integral

I[f] = \int_{\Omega} f(x)\,\mathrm{d}\mu(x),
```

where the domain $\Omega$, measure $\mu$, integrand $f$, numerical rule, error
evidence, and stopping policy are distinct objects. A paired-rule difference,
a sparse-grid surplus, and a randomized-QMC standard error do not have the same
meaning and must never share an unlabeled error field.

### JAX-native execution

Hot numerical kernels use JAX arrays and JAX control flow. Python may construct
static rules, recurrence data, and immutable configuration outside traced hot
paths. Increasing a numerical denominator changes an array shape or a JAX loop
bound; it does not increase Python call count.

### Differentiability with an explicit target

The initial public derivative modes are `replay` and `stop`:

- `replay` differentiates the accepted fixed quadrature formula while treating
  refinement choices, region ordering, stopping decisions, and randomization
  metadata as nondifferentiable.
- `stop` explicitly blocks derivatives through the integral result.

The initial public API does not expose `through`. Differentiating executed
adaptive refinement is reserved for a later experimental fixed-length
`jax.lax.scan` design with separate derivative meaning and evidence.

### Dimensional safety without a forced quantity cutover

Raw arrays remain the primary hot-kernel representation. An opt-in quantity
boundary validates dimensions, unwraps quantity values before the numerical
engine, and restores the integral unit afterward. This does not change the
current alpha adoption status of `jaxstro.quantity` or authorize downstream
quantity migrations.

### Evidence before SOTA claims

Feature breadth alone does not establish superiority. Jaxstro may claim a
capability after its contract and validation gates pass. It may claim an
accuracy, calibration, differentiation, memory, or performance advantage only
after a matched comparison supports that specific statement.

## Layered architecture

```text
integrand + explicit parameters
             |
domain + measure + transform
             |
shared vectorized evaluator
             |
   +---------+-----------+----------------+
   |                     |                |
fixed rule        adaptive regions   point-set methods
   |                     |                |
weighted sum      embedded evidence  sparse grid or QMC
   +---------+-----------+----------------+
             |
typed error evidence + work account + status
             |
          QuadResult
```

The shared layers own domain transformations, measure evaluation, vectorized
integrand evaluation, accumulation, tolerances, budgets, statuses, and result
types. Method families own their numerical controllers:

- fixed rules own nodes, weights, exactness metadata, and evaluation;
- one-dimensional adaptive methods own interval refinement;
- multidimensional adaptive cubature owns region subdivision;
- sparse grids own admissible multi-index refinement and node coalescing;
- randomized QMC owns point construction, scrambling, replication, and sample
  growth.

These controllers share infrastructure but not an artificial universal
refinement algorithm.

## Public package structure

Phase A creates only the files needed by the one-dimensional implementation:

```text
jaxstro/quad/
  __init__.py
  result.py
  tolerance.py
  domains.py
  measures.py
  transforms.py
  sampled.py
  fixed.py
  adaptive.py
  rules.py
```

Private helpers may be split when a file acquires more than one responsibility,
but the public module map remains organized by mathematical role. Phase B adds
bounded modules such as `cubature.py`, `sparse.py`, and `qmc.py` only when their
implementation slices begin.

### Public entry points

The public API uses explicit functions for numerically different workflows:

```python
quad.trapezoid(y, x=None, *, dx=1.0, axis=-1)
quad.cumulative_trapezoid(y, x=None, *, dx=1.0, axis=-1)
quad.simpson(y, x=None, *, dx=1.0, axis=-1)
quad.cumulative_simpson(y, x=None, *, dx=1.0, axis=-1)

quad.fixed(fun, domain, *, args=(), rule, measure=None)
quad.integrate(
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
    error_norm,
    gradient="replay",
)
```

Phase B later adds:

```python
quad.cubature(...)
quad.sparse_grid(...)
quad.qmc(..., key=...)
```

The same `fixed` and `integrate` entry points accept either raw-array inputs or
consistent `Quantity` inputs. A thin eager normalizer selects the raw numerical
kernel from the static unit metadata; there is no parallel quantity algorithm
and no second quantity-only method namespace.

There is no `method="auto"` policy. Methods and their immutable configuration
objects are explicit and static under JIT. Examples include fixed Gaussian
rules, adaptive Gauss-Kronrod pairs, adaptive Clenshaw-Curtis, adaptive
tanh-sinh, Smolyak sparse grids, and scrambled Sobol integration. The exact
Phase B configuration types are fixed in the Phase B design rather than
pre-scaffolded in Phase A.

Differentiable parameters belong in the explicit `args` PyTree. A closure may
be convenient for fixed rules, but the adaptive derivative contract applies
only to parameters passed through `args`, domain endpoints, supported measure
parameters, and supported transform parameters.

### Phase A configuration types

Phase A freezes these public immutable configuration families:

```python
Interval(lower, upper, *, breakpoints=())
RightInfinite(lower)
LeftInfinite(upper)
Infinite()

LebesgueMeasure()
WeightedMeasure(density, *, density_unit, normalized=False)
JacobiMeasure(alpha, beta, *, normalized=False)
LaguerreMeasure(alpha=0.0, *, normalized=False)
PhysicistsHermiteMeasure(*, normalized=False)
StandardNormalMeasure()

GaussianRule(order)
ClenshawCurtisRule(order)
FejerIRule(order)
FejerIIRule(order)
TanhSinhRule(level)

MaxNorm()
L1Norm()
L2Norm()

GaussKronrod(pair=21)
AdaptiveClenshawCurtis(initial_order=17)
AdaptiveTanhSinh(initial_level=3)
Romberg(initial_level=1)
RombergTanhSinh(initial_level=1)
```

Domains are registered PyTrees whose numerical endpoints and breakpoint values
are dynamic children. The domain type and breakpoint array length are static.
Every supplied breakpoint is active; changing the number of breakpoints changes
the static shape and may recompile. Breakpoints are sorted into domain order by
the primal engine, validated as interior points, and stopped in the derivative.

Rules and methods are frozen configuration objects whose order, pair, and level
fields are static auxiliary data. Classical measure parameters used to generate
Gaussian nodes are also static in Phase A. A general `WeightedMeasure` holds a
static density callable with signature `density(x, args)`, a density unit, and a
normalization declaration; any differentiable density parameters are passed
through the same explicit `args` PyTree as the integrand. `normalized=True`
declares that the supplied density is already normalized. It never causes a
hidden numerical normalization step.

The fixed Gaussian rule selects its recurrence from the measure:

- `LebesgueMeasure` on a finite interval selects Legendre;
- `JacobiMeasure` on a finite interval selects Jacobi;
- `LaguerreMeasure` on a right-infinite interval selects generalized Laguerre;
- `PhysicistsHermiteMeasure` on the full line selects physicists' Hermite; and
- `StandardNormalMeasure` on the full line selects normalized probabilists'
  Hermite.

Clenshaw-Curtis and Fejer rules accept finite intervals and either Lebesgue or a
general weighted measure. Fixed tanh-sinh accepts any Phase A domain through its
documented transform. Gauss-Kronrod and adaptive Clenshaw-Curtis accept finite
intervals, including breakpoints. Adaptive tanh-sinh accepts every Phase A
domain. Romberg accepts finite intervals; `RombergTanhSinh` accepts the tanh-sinh
domain set. Unsupported pairings raise eagerly when structural and return
`INVALID_INPUT` when their invalidity depends on traced values.

Representative calls are:

```python
value = quad.fixed(
    fun,
    quad.Interval(-1.0, 1.0),
    rule=quad.GaussianRule(32),
)

result = quad.integrate(
    fun,
    quad.Interval(0.0, 1.0, breakpoints=(0.25, 0.75)),
    args=params,
    method=quad.GaussKronrod(pair=21),
    epsabs=1e-10,
    epsrel=1e-8,
    max_evaluations=4096,
    max_regions=256,
    error_norm=quad.MaxNorm(),
)

tail = quad.integrate(
    fun,
    quad.RightInfinite(0.0),
    method=quad.AdaptiveTanhSinh(),
    epsabs=1e-10,
    epsrel=1e-8,
    max_evaluations=4096,
    max_regions=256,
    error_norm=quad.MaxNorm(),
)

expectation = quad.fixed(
    fun,
    quad.Infinite(),
    rule=quad.GaussianRule(32),
    measure=quad.StandardNormalMeasure(),
)
```

### Integrand contract

The engine presents a node axis to `fun`. For a one-dimensional rule with $n$
nodes, the coordinate input has shape `(n,)`; for a $d$-dimensional point set it
has shape `(n, d)`. The integrand returns either `(n,)` or `(n, ...)`, where the
leading axis corresponds to nodes and the trailing shape is a fixed scalar,
vector, complex, or array payload.

The output payload shape, dimension, method, rule order, domain rank, capacities,
error norm, and gradient mode are static. Bounds, explicit parameters,
tolerances, and fixed-capacity breakpoint values may be dynamic arrays.

## Domains, measures, and transformations

### Domains

Phase A owns:

- finite intervals;
- semi-infinite intervals;
- the full real line;
- oriented bounds;
- explicit interior breakpoints with fixed capacity; and
- explicit endpoint behavior declarations used only by methods that document
  support for them.

`Interval(lower, upper)` preserves orientation. The numerical map factors the
integral into a positive measure Jacobian and a separate orientation
$\sigma=\operatorname{sign}(\mathrm{upper}-\mathrm{lower})$. In Phase B the
hyperrectangle orientation is the product of the per-axis signs. Zero-width
axes have zero orientation and return an exact zero integral without evaluating
the integrand numerically after JAX has inferred the static output shape.

Phase B begins with static-dimensional hyperrectangles. Native simplexes and
spheres are committed Phase C capabilities. Before those native geometries
exist, a user may apply a documented coordinate transformation, but Jaxstro
does not describe that as native geometry support.

### Measures

The default is Lebesgue measure. Weighted integration separates three cases:

1. An unnormalized weighted measure, whose density multiplies the integrand and
   whose normalization is not silently changed.
2. A normalized probability measure, whose normalization is part of its public
   contract.
3. A classical orthogonality measure with a matched Gaussian rule and explicit
   support, normalization, recurrence coefficients, and exactness class.

Phase A includes Legendre, Jacobi, Laguerre, generalized Laguerre, physicists'
Hermite, and probabilists' normalized-Hermite conventions through one shared
recurrence and Jacobi-matrix construction. Existing probabilists' Hermite
behavior remains byte-compatible where the old public contract requires it. To
meet that exact legacy contract, `gauss_hermite_nodes` retains its current
NumPy `hermgauss` rescaling as one narrow compatibility backend. New
`GaussianRule` construction uses the shared recurrence engine. The compatibility
backend is removed only with the legacy helper in a breaking release; its output
is never silently replaced by a merely close recurrence result.

The API never applies a declared weight twice. A matched weighted rule consumes
the measure as part of the rule. A general adaptive rule evaluates the declared
density exactly once inside the transformed integrand.

### Transformations

Domain maps own the reference coordinate, physical coordinate, Jacobian, and
boundary convention. They do not own error estimation or stopping. A transformed
integral is evaluated as

```{math}
:label: eq-quad-domain-map

\int_{\Omega} f(x)\,\mathrm{d}\mu(x)
= \sigma \int_{\widehat{\Omega}}
f\!\left(T(t)\right)
\rho\!\left(T(t)\right)
\left|J_T(t)\right|\,\mathrm{d}t,
```

where $T$ maps into the positively ordered physical domain, $\sigma$ preserves
the requested orientation, and $\rho=1$ for Lebesgue measure. Infinite-domain
transforms and endpoint maps expose their assumptions. No transform claims to
detect a divergent tail or unresolved singularity automatically.

## One-dimensional methods

### Sampled-data rules

The existing trapezoid, cumulative trapezoid, Simpson, and cumulative Simpson
contracts move to `jaxstro.quad.sampled`. Their numerical ordering and current
edge behavior remain unchanged during canonicalization. These functions return
array values rather than `QuadResult` because they neither adapt nor estimate
the error of the underlying continuous integral.

### Fixed rules

Phase A includes:

- Gauss-Legendre;
- Gauss-Jacobi;
- Gauss-Laguerre and generalized Gauss-Laguerre;
- physicists' and probabilists' Gauss-Hermite;
- Clenshaw-Curtis;
- Fejer type I and type II; and
- fixed tanh-sinh.

Gaussian families share one recurrence/Jacobi-matrix engine. Clenshaw-Curtis and
Fejer rules share one Chebyshev construction substrate. `quad.fixed` returns the
integral value. Exactness and rule metadata are properties of the rule, not a
runtime error estimate.

### Adaptive rules

Phase A includes:

- Gauss-Kronrod pairs 15, 21, 31, 41, 51, and 61;
- nested adaptive Clenshaw-Curtis;
- adaptive tanh-sinh;
- Romberg integration; and
- Romberg integration with tanh-sinh refinement.

Every adaptive method returns `QuadResult`. The controller uses fixed-capacity
JAX arrays and a primal `jax.lax.while_loop`. The loop is never differentiated:
`replay` differentiates a fixed formula reconstructed from accepted regions,
and `stop` blocks derivatives.

Refinement priorities, tolerance checks, capacity exhaustion, nonfinite values,
and status precedence are deterministic. No Python loop advances intervals,
levels, node counts, or evaluation batches.

## Result, error, work, and status contracts

### Result structure

Adaptive one-dimensional integration and all Phase B methods return one
JAX-compatible result family:

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

All leaves are fixed-shape JAX arrays or integer codes. User-facing text for
status and error meanings is supplied by eager formatting helpers rather than
stored inside traced results.

For scalar output, `error.estimate` is a nonnegative scalar. For real vector or
array output, it is a nonnegative array with the same payload shape as `value`.
For complex output, it is a real nonnegative magnitude array with the same
payload shape. `error.norm` is the scalar reduction actually compared with the
scalar `tolerance`. The selected norm configuration is static.

Under `gradient="replay"`, only `QuadResult.value` participates in automatic
differentiation. Under `gradient="stop"`, the value is stopped as well. Error,
tolerance, status, and work leaves are always explicitly stopped. Their values
describe the executed primal solve and are not derivative observables.

### Error semantics

`QuadError.kind` distinguishes at least:

- `EMBEDDED_RULE`;
- `REFINEMENT_DIFFERENCE`;
- `SPARSE_GRID_SURPLUS`;
- `REPLICATE_STANDARD_ERROR`;
- `CONFIDENCE_INTERVAL_HALF_WIDTH`; and
- `UNAVAILABLE`.

Polynomial exactness is rule metadata, not an error kind. Deterministic QMC
without independent randomizations reports `UNAVAILABLE`; it does not invent a
statistical uncertainty. `confidence_level` is meaningful only when a method
constructs a documented randomization-based interval and is otherwise a
`NaN` sentinel. `REPLICATE_STANDARD_ERROR` contains a standard error and has no
confidence level. `CONFIDENCE_INTERVAL_HALF_WIDTH` contains the half-width of
the interval at the recorded confidence level. The first supported randomized
QMC stopping policy uses the latter. Work counters that do not apply to a method
are zero. This keeps the result shape uniform without pretending that, for
example, QMC has active subdivision regions.

For a deterministic method, the stopping comparison is

```{math}
:label: eq-quad-tolerance

E \leq \max\!\left(\epsilon_{\mathrm{abs}},
\epsilon_{\mathrm{rel}}\lVert \widehat{I}\rVert\right),
```

where the selected norm is static. Passing this comparison means that the
method's estimator met its criterion; it is not a universal proof of the true
error. Randomized QMC uses its separately documented statistical criterion.

### Status semantics

The initial status codes are:

- `CONVERGED`;
- `MAX_EVALUATIONS`;
- `MAX_REGIONS`;
- `NONFINITE_INTEGRAND`;
- `ROUNDOFF_LIMITED`;
- `DIVERGENCE_SUSPECTED`;
- `INVALID_INPUT`; and
- `ERROR_ESTIMATE_UNAVAILABLE`.

Static structural errors such as an invalid rule order, incompatible static
shapes, unsupported method-domain pairing, or dimensionally invalid absolute
tolerance raise eagerly. Dynamic numerical outcomes return a status. Status
precedence is fixed by each method contract so the same executed state cannot
produce different labels.

`CONVERGED` always means "converged according to the named estimator and
tolerance policy." The documentation must not shorten it to "the integral is
correct."

## Automatic differentiation

### Replay derivative

After the primal adaptive solve, the custom JVP/VJP freezes the accepted
partition in normalized reference coordinates with `jax.lax.stop_gradient`.
Replay reconstructs physical nodes and weights from that fixed reference
partition and the live domain bounds, supported transform parameters, and
explicit `args`, including supported weighted-density parameters. Classical
Gaussian measure parameters remain static in Phase A. Replay then differentiates
the reconstructed fixed formula. It
does not differentiate sorting, interval selection, region subdivision,
stopping, capacity logic, breakpoint motion, or status codes. Accepted physical
endpoints may be retained for diagnostics, but replay never uses stopped
physical endpoints as the source of quadrature nodes.

For smooth finite limits, replaying the transformed fixed formula includes the
dependence of the map and Jacobian on the bounds. In the converged limit this
corresponds to the Leibniz identity

```{math}
:label: eq-quad-leibniz

\frac{\mathrm{d}}{\mathrm{d}\theta}
\int_{a(\theta)}^{b(\theta)} f(x,\theta)\,\mathrm{d}x
=
\int_{a(\theta)}^{b(\theta)}
\frac{\partial f}{\partial\theta}(x,\theta)\,\mathrm{d}x
+ f\!\left(b(\theta),\theta\right)b'(\theta)
- f\!\left(a(\theta),\theta\right)a'(\theta).
```

The derivative contract applies only where the integrand, supported measure,
and transformation are smooth and differentiation may be interchanged with
integration. Breakpoint motion, branch changes, singularity declarations,
capacity changes, and method changes are explicit nonsmooth boundaries.

The primal error estimate does not certify derivative error. Validation reports
primal error and derivative error as separate metrics; the runtime
`QuadResult.error` describes only the primal integral.

### QMC derivative

Randomized QMC replay differentiates the integrand values and supported domain
maps at the realized point set. Keys, direction numbers, digital scrambling,
replicate count, and sample count are nondifferentiable. Reproducibility requires
an explicit JAX key and a frozen point-construction contract.

## Quantity boundary

If $f$ has unit $U_f$ and the coordinate has unit $U_x$, an unweighted
one-dimensional result has unit

```{math}
:label: eq-quad-result-unit

U_I = U_f U_x.
```

For multidimensional or weighted integrals, the measure and transformation
determine the complete result dimension. The quantity wrapper:

1. validates domain, measure, integrand, and tolerance dimensions eagerly;
2. converts values to a selected unit representation;
3. passes raw JAX arrays to the numerical kernel;
4. restores the result quantity and the matching error-estimate unit; and
5. records no hidden global unit context.

`epsrel` is dimensionless. `epsabs` must match the integral dimension. A
dimensionless probability measure does not contribute a coordinate unit; an
unnormalized density contributes exactly the dimension declared by that
measure.

## Phase B: general multidimensional integration

Phase B begins only after Phase A's public contracts and validation envelope are
stable. It adds four distinct method families on static-dimensional
hyperrectangles.

### Tensor-product rules

Tensor products reuse one-dimensional rules for low-dimensional smooth
problems. The API exposes the exponential node-growth cost and does not present
tensor products as a general high-dimensional solution.

### Adaptive cubature

Adaptive cubature subdivides hyperrectangles and uses a documented embedded
symmetric rule to estimate local error. The exact embedded rule, split-axis
policy, and estimator rescaling are selected and approved in the Phase B design.
That design must preserve the result, status, budget, transform, and replay-AD
contracts defined here.

### Sparse grids

Sparse integration uses Smolyak constructions built first from nested
Clenshaw-Curtis rules. It supports isotropic and explicitly anisotropic level
sets, admissible dimension-adaptive refinement, and duplicate-node coalescing.
Sparse-grid surpluses remain labeled as surplus evidence rather than absolute
error bounds.

### Randomized quasi-Monte Carlo

The first QMC implementation uses Sobol point sets with a documented Owen-style
scrambling contract, explicit JAX keys, independent randomized replicates, and
power-of-two sample growth. Before implementation, the Phase B design freezes:

- direction-number provenance and license;
- supported dimension and bit limits;
- prefix, skip, and nesting behavior;
- scrambling construction;
- key splitting and replicate ownership;
- dtype and integer-width behavior; and
- the statistical interval and coverage contract.

Later QMC methods may include randomized rank-one lattice rules and Halton
sequences. They are additions to the same point-set layer, not requirements for
the first supported QMC release.

## Phase C: scientific integration

Phase C adds methods that benefit recurring scientific calculations while
remaining domain-agnostic numerical machinery:

- native simplex and spherical domains;
- radial and angular measures;
- logarithmic and scale-aware transformations;
- oscillatory Fourier-type integration;
- Bessel-weighted and Hankel-type integrals;
- line-of-sight and projected-profile integration; and
- endpoint-asymptotic declarations and specialized rules.

Physical models, observational policy, astrophysical parameter conventions, and
scientific acceptance thresholds remain in downstream packages. `jaxstro.quad`
owns the numerical integral, not the interpretation of a particular observable.

## Compatibility and migration

Canonical implementations move to `jaxstro.quad`; legacy modules never contain
forked algorithms.

The Phase A compatibility manifest is exact:

| Existing public surface | Canonical owner |
| --- | --- |
| `jaxstro.numerics.integration.trapz` | `jaxstro.quad.trapezoid` |
| `jaxstro.numerics.integration.cumulative_trapz` | `jaxstro.quad.cumulative_trapezoid` |
| `jaxstro.numerics.integration.simpson` | `jaxstro.quad.simpson` |
| `jaxstro.numerics.integration.cumulative_simpson` | `jaxstro.quad.cumulative_simpson` |
| `jaxstro.numerics.quadrature.gauss_legendre_nodes` | `jaxstro.quad.gauss_legendre_nodes` |
| `jaxstro.numerics.quadrature.gauss_laguerre_nodes` | `jaxstro.quad.gauss_laguerre_nodes` |
| `jaxstro.numerics.quadrature.gauss_hermite_nodes` | `jaxstro.quad.gauss_hermite_nodes` |
| `jaxstro.numerics.quadrature.clenshaw_curtis_nodes` | `jaxstro.quad.clenshaw_curtis_nodes` |
| `jaxstro.numerics.quadrature.hermite_e_basis` | `jaxstro.quad.hermite_e_basis` |
| `jaxstro.numerics.quadrature.hermite_coefficients` | `jaxstro.quad.hermite_coefficients` |

The six quadrature helpers currently re-exported directly from
`jaxstro.numerics` continue to resolve there during compatibility and point to
the same canonical `jaxstro.quad` callables. The sampled integration functions
are not currently top-level `jaxstro.numerics` exports, so Phase A does not add
new legacy top-level aliases for them.

The migration stages are:

1. Add `jaxstro.quad` and make existing sampled and fixed behavior canonical
   there.
2. Re-export the existing names from `jaxstro.numerics.integration` and
   `jaxstro.numerics.quadrature` without changing behavior. In particular, the
   old `trapz` spelling remains a compatibility alias for canonical
   `quad.trapezoid`.
3. Teach only `jaxstro.quad` in current documentation and record old paths in a
   migration page.
4. Audit one sibling repository at a time after its active work permits.
5. Migrate and verify each sibling under a separately approved plan.
6. Add deprecation warnings only when warnings will not disrupt traced code or
   downstream test contracts.
7. Remove legacy paths in a declared breaking release after repository-wide
   consumer searches and pinned downstream tests are clean.

Backward compatibility is temporary migration infrastructure, not permission to
maintain duplicate implementations indefinitely.

## Validation and evidence

The program uses Jaxstro's existing three-tier test architecture.

### Unit evidence

- node, weight, symmetry, positivity, and normalization invariants;
- analytic polynomial and classical weighted moments;
- exactness through each documented rule degree;
- transform and Jacobian identities;
- measure normalization and no-double-weight tests;
- status precedence and exact work accounting;
- quantity conversion invariance and result dimensions;
- static-shape and capacity boundaries; and
- failures for invalid configurations and nonfinite integrands.

### JAX integration evidence

- `jax.jit` for each supported public workflow;
- `jax.vmap` over explicit parameter batches;
- scalar, vector, array, real, and complex outputs where claimed;
- custom JVP/VJP agreement for explicit parameters, moving finite bounds,
  transforms, and supported measure parameters;
- deterministic key reproduction and independent QMC scramble streams; and
- proof that no Python loop advances regions, levels, dimensions, or samples.

### Numerical validation

- analytic finite, weighted, improper, singular, peaked, discontinuous, and
  oscillatory benchmark families within each method's declared envelope;
- independently generated high-precision fixtures with recorded provenance;
- tolerance sweeps comparing reported estimates with observed primal error;
- central finite-difference and analytic derivative comparisons on smooth
  domains;
- separate primal and derivative error reports;
- hyperrectangle volumes and mixed moments;
- rotated and localized multidimensional functions;
- sparse-grid admissibility, nesting, coalescing, and convergence;
- authoritative Sobol prefixes, nesting, skip behavior, and scramble fixtures;
- randomized-QMC matched-budget RMSE and predeclared empirical coverage tests;
  and
- capacity, roundoff, divergence, and unresolved-feature failure cases.

Every method implementation plan must predeclare its numeric acceptance
thresholds, reference source, dtype, precision mode, and benchmark envelope
before implementation. A passing self-consistency test is not an independent
validation result.

### Comparison and performance evidence

SciPy, Quadax, and high-precision tools may appear in pinned validation or
benchmark environments; none becomes a Jaxstro runtime dependency. Comparisons
match:

- mathematical problem and domain transformation;
- dtype and precision configuration;
- requested and achieved accuracy;
- function-evaluation budget;
- cold compilation, warm execution, and synchronization policy;
- scalar, batched, and vector-valued workload;
- hardware and software versions; and
- memory accounting where claimed.

Jaxstro may state that it supports a method after its own gates pass. It may
state that it is better than another package only when confidence intervals or
deterministic matched evidence support the named metric on the named benchmark
envelope.

## Documentation design

The MyST table of contents remains the canonical navigation. The implementation
reorganizes the approximation and integration material by method type rather
than accumulating unrelated APIs on one page. Planned pages may exist before
their runtime method, but they must be visibly labeled as planned and must not
contain executable Jaxstro examples.

The documentation group will include:

- integration from samples;
- fixed and weighted quadrature;
- adaptive one-dimensional quadrature;
- domains, measures, and transformations;
- differentiating an integral;
- multidimensional cubature;
- sparse-grid integration;
- quasi-Monte Carlo integration; and
- scientific and oscillatory integration.

Each substantive method page follows the researcher-first sequence:

1. the scientific question;
2. required background and assumptions;
3. mathematical objects;
4. derivation in LaTeX rendered by KaTeX;
5. algorithm and cost model;
6. JAX and derivative contract;
7. units and shapes;
8. executable example when implemented;
9. audit procedure and evidence; and
10. warranted claim and limitations.

MyST admonitions distinguish warnings, important assumptions, tips, and
connections. Cards or tabs are used only when they improve navigation or
comparison. Authored prose uses ASCII punctuation; mathematical symbols remain
in LaTeX. The pages contain no course, lesson, assignment, grading, or instructor
framing.

API pages are grouped by sampled, fixed, adaptive, multidimensional, sparse,
QMC, domain/measure, and result owners. The roadmap, SOTA assessment, contract
registry, evidence index, limitations, scorecard, and API ownership must change
together as capabilities become real.

## Program slices

### Phase A: one-dimensional foundation

1. A0: canonical namespace, result/error/work types, tolerances, domains,
   measures, and compatibility re-exports.
2. A1: sampled-data canonicalization and the complete fixed-rule family.
3. A2: adaptive Gauss-Kronrod, Clenshaw-Curtis, tanh-sinh, and Romberg
   controllers with typed failure states.
4. A3: replay/stop derivative rules, moving-bound validation, quantity boundary,
   and evidence artifacts.
5. A4: public method/API documentation, roadmap correction, comparisons,
   migration guidance, and Phase A release gate.

### Phase B: multidimensional methods

1. B0: hyperrectangle and multidimensional evaluator contracts.
2. B1: tensor rules and adaptive cubature.
3. B2: Smolyak and dimension-adaptive sparse grids.
4. B3: scrambled Sobol QMC and statistical calibration.
5. B4: multidimensional replay derivatives, quantity behavior, comparisons,
   documentation, and release gate.

### Phase C: scientific methods

1. C0: native simplex and spherical geometry.
2. C1: radial, angular, logarithmic, and projection measures.
3. C2: oscillatory, Bessel-weighted, and Hankel-type methods.
4. C3: downstream consumer evidence, documentation, and release gate.

Each numbered slice receives a test-first implementation plan and a checkpoint
review. Phase transitions require Anna's approval.

## Out of scope

This design does not authorize:

- automatic method selection;
- ODE integration or replacement of Diffrax;
- generic Monte Carlo inference, MCMC, posterior policy, Fisher policy, or
  optimal experimental design;
- physical model ownership that belongs in a sibling package;
- Progenax OED-demo cleanup;
- sibling-package migrations;
- ecosystem-wide quantity adoption;
- immediate deletion of legacy import paths; or
- a public superiority claim without matched evidence.

## Completion criteria

The capability program is complete only when:

1. every promised method has a mathematical contract and explicit owner;
2. public APIs have typed failure, error-evidence, work, shape, unit, and JAX
   semantics;
3. replay derivatives have independent analytic or central-FD evidence;
4. deterministic and statistical uncertainty remain distinguishable;
5. validation artifacts are reproducible and fresh;
6. documentation derives the methods and teaches their limitations;
7. sibling migrations are separately approved and verified;
8. stale compatibility paths are removed only after consumer evidence permits;
   and
9. every SOTA statement is bounded by the evidence that supports it.
