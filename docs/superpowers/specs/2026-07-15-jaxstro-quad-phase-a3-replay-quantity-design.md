---
title: Jaxstro quad Phase A3 replay and quantity design
description: Approved design for replay derivatives, quantity-aware integration, and the replay-default evidence gate.
---

# Jaxstro quad Phase A3 replay and quantity design

## Status and scope

This specification defines Phase A3 of the approved `jaxstro.quad` capability
program. It is subordinate to
[](./2026-07-15-jaxstro-quad-capability-program-design.md) and begins from the
verified Phase A2 primal interfaces recorded in
[](../plans/2026-07-15-jaxstro-quad-phase-a2-adaptive-rules.md).

Phase A3 owns:

- normalized-reference replay for all five Phase A2 adaptive methods;
- `gradient="replay"` and the permanent `gradient="stop"` escape hatch;
- explicit-parameter and moving-bound derivative contracts;
- stopped error, tolerance, status, and work evidence;
- raw and opt-in quantity normalization through one numerical engine;
- quantity-valued result, error, norm, and tolerance leaves;
- separate primal and derivative validation evidence;
- researcher-facing derivations and contract documentation; and
- promotion of replay to the default after the complete evidence gate passes.

The Phase A3 quantity boundary belongs only to adaptive `quad.integrate`.
`quad.fixed`, `map_domain`, and `map_interval` retain their raw Phase A1
contracts and must eagerly reject quantity-valued domains. In particular, they
must reject `Infinite(unit is not None)` rather than silently discard its unit.
`Infinite()` and every existing raw fixed or transform call remain unchanged.

Phase A3 does not own:

- multidimensional integration;
- quasi-Monte Carlo integration;
- higher-order derivative claims;
- moving breakpoint derivatives;
- differentiation through adaptive control flow;
- sibling-package quantity migrations;
- deprecation or removal of superseded sibling paths;
- publication; or
- superiority claims relative to Quadax or another library.

Startrax, Gravax, Progenax, and the other sibling packages remain untouched by
this phase. Quantity support remains alpha and opt-in.

## Governing invariants

Phase A3 preserves the following ownership boundaries:

1. The adaptive controller is the sole owner of primal convergence, error,
   status, and work evidence.
2. Replay owns only the derivative of `QuadResult.value`.
3. The accepted adaptive decisions are evidence for a fixed formula, not a
   differentiable program.
4. Raw arrays remain the hot-kernel representation.
5. Quantity support is a static validation and normalization boundary, not a
   second integration implementation.
6. A finite derivative is not evidence that the primal solve converged or that
   the derivative is accurate.
7. Replay becomes the default only after every declared method and composition
   passes the predeclared evidence gate.

## Public API

Phase A3 preserves one adaptive entry point:

```python
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
    error_norm=MaxNorm(),
    gradient="stop",
)
```

During development and validation, `gradient="stop"` remains the default. At
the final Phase A3 promotion checkpoint, the default changes to
`gradient="replay"` if and only if the full gate in this specification passes.
There is no calendar delay or separate release requirement after the evidence
passes. `gradient="stop"` remains a permanent explicit mode after promotion.

The public API does not add `integrate_quantity`, `integrate_differentiable`, or
method-specific replay entry points. The accepted gradient strings are exactly
`"replay"` and `"stop"`; other values fail eagerly.

## Primal and replay architecture

### One private primal result

The raw adaptive implementation is refactored around a private result with two
owners:

```text
_PrimalSolve
|-- result: QuadResult
`-- evidence
    |-- _RegionalReplayEvidence
    |   |-- segment_local_lower
    |   |-- segment_local_upper
    |   |-- segment_id
    |   `-- active_mask
    `-- _GlobalReplayEvidence
        `-- accepted_level
```

The public `QuadResult` does not acquire replay fields. All evidence arrays have
static shapes determined by the existing capacities and method configuration.
Regional and global methods use distinct private evidence types rather than one
overfilled structure with irrelevant fields.

The existing regional controller already retains fixed-capacity normalized
region endpoints and an active mask. Phase A3 adds an original-segment identity
that propagates unchanged when a region is split. Accepted endpoints are
recorded locally within that original segment. This provenance lets replay use
live outer bounds while reconstructing interior segment endpoints from stopped
physical breakpoints. A globally normalized breakpoint is not sufficient:
replaying it through live outer bounds would move a breakpoint that the public
contract declares fixed.

The global Romberg engines retain the accepted level and reconstruct the exact
returned formula at that level, including the accepted extrapolation rather
than merely the finest unextrapolated base rule. Method configuration and fixed
rule data remain static rather than being copied into dynamic evidence.

### One custom JVP boundary

One private, all-positional custom-JVP core separates the primal solve from
derivative replay. The public keyword-oriented wrapper performs eager
validation and quantity normalization, creates one static configuration that
owns the callable, method, measure, norm, capacities, and gradient policy, and
passes only dynamic domain, argument, and tolerance PyTrees to the custom-JVP
core. Array-valued inputs are never hidden in `nondiff_argnums` or a traced
closure. A normal primal call returns the controller's exact `QuadResult` and
does not evaluate the integral a second time. When JAX requests a derivative,
the JVP rule:

1. executes the primal solve and obtains its private evidence;
2. applies `jax.lax.stop_gradient` to every evidence leaf;
3. reconstructs the accepted fixed formula from that evidence;
4. computes the tangent of the reconstructed formula with respect to the
   supported live inputs; and
5. returns a shape-matched numerical zero tangent for every floating or complex
   diagnostic leaf; and
6. returns a JAX `float0` tangent for every integer or Boolean diagnostic leaf.

JAX transposes this one JVP definition for reverse mode. Phase A3 does not
maintain separate hand-written forward and reverse replay formulas. Method
families provide only the reconstruction data and evaluation needed by the
shared derivative boundary.

This architecture preserves the exact primal summation and avoids the cost of
an unconditional second evaluation when the caller does not request a
derivative.

The zero-width primal branch synthesizes shape-compatible replay evidence even
though it does not execute adaptive refinement. A stopped breakpoint child is
rebuilt explicitly before replay; the fact that `Interval.breakpoints` are
ordinary PyTree children does not make them differentiable.

## Mathematical derivative contract

For `gradient="replay"`, Jaxstro differentiates the accepted fixed quadrature
approximation

```{math}
:label: eq-a3-replay-formula

\widehat{I}(\theta)
=
\sum_{r \in \mathcal{P}_{\mathrm{accepted}}}
\sum_{i=1}^{n}
w_{ri}(\theta)
f\!\left(x_{ri}(\theta),\theta\right),
```

while treating the accepted partition
`\(\mathcal{P}_{\mathrm{accepted}}\)` as fixed. Replay does not differentiate
sorting, interval selection, region subdivision, stopping, capacity logic,
breakpoint motion, status selection, or error estimation.

Regional replay freezes accepted endpoints in original-segment-local normalized
coordinates. It reconstructs physical nodes, weights, transformations, and
Jacobians from stopped segment-local evidence and live domain values. Stopped
accepted adaptive endpoints are never the source of replay nodes; explicitly
declared physical breakpoints are stopped by contract and do define their
segment boundaries. Global replay reconstructs exactly the accepted Romberg
level from the live domain and parameters.

For a finite segment with differentiable endpoints, replay uses the signed
affine formula directly:

```{math}
:label: eq-a3-signed-affine-map

x_i
=
\frac{a+b}{2}
+
\frac{b-a}{2}\xi_i,
\qquad
w_i^{(a,b)}
=
\frac{b-a}{2}w_i.
```

The differentiable replay path does not express this map through `minimum`,
`maximum`, `absolute`, or `sign`. This distinction is required at coincident
bounds, where the algebraically equivalent primal maps do not have equivalent
JAX derivatives.

The supported differentiable inputs are:

- floating or complex leaves supplied explicitly through `args`;
- finite interval bounds;
- supported semi-infinite boundary and transformation values;
- supported weighted-density parameters supplied through `args`; and
- numerical values inside a quantity-normalized call after unit metadata is
  validated and held static.

The following inputs are nondifferentiable controls or metadata:

- the callable;
- method configuration;
- classical Gaussian measure parameters in Phase A3;
- breakpoint positions;
- `epsabs` and `epsrel`;
- evaluation and region capacities;
- error-norm configuration; and
- units and dimensions.

Model parameters intended for differentiation must be explicit in `args` or in
a supported domain value. Hidden mutable state is outside the contract.
Passing a traced model or density parameter through a callable closure is
rejected; the same parameter is supported when supplied explicitly through
`args`.

### Moving finite bounds

For smooth finite limits, replay includes the dependence of the map and
Jacobian on both bounds. Validation checks the Leibniz identity

```{math}
:label: eq-a3-leibniz

\frac{\mathrm{d}}{\mathrm{d}\theta}
\int_{a(\theta)}^{b(\theta)} f(x,\theta)\,\mathrm{d}x
=
\int_{a(\theta)}^{b(\theta)}
\frac{\partial f}{\partial\theta}(x,\theta)\,\mathrm{d}x
+ f\!\left(b(\theta),\theta\right)b'(\theta)
- f\!\left(a(\theta),\theta\right)a'(\theta).
```

Reversed intervals preserve their orientation and derivative sign. A
zero-width interval still reconstructs its transformed formula during replay;
it is not replaced by a derivative-blind constant zero. This permits the
coincident-bound case to recover the boundary contribution when the two bounds
have different tangents.

Breakpoint positions are stopped in physical coordinates. Accepted regions
retain their original-segment identity and segment-local coordinates. Replay
therefore reconstructs an outer segment endpoint from its live domain bound
and an interior segment endpoint from its stopped physical breakpoint. A
change that crosses a breakpoint, refinement boundary, capacity boundary,
method boundary, or singularity declaration is outside the smooth replay
contract.

### Derivative order

Phase A3 guarantees first-order forward and reverse derivatives. Higher-order
differentiation is not claimed even if a particular composition happens to be
executable. A later capability must define and validate that contract before
the documentation recommends Hessians or higher derivatives.

## Result and failure semantics

Under `gradient="replay"`, only `QuadResult.value` participates in automatic
differentiation. Under `gradient="stop"`, the value is stopped as well. The
following leaves are always explicitly stopped:

- `QuadError.estimate`;
- `QuadError.norm`;
- `QuadError.kind`;
- `QuadError.confidence_level`;
- `QuadResult.tolerance`;
- `QuadResult.status`; and
- every `QuadWork` field.

Their primal values continue to describe the executed solve. The primal error
estimate never certifies derivative error.

Finite nonconverged solves differentiate the approximation they actually
returned while preserving their status:

- `MAX_EVALUATIONS`;
- `MAX_REGIONS`;
- `ROUNDOFF_LIMITED`;
- finite `DIVERGENCE_SUSPECTED`; and
- finite `ERROR_ESTIMATE_UNAVAILABLE`.

`INVALID_INPUT` and `NONFINITE_INTEGRAND` return a nonfinite primal value so a
failed solve cannot look scientifically usable. Replay derivatives are
undefined for those statuses. The API does not promise a particular NaN layout
across JVPs, VJPs, Jacobians, JIT, or VMAP because no one linear custom-JVP rule
can make such a layout invariant under every transposition and batching order.
Callers must check `status` before interpreting a derivative. A
`DIVERGENCE_SUSPECTED` result is differentiated only when its accepted formula
is finite. Replay never upgrades or hides a primal status.

### Complex differentiation envelope

Complex integration and complex differentiation are separate claims. Phase A3
uses the following JAX-compatible envelopes:

- real parameters to complex output use direct JVP/VJP checks and Jacobians of
  stacked real and imaginary output components;
- complex parameters to real scalar output use JAX's documented complex
  cotangent convention; and
- complex parameters to complex output are realified as a map from
  `\(\mathbb{R}^2\)` to `\(\mathbb{R}^2\)` unless the integrand is explicitly
  declared and independently checked as holomorphic.

The implementation never sets `holomorphic=True` merely to bypass a transform
error.

## Quantity boundary

### Activation and normalization

Quantity mode uses the same `quad.integrate` entry point. It performs eager
dimensional validation and normalization before any current raw dispatch calls
`jnp.asarray` or `jnp.result_type`, then wraps both the user integrand and any
weighted density before calling the existing raw adaptive and replay engines.

Mode resolution is explicit:

| Submitted call | Selected mode | Required outcome |
| --- | --- | --- |
| Any coordinate is a `Quantity` | quantity | All dimensional coordinates and breakpoints are compatible quantities |
| `Infinite.unit` is set | quantity | The declared unit is the coordinate unit |
| `epsabs` is a `Quantity` | quantity | A raw domain is interpreted as dimensionless unless another coordinate unit is present |
| No quantity trigger | raw | The integrand and tolerance outputs remain raw arrays |
| Quantity integrand output without a quantity trigger | invalid | Fail eagerly and explain that quantity `epsabs` activates a dimensionless quantity domain |

Quantity mode requires a stable quantity integrand output and a quantity
`epsabs`. Raw dimensionless coordinates are valid in quantity mode and use
`\(U_x=1\)`. Incidental `jnp.asarray` failures are not part of mode selection.

Finite and semi-infinite domains infer a coordinate unit from their dimensional
bound. All other bounds and breakpoints must be quantities compatible with that
unit and are converted to the selected representation. Mixed dimensional raw
and quantity coordinates fail eagerly.

The fully infinite domain gains optional static metadata:

```python
Infinite(unit=units.cm)
```

`Infinite()` remains the raw, dimensionless-compatible form. A fully infinite
dimensional quantity calculation must provide its coordinate unit because no
finite bound exists from which to infer one.

Because `Infinite` is shared by fixed and adaptive APIs, adding this metadata
does not authorize unit handling elsewhere. `quad.fixed`, `map_domain`, and
`map_interval` reject quantity-valued domains, including unit-bearing
`Infinite`, until a separate fixed-rule quantity contract is approved. Focused
regressions prove that these paths cannot silently strip metadata.

The quantity wrapper presents `Quantity` coordinates to the user integrand and
requires a stable `Quantity` output unit. In quantity mode it also presents
`Quantity` coordinates to `WeightedMeasure.density`, requires a quantity output
compatible with the declared `density_unit`, converts that output to the
declared representation, and unwraps it. The raw weighted-density contract is
unchanged in raw mode. This paired adapter is required for centimetre-versus-
metre representation invariance; an output-unit declaration alone cannot tell
a raw density callable which coordinate representation it received.
Differentiable density parameters are explicit in `args`.

### Result units

If the integrand, coordinate, and density have units `\(U_f\)`, `\(U_x\)`, and
`\(U_\rho\)`, the integral unit is

```{math}
:label: eq-a3-result-unit

U_I = U_f U_\rho U_x.
```

Lebesgue integration uses `\(U_\rho=1\)`. A dimensionless normalized
probability measure commonly has `\(U_\rho=U_x^{-1}\)`, which gives
`\(U_I=U_f\)`.

Quantity mode restores `\(U_I\)` on:

- `QuadResult.value`;
- `QuadResult.error.estimate`;
- `QuadResult.error.norm`; and
- `QuadResult.tolerance`.

The public `NamedTuple` field layout remains unchanged, but the annotations for
`QuadError.norm` and `QuadResult.tolerance` become quantity-capable rather than
`Array`-only. Raw calls retain their current array leaves. Raw and quantity
results must preserve compatible PyTree structure through primal execution,
`lax.cond`, JVP, VJP, JIT, and VMAP.

Status, work, error kind, and confidence level remain unitless. `epsabs` must be
a quantity compatible with `\(U_I\)`. `epsrel` may be a raw dimensionless
scalar or a dimensionless quantity. Incompatible bounds, breakpoints, density
declarations, integrand outputs, or tolerances fail eagerly with the quantity
layer's dimensional error types.

Unit metadata remains static under JIT. Replay differentiates only quantity
values. Quantity conversion must change numerical representation without
changing the represented physical integral.

### Derivative units

JAX tangents and physical Jacobians require distinct unit statements. A JVP
direction represents a physical perturbation in the selected input
representation, and its output tangent has integral unit `\(U_I\)`. A reported
Jacobian is computed from normalized raw values,

```{math}
:label: eq-a3-dimensional-jacobian

\frac{\partial I_{\mathrm{value}}}{\partial \theta_{\mathrm{value}}},
\qquad
U_{\partial I/\partial\theta}
=
\frac{U_I}{U_\theta}.
```

The derivative evidence records `parameter_unit`, `integral_unit`, and
`derivative_unit` explicitly. Bare `jax.grad`, `jax.jacfwd`, or `jax.jacrev`
over a `Quantity` PyTree does not synthesize quotient-unit algebra and is not a
Phase A3 claim. Researcher-facing examples differentiate selected raw values
inside a fixed-unit quantity calculation and then attach the declared quotient
unit. Metre-versus-centimetre tests verify the expected numerical rescaling and
the invariant physical derivative.

## Delivery slices

Phase A3 is delivered in the following reviewable order:

1. **A3.1: internal primal and evidence separation.** Preserve current public
   stop behavior while carrying private regional and global replay evidence.
2. **A3.2: finite-interval Gauss-Kronrod replay.** Establish the complete
   vertical derivative path, including explicit parameters, moving bounds,
   reversed bounds, and coincident bounds.
3. **A3.3: regional-family replay.** Reuse the regional contract for adaptive
   Clenshaw-Curtis and adaptive tanh-sinh.
4. **A3.4: global-family replay.** Add accepted-level reconstruction for
   Romberg and Romberg tanh-sinh.
5. **A3.5: opt-in quantity boundary.** Add dimensional normalization, fully
   infinite coordinate metadata, result-unit restoration, and conversion
   invariance.
6. **A3.6: complete evidence and documentation.** Run the full derivative,
   quantity, primal-regression, and documentation gates and publish the
   machine-readable evidence.
7. **A3.7: replay-default promotion.** Change the default from `"stop"` to
   `"replay"` only after every A3.6 gate passes.

Each slice leaves the complete existing suite green. A later method does not
weaken an earlier method's contract or validation threshold.

## Verification design

### Unit evidence

Unit tests establish:

- regional reconstruction agrees with the controller's accepted fixed formula;
- global reconstruction agrees at the accepted Romberg level;
- inactive capacity slots contribute exactly zero;
- stopped evidence cannot carry tangents;
- reversed and zero-width interval contracts;
- nonfinite primal values and explicitly undefined derivative semantics for
  invalid and nonfinite statuses;
- stopped diagnostic leaves;
- eager rejection of unsupported gradient strings;
- quantity conversion invariance;
- result, error, norm, and tolerance units; and
- eager failure for inconsistent dimensions.

### JAX integration evidence

Each supported adaptive method is exercised through:

- `jax.jvp` over the complete nested result, including exact zero and `float0`
  diagnostic tangents;
- selected `jax.vjp` projections proving that diagnostics contribute no
  cotangent;
- `jax.jacfwd` and `jax.jacrev` applied to
  `lambda ...: integrate(...).value`, never to the integer-bearing complete
  result;
- `jax.jit`;
- `jax.vmap`;
- explicit scalar and array parameters;
- moving finite bounds where applicable;
- real, vector, and array outputs;
- the separately declared complex differentiation envelopes; and
- raw-array calls plus fixed-unit quantity workflows differentiated through
  selected numerical values.

The tests also establish that no Python loop advances regions or global levels
and that the custom JVP does not differentiate the primal adaptive
`jax.lax.while_loop`.

### Numerical validation

A machine-readable derivative artifact records primal and derivative evidence
separately. Each case includes, where available,

```{math}
:label: eq-a3-validation-columns

I,
\qquad
\left|\widehat{I}-I\right|,
\qquad
\widehat{\epsilon}_{\mathrm{primal}},
\qquad
\frac{\mathrm{d}\widehat{I}}{\mathrm{d}\theta},
\qquad
\frac{\mathrm{d}I}{\mathrm{d}\theta},
\qquad
D_{\mathrm{FD}}.
```

The benchmark set covers:

- smooth finite integrals with analytic parameter derivatives;
- moving lower and upper bounds;
- reversed and coincident bounds;
- vector and complex outputs;
- improper tails;
- endpoint singularities within each method's supported envelope;
- weighted densities with explicit parameters;
- deliberately exhausted finite solves;
- invalid and nonfinite cases; and
- quantity conversion invariance.

Thresholds are predeclared, dtype-aware, and tied to each method's stated
envelope. Agreement between two JAX transforms alone is insufficient. The
validation distinguishes three different derivative questions:

1. replay AD versus the analytic derivative of the exact integral;
2. replay AD versus central differences of the explicitly frozen replay
   formula; and
3. central differences of independently rerun public adaptive solves as a
   partition-stability diagnostic, not a universal replay-correctness gate.

Every finite-difference column names which function it perturbs. A refinement
or stopping boundary may make the third quantity differ from the first two
without invalidating a correct frozen-formula replay derivative.

Before replay becomes the default, tolerance and capacity ladders record both
primal and derivative stabilization, accepted region counts or global levels,
and cases near but not across partition changes. The evidence artifact records
`parameter_unit`, `integral_unit`, `derivative_unit`, replay partition or level
metadata, and metre-versus-centimetre derivative rescaling where quantities are
used.

### Replay-default promotion gate

The default changes to `gradient="replay"` only when all five Phase A2 adaptive
methods pass:

- analytic exact-integral derivative comparisons;
- independent finite differences of the frozen replay formula;
- adaptive-rerun finite differences reported as a stability diagnostic;
- tolerance and capacity ladders for both primal and derivative stabilization;
- moving-bound Leibniz checks where applicable;
- forward and reverse automatic differentiation;
- JIT and VMAP compositions;
- declared payload and dtype cases;
- failure-status derivative tests;
- stopped diagnostic evidence tests;
- quantity representation and derivative-unit rescaling tests for the
  supported quantity envelope; and
- the full existing primal, numerical-validation, lint, type, and strict-docs
  gates.

If any gate fails, the default remains `"stop"`; the failure is recorded rather
than weakened or waived.

## Documentation design

Phase A3 updates the MyST table of contents and the method-family navigation
rather than adding disconnected pages. The required reading route is

```text
Quadrature
  -> Adaptive Quadrature
  -> Differentiating an Integral
  -> Auditing Derivatives
  -> Quadrature Replay Derivative Evidence
```

The new methods page is
`docs/20-methods/approximation-integration/differentiating-an-integral.md` and
appears immediately after `adaptive-quadrature.md` in `docs/myst.yml`. The
generated validation page is
`docs/60-validation/numerical/quadrature-replay-derivatives.md` and appears in
the Validation and evidence section. Phase A3 adds or revises:

- a dedicated **Differentiating an Integral** methods page;
- **Adaptive Quadrature**;
- the grouped `jaxstro.quad` API reference;
- **Auditing Derivatives**;
- the validation ledger and a generated derivative-evidence page; and
- quantity-aware integration examples.

The methods material is written for researchers who may be new to numerical
integration and JAX. It begins with the mathematical question, derives the
fixed-formula replay rule and the moving-bound identity, then connects those
ideas to the public API. It uses LaTeX for mathematical notation, MyST
admonitions for assumptions and failure boundaries, worked derivations,
contract tables, and explicit links between method, API, workflow, and evidence
pages. It contains no course or instructor framing.

The derivation route links explicitly to **Why JAX?**, **What is a
Derivative?**, **What JAX Differentiates**, and **Quantities**. It states the
conditions for differentiating under the integral sign; derives the exact
integral derivative and the accepted fixed-formula derivative separately;
explains why those derivatives may differ near adaptive decision boundaries;
works one analytic, AD, frozen-formula finite-difference, and adaptive-rerun
finite-difference example completely; derives quantity rescaling; and uses
admonitions to mark convergence, smoothness, breakpoint, invalid-status, and
unit assumptions.

The documentation distinguishes:

- the exact integral from the returned approximation;
- primal error evidence from derivative validation;
- replay from differentiation through adaptive decisions;
- convergence from finite execution;
- raw numerical representation from physical quantity meaning; and
- an implemented capability from a comparative superiority claim.

## Claim boundary and completion

Passing Phase A3 permits Jaxstro to state that:

- it provides replay-differentiable adaptive one-dimensional quadrature;
- the documented derivative cases satisfy their declared analytic and
  numerical gates; and
- quantity-aware integration is available as an alpha, opt-in boundary.

Phase A3 does not establish that Jaxstro is faster, more accurate, more robust,
or generally better than Quadax. Those statements require a later matched
comparison with identical integrands, tolerances, dtypes, devices, compilation
accounting, work metrics, and failure classification.

Phase A3 is complete only when A3.1 through A3.7 are implemented, all promotion
gates pass without weakened tests, the evidence artifact and researcher-facing
documentation are current, and the repository status records the next bounded
quadrature slice.
