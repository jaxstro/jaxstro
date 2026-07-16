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
    |   |-- reference_lower
    |   |-- reference_upper
    |   `-- active_mask
    `-- _GlobalReplayEvidence
        `-- accepted_level
```

The public `QuadResult` does not acquire replay fields. All evidence arrays have
static shapes determined by the existing capacities and method configuration.
Regional and global methods use distinct private evidence types rather than one
overfilled structure with irrelevant fields.

The existing regional controller already retains fixed-capacity normalized
region endpoints and an active mask. Phase A3 preserves those arrays through
the private solve boundary. The global Romberg engines retain the accepted
level required to reconstruct the executed rule. Method configuration and
fixed rule data remain static rather than being copied into dynamic evidence.

### One custom JVP boundary

One internal custom-JVP boundary separates the primal solve from derivative
replay. A normal primal call returns the controller's exact `QuadResult` and
does not evaluate the integral a second time. When JAX requests a derivative,
the JVP rule:

1. executes the primal solve and obtains its private evidence;
2. applies `jax.lax.stop_gradient` to every evidence leaf;
3. reconstructs the accepted fixed formula from that evidence;
4. computes the tangent of the reconstructed formula with respect to the
   supported live inputs; and
5. returns stopped tangents for every diagnostic leaf.

JAX transposes this one JVP definition for reverse mode. Phase A3 does not
maintain separate hand-written forward and reverse replay formulas. Method
families provide only the reconstruction data and evaluation needed by the
shared derivative boundary.

This architecture preserves the exact primal summation and avoids the cost of
an unconditional second evaluation when the caller does not request a
derivative.

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

Regional replay freezes accepted endpoints in normalized reference
coordinates. It reconstructs physical nodes, weights, transformations, and
Jacobians from the stopped reference partition and the live domain values.
Stopped physical endpoints are never the source of replay nodes. Global replay
reconstructs exactly the accepted Romberg level from the live domain and
parameters.

The supported differentiable inputs are:

- floating or complex leaves supplied explicitly through `args`;
- finite interval bounds;
- supported semi-infinite boundary and transformation values;
- supported weighted-density parameters supplied through `args`; and
- quantity values after their unit metadata is validated and held static.

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

Breakpoint positions are stopped. A change that crosses a breakpoint,
refinement boundary, capacity boundary, method boundary, or singularity
declaration is outside the smooth replay contract.

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

`INVALID_INPUT` and `NONFINITE_INTEGRAND` must not return a silently plausible
zero or finite value derivative. Their value tangent is nonfinite. A
`DIVERGENCE_SUSPECTED` result is differentiated only when its accepted formula
is finite. Replay never upgrades or hides a primal status.

## Quantity boundary

### Activation and normalization

Quantity mode uses the same `quad.integrate` entry point and activates when a
domain carries quantity-valued bounds or a fully infinite domain declares a
static coordinate unit. It performs eager dimensional validation, chooses a
coordinate representation, wraps the user integrand, and calls the existing
raw adaptive and replay engines.

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

The quantity wrapper presents `Quantity` coordinates to the user integrand and
requires a stable `Quantity` output unit. It unwraps only numerical values for
the hot kernel. The existing `WeightedMeasure` numerical density contract
remains raw and uses its required static `density_unit` declaration for result
unit accounting; this avoids duplicating or breaking the raw measure engine.
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

Status, work, error kind, and confidence level remain unitless. `epsabs` must be
a quantity compatible with `\(U_I\)`. `epsrel` may be a raw dimensionless
scalar or a dimensionless quantity. Incompatible bounds, breakpoints, density
declarations, integrand outputs, or tolerances fail eagerly with the quantity
layer's dimensional error types.

Unit metadata remains static under JIT. Replay differentiates only quantity
values. Quantity conversion must change numerical representation without
changing the represented physical integral or derivative.

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
- exact status-dependent tangent behavior;
- stopped diagnostic leaves;
- eager rejection of unsupported gradient strings;
- quantity conversion invariance;
- result, error, norm, and tolerance units; and
- eager failure for inconsistent dimensions.

### JAX integration evidence

Each supported adaptive method is exercised through:

- `jax.jvp` and `jax.vjp`;
- `jax.jacfwd` and `jax.jacrev`;
- `jax.jit`;
- `jax.vmap`;
- explicit scalar and array parameters;
- moving finite bounds where applicable;
- real, vector, array, and complex outputs where claimed; and
- raw-array and quantity-valued calls.

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
envelope. Analytic derivatives and central finite differences are independent
references; agreement between two JAX transforms alone is insufficient.

### Replay-default promotion gate

The default changes to `gradient="replay"` only when all five Phase A2 adaptive
methods pass:

- analytic derivative comparisons;
- independent finite-difference comparisons;
- moving-bound Leibniz checks where applicable;
- forward and reverse automatic differentiation;
- JIT and VMAP compositions;
- declared payload and dtype cases;
- failure-status derivative tests;
- stopped diagnostic evidence tests;
- quantity tests for the supported quantity envelope; and
- the full existing primal, numerical-validation, lint, type, and strict-docs
  gates.

If any gate fails, the default remains `"stop"`; the failure is recorded rather
than weakened or waived.

## Documentation design

Phase A3 updates the MyST table of contents and the method-family navigation
rather than adding disconnected pages. It adds or revises:

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
