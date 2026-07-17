# Jaxstro quadrature Phase A4 Romberg optimization addendum

> **Execution rule:** Implement inline with test-first changes and independent code review at the regression and final-evidence checkpoints.

**Goal:** Remove capacity-proportional execution overhead from classical Romberg integration without changing its numerical controller, accepted formulas, work accounting, failure semantics, or replay derivative contract.

**Primary authorization:** VMAP batch 128 is the only trigger reproduced in both clean fresh-process baseline suites. Smooth, oscillatory, and expensive-integrand Romberg exceed the frozen ratio and two-MAD gates in both runs.

**Secondary evidence:** Scalar and JVP slowdowns repeatedly occur on the same Romberg records, but their aggregate trigger classification changes with timing noise. They are supporting views of shared overhead, not independent authorization to broaden scope.

## Profile evidence

The matched smooth Romberg record converges with 33 evaluations for every tested maximum capacity. XLA cost analysis nevertheless scales with the declared maximum:

| Maximum evaluations | Scalar FLOPs | VMAP-128 FLOPs | Scalar bytes accessed | VMAP-128 bytes accessed |
| ---: | ---: | ---: | ---: | ---: |
| 65 | 2,291 | 143,950 | 14,520 | 2,860,030 |
| 129 | 3,731 | 209,998 | 26,172 | 4,563,398 |
| 257 | 6,739 | 351,822 | 43,588 | 6,828,810 |
| 513 | 12,755 | 627,790 | 76,620 | 10,668,434 |
| 1025 | 24,787 | 1,171,022 | 140,644 | 17,564,322 |

At capacity 1025, the matched smooth record has:

| Mode | Jaxstro StableHLO characters | Quadax StableHLO characters | Jaxstro FLOPs | Quadax FLOPs |
| --- | ---: | ---: | ---: | ---: |
| Scalar | 118,021 | 56,442 | 24,787 | 337 |
| VMAP 128 | 162,284 | 86,537 | 1,171,022 | 64,650 |
| JVP | 173,378 | 61,514 | 36,939 | 396 |

These are compiler cost estimates, not wall-time claims. They identify the owner: `_romberg_initialize` and `_romberg_advance_level` allocate a lane array at the maximum capacity and call `_masked_evaluate`, which maps a conditional across every lane even when the accepted level activates only 3 initial or 16 new nodes.

## Scope and invariants

Modify only classical Romberg execution in `src/jaxstro/quad/_romberg.py` and its targeted tests unless profiling after the first change proves a separate replay-only owner.

Preserve exactly:

- the nested trapezoid nodes and their deterministic evaluation order;
- Richardson table and propagated roundoff-floor formulas;
- convergence, minimum-level, nonfinite, roundoff, and capacity statuses;
- reported evaluations, refinements, and levels;
- scalar, vector, complex, JIT, VMAP, JVP, replay, and stopped-gradient behavior;
- public APIs, defaults, tolerances, and matched benchmark controls.

The counted accumulator changes floating-point summation association from a padded array reduction to a sequential recurrence. Require exact status and work parity, plus dtype-aware numerical or ULP agreement for value, error, and tolerance. Do not claim bitwise value identity unless tests prove it across the validation catalog.

Do not modify Clenshaw-Curtis, Quadax adapters, tolerances, numerical controllers, or benchmark eligibility rules.

## Proposed execution change

1. In initialization, evaluate the static initial grid of `2**initial_level + 1` nodes with an exact-size `lax.map`. `evaluate_one` remains a scalar-node interface. Do not pad the grid to the maximum refinement lane count or silently vectorize the user integrand.
2. In each later level, use a dynamic counted loop from zero to `2**(level - 1)`. Accumulate the new-node sum, absolute-value sum, nonfinite flag, and roundoff flag directly.
3. Feed those accumulators into the unchanged trapezoid recurrence, floor calculation, Richardson row, convergence test, and status logic.
4. Keep accepted-formula replay on its existing static padded path during this optimization. A dynamic counted replay loop lowers to `while_loop` and cannot be transposed by compiled reverse mode. Share only recurrence code that remains transformation-safe; do not force sampling-path DRYness across incompatible JAX transformation requirements.

The dynamic counted primal loop may batch to the largest accepted level within a heterogeneous VMAP input. That is still proportional to accepted work rather than declared maximum capacity and preserves exact per-lane masking semantics. It must not be reused inside replay without a separate reverse-mode design and review.

## Design review correction

The first independent review blocked the original proposal because `2**(level - 1)` is a traced loop bound. JAX lowers that counted loop to `while_loop`. Forward-mode differentiation is supported, but compiled reverse-mode transposition is not. Romberg replay is executed inside a `custom_jvp`, and its tangent program must remain transposable for `jax.jit(jax.grad(...))` and heterogeneous `jax.jit(jax.vmap(jax.grad(...)))`.

The approved minimal design therefore optimizes only the primal solve. Replay retains static padded sampling. This is deliberate duplication at the execution-policy boundary, not accidental formula duplication.

## Test-first implementation plan

### Task 1: Ratchet away capacity-proportional compiled work

Add a targeted classical Romberg regression that compiles the same smooth problem at capacities 65 and 1025, verifies both accept with the same 33 evaluations, and asserts the larger capacity does not materially increase XLA FLOPs or bytes accessed. If the active backend does not expose cost analysis, skip with an explicit reason rather than weakening numerical tests.

The test must fail on the reviewed baseline, where scalar FLOPs rise from 2,291 to 24,787 and bytes from 14,520 to 140,644.

### Task 2: Replace padded masked evaluation

Refactor primal initialization and advance-level evaluation to exact-count loops. Keep Richardson, controller, and replay sampling code unchanged. Run the full Romberg unit suite immediately.

### Task 3: Prove transformation and replay parity

Run targeted integration tests for global replay derivatives, full-result JVP tangents, VMAP compilation, vector and complex payloads, nonfinite behavior, roundoff floors, oscillatory alias protection, and exact work counts.

Add a heterogeneous compiled replay regression using `jax.jit(jax.vmap(jax.grad(value)))` at parameters that accept different Romberg levels. Compare with analytic derivatives and ratchet that compiled reverse mode remains supported.

Compare deterministic benchmark records before and after with the existing freshness gate. No deterministic field may drift.

### Task 4: Independent code review

Review the diff for changed node order, inactive-node evaluation, status precedence, dtype drift, JAX transformation hazards, and accidental test weakening. Resolve all Critical and Important findings before timing.

### Task 5: Reprofile before full suites

Repeat the capacity ladder and matched smooth scalar, VMAP-128, and JVP cost analysis. Require capacity-independent compiler work within a small fixed structural envelope and no trace-size explosion.

### Task 6: Two clean post-change evidence suites

Emit two full fresh-process suites from the same clean optimized revision. Preserve the immutable baseline payload and store optimized evidence separately. Accept the optimization only when:

- all deterministic correctness and contract gates remain unchanged;
- smooth, oscillatory, and expensive Romberg VMAP-128 improve in both suites;
- no material scalar or JVP regression appears;
- work counts and failure semantics are unchanged;
- the authored report distinguishes reproducible gains from timing noise.

If any condition fails, revert the runtime optimization and retain the baseline artifact plus profiling report as the honest result.

## Implementation checkpoint

The reviewed minimal implementation separates dynamic exact-count primal sampling from static padded replay sampling. Focused verification passes 61 pre-existing and new Romberg/replay tests, plus targeted later-level nonfinite and roundoff aggregation tests.

Compiler cost estimates for the matched smooth record changed as follows:

| Mode | Baseline FLOPs | Optimized FLOPs | Reduction |
| --- | ---: | ---: | ---: |
| Scalar | 24,787 | 735 | 97.0 percent |
| VMAP 128 | 1,171,022 | 103,934 | 91.1 percent |
| JVP | 36,939 | 16,983 | 54.0 percent |

Fresh targeted VMAP-128 ratios, with values below one favoring Jaxstro, are 0.73 for smooth Romberg, 0.63 for oscillatory Romberg, and 0.43 for expensive-integrand Romberg. These are preliminary samples; the two full clean suites remain the acceptance gate.

Sequential accumulation changes two oscillatory deterministic diagnostics. The float32 exhausted solve moves by $4.66\times10^{-10}$ against a $7.67\times10^{-6}$ truth threshold. The float64 value drift remains below $10^{-13}$; its calibration ratio moves because the observed-error denominator is near zero. Status, convergence, work, truth gates, failure classifications, derivative policies, derivative truths, derivative thresholds, and derivative pass/support contracts remain unchanged.

The optimized evidence path therefore preserves the reviewed baseline subtree and applies an explicit contract-parity projection. The projection authenticates schema, controls, identities, dtypes, statuses, work, warrant thresholds, classifications, derivative policies, derivative truth, and derivative support/pass semantics while allowing independently truth-gated floating-point and calibration drift.
