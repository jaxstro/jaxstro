# Phase B1 whole-rung review

Date: 2026-07-18

Branch: `codex/quad-phase-b`

Review range: `8862295..ea3e298`

Verdict: **GREEN. Phase B1 is complete.**

## Findings

No Critical, Important, or high-confidence Minor findings remain.

The review covered the complete B1 source, test, reference-artifact, dependency,
plan, report, and status diff. It applied the repository's numerical, JAX,
public-API, provenance, dependency, and capability-honesty rules.

## Review corrections closed

The final Task 5 review sequence found and closed three evidence or API-contract
defects:

1. The reference generator originally formatted 80 digits after computing at
   `mpmath`'s default 15-digit precision. The generator now owns
   `mp.workdps(100)`, records reported and working precision separately, restores
   caller precision, and matches an independent 200-digit audit for all 24
   records.
2. Formula IDs were originally checked only by prefix. The validation layer now
   owns an independent exact family-to-formula-ID mapping and mutation tests
   reject stale, unrelated, and swapped identities.
3. The non-stop capability message originally omitted the approved B4 replay
   direction. All three B1 methods now share one DRY validator and exact-equality
   tests require:

   ```text
   <Method> supports only gradient="stop" in Phase B1;
   gradient="replay" is introduced in Phase B4
   ```

These corrections changed no numerical rule, scientific threshold, evaluation
control, limitation classification, runtime dependency, quantity boundary, or
accepted gradient mode.

## Numerical and architectural assessment

- Fixed tensor integration owns heterogeneous Cartesian products for the five
  declared one-dimensional rule families, exact pre-materialization work
  checks, target-dtype rule construction, unavailable estimator semantics, and
  stopped JAX results.
- Adaptive tensor integration owns exact represented-node identities,
  anisotropic error-per-new-node refinement, active-only cache reuse, bounded
  preflight, exact work/depth, and fail-closed nonfinite and roundoff behavior.
- Genz-Malik owns target-dtype local degree-7 and embedded degree-5 formulas,
  exact symmetry/moment evidence, deterministic split-axis evidence, and
  fail-closed overflow/conditioning checks.
- Adaptive cubature owns a bounded h-adaptive region store, authoritative
  active-leaf reductions, exact evaluation/region status precedence, scalar/JIT
  physical child skipping, documented VMAP logical-work semantics, and stopped
  replay metadata reserved for B4.
- The facade remains thin: one-dimensional ownership is unchanged, each B1
  method accepts only `gradient="stop"`, and replay is neither implemented nor
  implied.
- The truth layer is independent of Jaxstro runtime methods. It uses exact
  rational inputs, guarded arbitrary-precision closed forms, byte-exact
  freshness, exact formula identities, and redundant direct JAX formulas.

## Verification

Final post-correction gates:

```text
Task 5 validation and transformations:
169 passed in 307.94 s

Complete B1 scientific gate:
456 passed in 354.53 s

Phase A and facade compatibility gate:
408 passed in 142.27 s

Reference artifact:
byte-exact fresh

Ruff:
364 files passed lint and format checks

MyPy:
128 source files passed

Diff and worktree:
git diff --check passed; tracked worktree clean
```

Earlier complete-Quad evidence remains valid for the unchanged numerical
implementation: 788 tests passed in 1,035.80 s before the final provenance-test
and message-only corrections. The final post-correction gates reran every
changed behavior plus all B1 scientific and Phase A/facade compatibility paths.

## Honest limits carried forward

- Fixed Gaussian-12 tensor does not claim high-accuracy resolution of continuous
  kinks or discontinuities and has no embedded runtime estimator.
- Practical adaptive tensor certification is limited to dimensions 2 and 4
  under 32,768 evaluations; dimensions 5 through 8 require explicit B4
  compile/runtime/process/device-memory evidence.
- Adaptive cubature retains the continuous dimension-6
  `MAX_EVALUATIONS` limitation at the frozen 500,000-evaluation control.
- The exhaustive complete-Quad process previously peaked at 4.756 GB RSS.
  B4 must benchmark and harden adaptive-tensor and cubature memory behavior
  rather than treating this as an optimization success.
- Replay derivatives, quantity adoption, sparse grids, RQMC, sibling migrations,
  publication, push, and deployment remain outside B1.

## Next checkpoint

Pause at the completed B1 boundary. On explicit continuation, begin the reviewed
Phase B2 sparse-grid plan. Do not begin B3 or B4 early.
