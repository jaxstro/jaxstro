# Phase A4 baseline fairness review

**Reviewed baseline revision:** `7c3a61243d4d9a2d1f19ce8b43b19adbba2371c8`

**Verdict:** Diagnostic evidence only. The numerical comparison design is strong and the baseline identifies a credible Romberg batching and forward-mode differentiation target, but the baseline is not yet suitable for a final optimization decision or public performance claim.

## Critical finding

The initial timing suite measured all method-case records sequentially in one Python process. Quadax uses cached compiled entry points, and some family and practical-choice records intentionally reuse configurations. Consequently, later compilation measurements could inherit cached internal compilation. A second whole-suite run repeated the same within-process contamination. Compilation evidence must use a fresh process for every method-case record.

## Important findings

1. Quadax status `2` is family-dependent. It means exhausted regional capacity for adaptive regional methods, but tolerance not met for Romberg methods.
2. Primal, JVP, and reverse-mode performance require separate warrants. A derivative failure must not invalidate an otherwise correct primal timing, and missing derivative truth must make AD timing ineligible.
3. The practical lane is a predeclared practical choice. It is not an independently selected winner for each library.
4. The public report must expose eligible-record counts, ratios, dispersion, derivative failures, and timing stability instead of only a single aggregate count.
5. CPU provenance needs an actual hardware model where the operating system makes one available.
6. Timing noise is substantial in some records. Aggregate trigger decisions must retain the approved ratio threshold, minimum-case rule, and separation by more than twice the larger median absolute deviation.

## Verified strengths

- Gauss-Kronrod rule and order controls are genuinely matched.
- Clenshaw-Curtis work is normalized to actual node calls.
- Romberg evaluation capacity is matched at 1025 evaluations.
- Analytic truths and independent NumPy Gauss-Legendre reference ladders are sound.
- Both precision lanes contain actual requested dtypes.
- Timed output trees are synchronized before clocks stop.
- Nonfinite values use portable JSON classifications.
- The initial report makes no unsupported winner claim.

## Diagnostic trigger readout

Under the approved primary float64, family-matched, representative-case policy, with both primals accurate and converged and each ratio separated by more than twice the larger timing dispersion:

| Mode | Significant eligible cases | Required | Diagnostic result |
| --- | ---: | ---: | --- |
| Warm scalar | 1 | 3 | Does not trigger |
| Normalized work | 1 | 3 | Does not trigger |
| VMAP, batch 16 | 2 | 3 | Does not trigger |
| VMAP, batch 128 | 3 | 3 | Triggers |
| JVP | 3 | 3 | Triggers |
| Compile | Not classifiable | 2 | Invalid until process isolation is fixed |

The three float64 VMAP-128 and JVP cases are smooth, oscillatory, and expensive-integrand Romberg comparisons. Their derivative truths pass. This is enough to justify profiling Romberg batching and JVP execution after the evidence corrections, but not enough to skip the corrected baseline rerun.

## Green conditions

- Fresh-process isolation for every timing record.
- Family-aware Quadax status normalization.
- Separate primal, JVP, and reverse-mode warrants.
- AD ineligibility when derivative truth is absent.
- Auditable researcher-facing report with stability and eligibility summaries.
- Independent post-fix review of the re-emitted clean baseline.

## Post-fix independent review

The clean baseline emitted from revision `20e8796afb4d8fb5af2d19ddd7fb03c462862674` is green for optimization profiling. All 78 deterministic identities have one matching fresh-process timing record with 21 repetitions. The reviewer found no remaining Critical fairness defects and verified that the status, mode-specific warrant, derivative-truth, wording, noise-disclosure, and report-auditability corrections are effective.

The independent mechanical recomputation found:

| Gate | Eligible cases | Required | Result |
| --- | ---: | ---: | --- |
| Warm, ratio above 1.25 | 2 | 3 | Does not trigger |
| Compile, ratio above 2.0 | 0 | 2 | Does not trigger |
| Work, ratio above 1.50 | 1 | 3 | Does not trigger |
| VMAP 16, ratio above 1.50 | 2 | 3 | Does not trigger |
| VMAP 128, ratio above 1.50 | 3 | 3 | Triggers |
| JVP, ratio above 1.50 | 2 | 3 | Does not trigger |
| Reverse mode, ratio above 1.50 | 0 | 3 | Does not trigger |

The qualifying VMAP-128 records are smooth Romberg at 2.78, oscillatory Romberg at 1.63, and expensive-integrand Romberg at 6.74. Each passes the correctness, comparison-label, representative-case, ratio, and two-MAD separation gates.

One Important provenance correction remains before final public performance evidence: the sandboxed run recorded `cpu_model=arm`. This does not invalidate paired within-machine ratios or the profiling trigger, but the final clean emission must capture the actual Apple chip and hardware model.

## Final provenance-corrected review

The final clean artifact from revision `2378e4d19d0e63bd13e17f8dbf6b499128f1ae02` records `Apple M2 Max / Mac14,5`, retains all 78 fresh-process records, and passes deterministic freshness. An independent recomputation exactly reproduces the stored mechanical decision: warm scalar, VMAP 128, and JVP fire in this run; VMAP 16, compile, work, and reverse mode do not.

Across the two clean isolated runs, VMAP 128 is the reproducible trigger: smooth, oscillatory, and expensive-integrand Romberg exceed the threshold in both runs. Warm and JVP repeatedly suggest shared Romberg execution overhead, but their minimum-count and two-MAD classification changed between runs. The final oscillatory Clenshaw-Curtis warm result also reversed direction and is treated as timing noise.

The optimization addendum is therefore authorized to profile Romberg VMAP 128 as the primary target, with scalar and JVP as secondary views of shared overhead. It must not expand to Clenshaw-Curtis, change numerical controllers, alter tolerances, or change work accounting. Acceptance requires two fresh post-change suites, improvement across all three Romberg VMAP-128 cases in both suites, unchanged correctness contracts, and no material scalar or JVP regression.
