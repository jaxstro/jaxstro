---
title: Jaxstro.quad Phase A4 comparison, performance, and hardening design
description: >-
  Approved design for the matched Jaxstro-versus-Quadax evidence gate,
  conditional optimization, public reporting, and Phase A release boundary.
---

# Jaxstro.quad Phase A4 comparison, performance, and hardening design

**Status:** Approved section by section on 2026-07-16.

## Purpose

Phase A4 closes the one-dimensional quadrature program with an honest external
comparison, evidence-driven optimization, researcher-facing documentation, and
the final Phase A release gate. The comparison measures the current package
before any new method family is added.

The goal is not to manufacture a claim that Jaxstro wins every case. The goal is
to determine, under declared and reproducible controls, where Jaxstro is more
accurate, more efficient, more robust, or more differentiable; where it reaches
practical parity; and where a measured deficiency justifies hardening or
optimization.

Quadax 0.2.13 is an independent benchmark comparator. It does not become a
runtime dependency, public owner, or hidden execution backend for Jaxstro.

## Approved order

The execution order is fixed:

1. commit and verify the current Phase A3 review fixes;
2. implement and emit the clean Phase A4 baseline comparison;
3. classify every observed difference before changing runtime code;
4. optimize or harden only when a predeclared gate is crossed;
5. rerun the unchanged benchmark after every runtime change;
6. complete the website, evidence, migration, and Phase A release records; and
7. only then brainstorm the next method family.

No multidimensional, sparse-grid, QMC, oscillatory, or other new quadrature
method enters this slice.

## Comparison architecture

The benchmark remains outside the runtime package:

```{mermaid}
flowchart LR
    A["Case catalog and analytic truth"] --> B["Jaxstro adapter"]
    A --> C["Quadax 0.2.13 adapter"]
    B --> D["Matched result records"]
    C --> D
    D --> E["Accuracy and work gates"]
    D --> F["Compile, warm, VMAP, and AD timings"]
    E --> G["Machine-readable evidence"]
    F --> G
    G --> H["Researcher-facing MyST report"]
    G --> I["Optimization decision"]
```

The implementation adds these owners:

- `scripts/benchmark_quad.py` owns the case catalog, comparator adapters,
  measurements, artifact rendering, and freshness checks;
- `docs/validation/quad-performance.json` is the machine-readable artifact;
- `docs/60-validation/numerical/quadrature-performance.md` is the public report;
  and
- `tests/unit/test_benchmark_quad_script.py` ratchets the schema, adapters,
  fairness labels, deterministic controls, and numerical evidence.

Supporting changes register the artifact in the evidence index, navigation,
validation table, route manifest, contract evidence, changelog, roadmap, and
`STATUS.md`.

Quadax is pinned only in a PEP 735 dependency group:

```toml
[dependency-groups]
benchmark = [
  "quadax==0.2.13",
]
```

It is absent from `[project].dependencies`, public installation extras, and the
ordinary development group.

## Two comparison lanes

The benchmark reports two different questions and never collapses them into one
ranking.

### Family-matched lane

This lane compares corresponding numerical families under identical integrand,
truth, tolerance, dtype, device, norm, domain, breakpoint, and capacity
controls. Every method pair carries one of these labels:

- `exact`: rule and important control semantics genuinely match;
- `strong_match`: the mathematical algorithm and capacity can be matched, with
  named implementation differences;
- `node_matched`: the local node budget matches but estimators or controllers
  differ;
- `family_matched`: the numerical family matches but the detailed construction
  differs; or
- `capability`: only the practical problem-solving capability is comparable.

Timing ratios never imply algorithm-level superiority for `family_matched` or
`capability` records.

### Practical best-method lane

This lane asks which library solves a declared research problem most effectively
when each uses its most appropriate shipped method. Method choices are frozen
from mathematical suitability and public method guidance before timing results
are inspected. Post hoc winner selection is prohibited.

## Method fairness matrix

| Jaxstro method | Quadax method | Label | Matching rule |
| --- | --- | --- | --- |
| `GaussKronrod(pair=21)` | `quadgk(order=21)` | `exact` | Same embedded pair, explicit tolerance, finite domains, and infinity norm |
| `AdaptiveClenshawCurtis(initial_order=17)` | `quadcc(order=16)` | `node_matched` | Both local rules use 17 nodes; estimator rescaling differs |
| `AdaptiveTanhSinh` | `quadts` | `family_matched` | Closest declared local work and native-default records are both retained |
| `Romberg(initial_level=1)` | `romberg(divmax=d)` | `strong_match` | Match the global trapezoid and Richardson capacity with $2^d+1$ evaluations |
| `RombergTanhSinh` | `rombergts` | `capability` | Quadax adds Richardson extrapolation; Jaxstro uses adjacent tanh-sinh level refinement |

For breakpoint cases, both libraries receive the same ordered segments and
region capacity. Reported Quadax Clenshaw-Curtis evaluations and normalized
actual node evaluations are stored separately because the public counter does
not include the additional endpoint node per local-rule call.

Semi-infinite scale-one transforms may be compared directly when their maps
match. Full-infinite results remain capability-level because Quadax and Jaxstro
use different transformations.

## Scientific case catalog

The catalog is fixed before the authoritative run.

| Case family | Question | Truth source |
| --- | --- | --- |
| Smooth finite | What is the baseline convergence and controller overhead? | Analytic |
| Polynomial and vector output | Are exactness, shapes, and payload scaling correct? | Analytic |
| Localized Gaussian | Does adaptive refinement discover a narrow region? | Analytic or independently converged high precision |
| Explicit breakpoint or kink | Does declared segmentation handle nonsmooth structure? | Analytic |
| Endpoint singularity | Does the declared tanh-sinh envelope converge honestly? | Analytic |
| Semi-infinite exponential | Is the improper-domain map accurate and efficient? | Analytic |
| Full-line Gaussian | Can the library solve a full-infinite reference problem? | Analytic |
| Oscillatory integral | How efficiently does refinement resolve cancellation? | Analytic |
| Expensive integrand | How important is controller overhead when function cost dominates? | Independently converged high precision |
| Narrow missed feature | Does the controller avoid or expose false convergence? | Independent reference |
| Nonfinite integrand | Does the implementation fail closed under its published contract? | Expected failure classification plus independent truth |

Quadax replaces nonfinite integrand samples with zero, while Jaxstro fails
closed. The benchmark reports this as a semantic difference. Neither library's
status decides mathematical correctness without independent truth.

## Numerical metrics

Correctness is a hard prerequisite for performance interpretation. Every record
contains, where applicable:

```{math}
:label: eq-a4-benchmark-metrics

I,
\qquad
\widehat{I},
\qquad
\left|\widehat{I}-I\right|,
\qquad
\frac{\left|\widehat{I}-I\right|}{\max(1,|I|)},
\qquad
\widehat{\epsilon},
\qquad
\frac{\mathrm{d}\widehat{I}}{\mathrm{d}\theta}.
```

Records also contain convergence or failure status, evaluations, normalized
evaluations when necessary, refinements, active regions, global levels, and
capacity use. Work fields are compared only when their semantics match.

The primary scientific lane uses float64. A bounded float32 lane checks
representability and transform behavior with dtype-appropriate tolerances; it
does not request precision that float32 cannot represent.

## Performance protocol

The first authoritative artifact is a CPU baseline with full machine
provenance. The harness is backend-neutral so identical cases can later be
emitted on GPU or TPU without redesign.

Each method-case pair records:

- lowering and compilation time separately;
- cached scalar execution time;
- VMAP execution for batches of 16 and 128;
- JVP execution for every supported pair;
- reverse-gradient execution where both methods support it; and
- executable or peak-memory evidence when the measurement is reproducible.

Timing follows the official JAX benchmarking contract:

- enable and record the selected precision before array construction;
- place inputs on the measured device before timing;
- JIT the outermost callable;
- compile every method-case pair independently;
- separate compilation from execution;
- synchronize every timed output with `jax.block_until_ready`;
- retain value, error, status, and work leaves so diagnostics are not removed as
  dead code;
- use identical dtype and device controls;
- interleave comparator repetitions to reduce temporal bias; and
- use at least 21 repetitions, reporting the median and dispersion.

Jaxstro replay differentiation and Quadax loop differentiation have different
semantics. Derivatives are first judged against analytic truth. Runtime ratios
are labeled by derivative policy and never treated as semantic equivalence.
Quadax Romberg methods are compared in JVP mode; unsupported reverse mode is
recorded as unsupported rather than as a performance failure.

## Optimization and hardening gates

Runtime changes are authorized only after the baseline is green for correctness
and at least one predeclared condition is met:

- Jaxstro is more than 25 percent slower across at least three representative
  matched and converged cases;
- compilation or memory cost exceeds Quadax by more than a factor of two across
  at least two representative cases;
- comparable work counts expose a repeatable algorithmic inefficiency;
- VMAP scaling is materially worse;
- replay AD has an avoidable performance or memory regression; or
- profiling identifies redundant evaluations, oversized traces, unnecessary
  diagnostics, or another concrete cost owner.

Warm execution within approximately 20 percent is practical parity and does not
justify complicating clean code. A single noisy case does not authorize an
optimization.

Every observed issue is classified before modification as one of:

- correctness defect;
- false convergence;
- error-estimator calibration problem;
- status or failure mismatch;
- JAX transformation defect;
- derivative semantic or performance defect;
- quantity inconsistency;
- work-accounting defect;
- compile, runtime, or memory inefficiency; or
- documentation or claim defect.

Correctness and false-success defects block Phase A regardless of speed. Each
runtime correction starts with a failing regression. Optimization must preserve
the existing numerical, status, work, AD, quantity, compatibility, and public
documentation contracts, then pass the unchanged benchmark.

## Evidence artifact contract

`scripts/benchmark_quad.py --emit` performs an authoritative run and writes the
complete artifact. `--check` recomputes deterministic configuration,
accuracy, work, status, and structural evidence. Wall-clock measurements are
informational and never become a noisy CI freshness threshold.

The artifact records:

- schema and benchmark versions;
- repository revision and dirty state;
- backend, device, platform, Python, JAX, Jaxstro, and Quadax versions;
- dtype, tolerances, capacities, batch sizes, and repetition controls;
- case definitions and truth provenance;
- method-pair labels and adapter settings;
- numerical, work, timing, transform, and failure results;
- limitations and excluded comparisons; and
- the explicit optimization decision.

Authoritative timing artifacts must come from a clean committed tree. If an
optimization occurs, the artifact retains baseline and optimized measurements,
their ratios, and the contract-parity verdict. Evidence is not overwritten in a
way that hides the original baseline.

## Researcher-facing website contract

The public report at
`docs/60-validation/numerical/quadrature-performance.md` explains:

1. what was compared and why;
2. the method-pair labels;
3. the mathematical cases and truth sources;
4. accuracy and reported-error calibration;
5. work and convergence behavior;
6. compilation, warm, VMAP, and AD performance;
7. failure-semantic differences;
8. hardware and reproducibility controls;
9. optimization decisions; and
10. warranted claims and limitations.

The adaptive-quadrature overview gains a concise MyST admonition that reports
benchmark status and links to the evidence. The API page explains the same
comparison labels. The website contains a capability map with `shipped and
validated`, `benchmarking`, `alpha`, `approved but planned`, and `intentionally
unsupported` states.

All prose remains research-software framing for new researchers. Mathematics is
written in LaTeX, and MyST tables, figures, and admonitions are used only when
they make the evidence easier to audit.

## Verification gates

The slice is complete only when all of these pass:

- benchmark schema and adapter unit tests;
- analytic-truth and method-choice tests;
- comparison-label and capacity ratchets;
- artifact emission and deterministic freshness tests;
- all existing quadrature unit, integration, and validation tests;
- the full repository regression suite;
- Ruff formatting and lint;
- MyPy;
- contract and evidence registry freshness;
- strict MyST build, route, link, identifier, alternative-text, and
  accessibility checks;
- independent benchmark-fairness review; and
- independent post-optimization review when runtime code changes.

## Completion claim

Passing Phase A4 permits Jaxstro to state only the comparative claims supported
by the emitted artifact. It does not establish universal hardware superiority,
arbitrary-precision behavior, multidimensional integration, or completeness for
specialized scientific integrals.

At Phase A4 completion, `jaxstro.quad` is an evidence-complete,
replay-differentiable, quantity-aware alpha, one-dimensional JAX quadrature
package with public comparison evidence. Phase B and Phase C methods remain
planned until their own designs, plans, implementations, and evidence gates
pass.
