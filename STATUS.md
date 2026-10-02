# jaxstro — status

Updated: 2026-09-25

## Current checkpoint

- 2026-09-25: The unit-aware scientific computing program page now includes
  Phase 2 feasibility probes for affine and logarithmic AD, mixed-unit linear
  algebra, compiled custom-AD loops, and direct JAX interoperation. Later
  phases specify angle composition, conversion domains and dynamic contexts,
  tangent/cotangent units, likelihood measures, compatible-unit invariance,
  specialization and accuracy checks, and v2 dependency/artifact migration.
  These are planned gates, not implemented capabilities or measured results.
  The strict 189-route MyST docs gate passed after the page update. Next:
  freeze the Phase 0 Stage 1 baseline and unit ledger before a comparison.
- 2026-09-25: The MyST Development section now has a planned Programs
  subsection. Its first unit-aware scientific computing program names the
  existing Progenax to Informax kinematic-design and survey-depth examples,
  their owners, and phases for architecture comparison, physical semantics,
  math coverage, AD and inference contracts, ecosystem workflows, and
  qualification. Anna selected physical-boundary Quantity checks for Phase 1
  and a Jaxstro-owned system as the preferred end state, with Phase 2 external
  comparisons informing the design. Anna selected a hybrid public math
  interface: `q.math` defines tested operations, with direct JAX calls where
  they preserve the same unit contract. Anna approved one public `Quantity`
  container for multiplicative, affine, and logarithmic representations with
  kind-specific conversion and operator rules. Compatible and fully specified
  affine or logarithmic conversions are direct; physical equivalencies and
  external calibrations require explicit context. Anna selected a verified
  hybrid AD interface and required custom AD and dimensionless edge cases in
  its unit and transform contracts. Anna selected photometry, dynamics, and
  learned-surrogate workflows as independent lanes and wants Quantity adopted
  across all ecosystem packages in their post-v1 development cycle, without
  adding a migration gate to their first releases. Anna selected `Quantity`
  as required for physical public inputs and outputs in v2, including
  dimensionless physical values. She selected `Quantity` through physical
  computations by default, with raw compiled kernels only behind justified,
  tested conversion boundaries. The program now explains how this supports
  scientific ML and Codex/Claude research workflows. Anna selected a clean
  major-version cutover: each qualified package removes raw physical public
  entry points in v2 and provides migration guidance; v1 gates stay separate.
  Anna selected predeclared, workload-specific performance limits with trace,
  lower, compile, warm execution, memory, and AD costs separated. Numerical
  limits remain open. The first representative measurement is the frozen
  Progenax to Informax Stage 1 forward, Jacobian, and Fisher workload, with
  fixed-unit and changed-unit call sequences. The program now records
  specialization, tracing, compiled-graph, AD, memory, and whole-workflow cost
  controls; no Quantity performance result has been measured.
- 2026-09-25: `jaxstro.quantity` unit identity now includes metadata, so JIT
  distinguishes tagged radians from untagged dimensionless input. Unit
  canonicalization preserves exact stored scales; structured serialization
  retains custom metadata. Scaled dimensionless `log`, `exp`, and raw-scalar
  arithmetic now use canonical values; invalid CGS scales are rejected.
  Focused quantity checks pass. Anna approved strict angle semantics: tagged
  angles remain tagged under untagged dimensionless scaling, while implicit
  angle/plain mixing is rejected. The scalar-output `quantity.grad` contract
  now returns derivative units as output/input; the quantity-aware quad replay
  test is its first consumer.
- 2026-09-24: cleanup and quad hardening, `main` at `3de93bc`, full gate green
  on all 8 parallel stages (run 36063431163). Task list:
  `docs/plans/2026-09-24-cleanup-ledger.md` (110 tasks: 62 done, 6 doing,
  42 todo). Handoff: `docs/plans/2026-09-24-handoff-quad-sota.md`.
- quad fixes measured on 2026-09-24: Gaussian tail weights (normal n=256,
  E[exp(6g)] error 1.3e18 -> 1.1e-16; NumPy hermgauss removed; rules cached,
  warm jit n=256 6.0 -> 0.029 ms); Gauss-Chebyshev NaN fixed; adaptive regions
  stored as (1+t, 1-t) so GK integrates x^-0.9 on [0,1] to 7.6e-11 where it
  returned NaN; sampled trapezoid on non-last axes and nonuniform Simpson;
  AdaptiveSmolyak no longer converges past its node capacity; replay fails
  closed at zero width in reverse mode; RQMC two-sided interval uses
  ln(4/alpha). Romberg is documented as smooth-integrand only.
- CI: the full gate runs `scripts/check.sh <stage>` as 8 parallel jobs (about
  22 min instead of about 50); `tests.yml` runs on every push to `main`.
- 2026-09-23: the ten integration tests that failed on `main` pass (55 owner
  tests). Causes: the docs-gate lifecycle harness faked `myst` after the gate
  moved to `npx --no-install myst`; nine Foundations figures lacked prose
  references; two route pins stayed at 181 after three pages were added (184);
  the landing-only card rule is removed so leaf pages may use cards.
- Foundations now has white-ground scientific figures, a two-channel running
  measurement, and researcher-facing learning pages. Every leaf page uses native
  MyST cards and semantic callouts for its compact predict-compute-audit practice
  loop; each opens with a concrete scientific problem rather than navigation
  copy; figure contracts require accessible vector structure, a white ground,
  and no warm fill panels. The functions, units, and scales page's SI, CGS, and
  solar-scale examples distinguish nominal conversions from Jaxstro's solar-mass
  compatibility scale.
- Published checkpoint: Python 3.13 runtime floor and the completed Lane–Emden
  ownership migration are on `origin/main`.
- Jaxstro is the shared Lane–Emden numerical owner. Progenax consumes it for
  Bonnor–Ebert and polytropic initial conditions; Hydrax consumes it for
  Bonnor–Ebert initial conditions and polytropic protostar structure.
- The release gate passed on Python 3.13.7: Ruff, MyPy over 138
  source files, all generated registries, the strict 181-route rendered-site
  audit, 2,890 non-slow tests with 24 declared optional-data skips, 580 ML
  integration tests, and a clean-wheel import.
- The release gate now installs the existing pinned `reference` group, so its
  arbitrary-precision multidimensional reference-generator test executes in
  the declared environment.
- **Riccati-Bessel `S_l` and `C_l` landed in `numerics.special` (2026-08-02),
  retiring the "spherical Bessel functions are deferred until a downstream
  contract exists" known-limit.** micrax's H-H scattering work supplied the
  contract. The Miller seed order is a **caller obligation** -- it must clear
  both the degree and the argument, not the degree alone -- because the
  downward sweep self-corrects only where `l > x`. A seed that is too low
  returns finite, smooth, wrong values; `riccati_wronskian_residual` is the
  gate that detects it.
- **`riccati_bessel_basis` returned an array carrying two scales (fixed
  2026-08-03).** Miller's mid-sweep rescale reached the `scan` carry and every
  later value, but never the outputs already stacked, so a rescale firing inside
  the retained window `[0, degree]` left the upper orders larger than the lower
  ones by exactly the rescale factor. The source documented this and declined to
  fix it as "no longer stepped on" -- it *was* being stepped on at the production
  seed order: 16 of 40 temperatures in micrax's H-H solve carried a corrupt node,
  worst Wronskian residual `1.0e+150`. The sweep now carries its cumulative
  rescale exponent alongside each value and reconciles the window onto one scale;
  the rescale factor became `2**500` so the scaling is exact and the result is
  provably independent of how often it fired. `riccati_bessel_at_order` never had
  the defect -- it keeps its saved value in the carry. After: 0 of 40
  temperatures corrupt, worst residual `4.0e-12`.
- Regenerating the contract registry surfaced pre-existing drift:
  `jaxconfig.ensure_jax_compilation_cache` (commit `7b1a116`) had never been
  added to the generated inventory, so `docs/validation/contracts.json` had
  been stale since then. Both that gap and this change are now recorded.
- The contract inventory records 18 public modules (composition added
  2026-09-24), 18 callable-level contracts, 235 explicitly unclassified
  callables, and 175 inherited record symbols.

## Next

next: QD-11 QAGS-style extrapolation remains queued — see `docs/plans/2026-09-24-handoff-quad-sota.md`
blocker: fluxax `0321bcc` and progenax `cc3a0e1` are committed but unpushed, alongside other sessions' commits; Anna to decide the push

1. QD-11 extrapolation, Phase 1 (`gradient="stop"`).
2. ARC-03/ARC-09: remove the flat `numerics` re-exports and compatibility
   modules (gravax migrated on its feature branch `551d4238`).
3. Run the Phase B observed process/device-memory campaign when that scientific
   performance decision is scheduled.
4. Use the single consolidated checkpoint review to decide Phase B release
   closure without broadening the method or geometry scope.
5. Freeze the Progenax to Informax Stage 1 example before a quantity-boundary
   comparison; the planned program does not authorize a sibling migration.

## Scientific boundary

- Lane–Emden focused owner evidence is green: exact solutions for
  `n = 0, 1, 5`, numerical first-zero checks for `n = 1.5, 3`, explicit output
  grids, and AD/finite-difference coverage passed 28 tests.
- No Lane–Emden equation, control, tolerance, or public scientific claim
  changed in this closeout; the changes align runtime/dependency and release
  infrastructure with the already-promoted owner.
- Phase B finite-hyperrectangle tensor integration, adaptive Genz–Malik
  cubature, Smolyak sparse grids, deterministic and randomized Sobol methods,
  accepted-formula replay, and heterogeneous quantity axes remain implemented.
- Phase B's observed process/device-memory campaign and single consolidated
  checkpoint review remain open. Phase C geometries and downstream adoption
  remain separate work; no universal quadrature-superiority claim is made.
