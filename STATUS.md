# jaxstro — status

Updated: 2026-08-30

## Current checkpoint

- Foundations now has white-ground scientific figures, a two-channel running
  measurement, and researcher-facing learning pages. Every leaf page uses native
  MyST cards and semantic callouts for its compact predict-compute-audit practice
  loop; figure contracts require accessible vector structure, a white ground, and
  no warm fill panels. The functions, units, and scales page's SI, CGS, and
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
- The contract inventory records 17 public modules, 18 callable-level
  contracts, 235 explicitly unclassified callables, and 174 inherited record
  symbols.

## Next

1. Run the Phase B observed process/device-memory campaign when that scientific
   performance decision is scheduled.
2. Use the single consolidated checkpoint review to decide Phase B release
   closure without broadening the method or geometry scope.

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
