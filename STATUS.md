# jaxstro — status

Updated: 2026-07-29

## Current checkpoint

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
- The contract inventory records 17 public modules, 18 callable-level
  contracts, 230 explicitly unclassified callables, and 174 inherited record
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
