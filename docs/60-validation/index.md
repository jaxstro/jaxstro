---
title: Validation
description: >-
  Where quantitative claims meet their tests — Property | Tolerance | Measured |
  Anchor tables, FD-vs-AD grad audits, and convergence checks.
---

This section is where the docs earn trust. The table records
`Property | Tolerance | Measured | Anchor`: the bounded property, the comparison
policy, what is measured, and the tests that enforce the claim. It includes the
FD-vs-AD gradient audit (principle
[1](../10-theory/index.md#p1-differentiability)), discrete spatial contracts,
quantity transforms, provenance-card integrity, and convergence checks for the
numerical methods. Each row states the bounded claim its cited tests enforce;
absence from this table is not evidence of validation.

Read the rows with the corresponding explanations: gradient contracts in
[](../10-theory/index.md), discrete and exact neighbor contracts in
[](../10-theory/spatial.md), implemented quantity semantics in
[](../10-theory/quantities.md), and source-backed registry pages in
[](../40-api/provenance/index.md).

## Validation anchors

| Property | Tolerance | Measured | Anchor |
| --- | --- | --- | --- |
| Constants match their recorded CGS values | Exact or documented floating tolerance per constant | Unit tests compare exported constants and derived values | `tests/unit/test_constants.py` |
| Unit-system conversions round-trip through CGS | Floating tolerance in the unit tests | Mass, length, time, velocity, and `G` conversions | `tests/unit/test_units.py` |
| Astrometry and coordinate transforms preserve the documented ICRS-to-Galactic convention and round trips | Exact reference identities and focused floating tolerances | Parallax, proper-motion conversion, Cartesian/spherical transforms, and the adopted Galactic rotation | `tests/unit/test_astrometry.py`, `tests/unit/test_coords.py` |
| Coordinate gradients state explicit contracts at smooth and singular coordinate locations | Contract-specific AD/FD agreement, finite blocked results, or expected zeros | Public astrometry and coordinate cases at ordinary points, poles, origins, and coincident geometries | `tests/validation/test_coords_grad_audit.py` |
| The implemented quantity layer preserves explicit dimension, unit, parser, serialization, basis, constant, equivalency, and math contracts | Exact rational equality, CGS scale parity, structured errors, and focused floating tolerances | `Quantity` PyTree arithmetic, unit parsing/formatting, registry lookup, constants provenance, opt-in equivalencies, and dimension-aware math; this does not claim ecosystem adoption | `tests/unit/test_quantity_dimensions.py`, `tests/unit/test_quantity_units.py`, `tests/unit/test_quantity_parser.py`, `tests/unit/test_quantity_serialization.py`, `tests/unit/test_quantity_bases.py`, `tests/unit/test_quantity_constants.py`, `tests/unit/test_quantity_equivalencies.py`, `tests/unit/test_quantity_math.py`, `tests/unit/test_quantity_arithmetic.py`, `tests/unit/test_quantity_registry.py` |
| Quantity values survive representative JAX transforms | Exact or close transform outputs | `jit`, `vmap`, and `grad` through quantity construction, conversion, where, reductions, and math wrappers | `tests/validation/test_quantity_jax_transforms.py`, `tests/validation/test_quantity_math_gradients.py` |
| `enable_high_precision()` configures JAX x64 before array creation | Exact config state | `jax_enable_x64=True` and highest matmul precision | `tests/unit/test_jaxconfig.py` |
| AD-safe numerical primitives avoid NaN gradients on guarded paths | Test-specific finite/close checks | Root-finding, interpolation, sampling, quadrature, and safe math | `tests/unit/test_numerics.py`, `tests/validation/test_grad_checks.py` |
| Root bracketing and monotone inverse tables expose explicit value/gradient contracts | Exact synthetic roots, fixed trace masks, explicit exhaustion/missing-bracket flags, and FD-vs-AD checks only where claimed | Safeguarded IQI/secant/midpoint proposals, no post-convergence evaluations, JIT/VMAP values, physical-cost `lax.map`, and documented value-first AD policy | `tests/unit/test_bracketed_root.py`, `tests/unit/test_numerics.py`, `tests/validation/test_bracketed_root_algorithms.py` |
| Certified implicit scalar roots expose an IFT derivative only behind explicit assumptions and runtime evidence | Analytic and central-FD gradient agreement; fail-closed NaN value/gradient on rejected certificates | Linear, positive quadratic, exponential, rejected uniqueness, zero slope, and no-while JAXPR checks | `tests/unit/test_implicit_root.py`, `tests/validation/test_implicit_root_gradients.py`, `docs/validation/implicit-root-gradients.json` |
| Safeguarded scalar root costs are recorded by function evaluations as well as bounded warm timings | Exact schema/units; no hardware-dependent speed threshold | Bisection versus safeguarded-hybrid evaluations, executed iterations, absolute/relative residuals, and warm scalar wall time on five analytic cases | `scripts/benchmark_rootfinding.py`, `docs/validation/rootfinding-performance.json`, `tests/unit/test_benchmark_rootfinding_script.py` |
| Public finite-difference diagnostics report AD-vs-FD agreement with tolerance metadata | Exact analytic comparisons and custom-JVP mismatch detection | Central gradients, Jacobians, directional derivatives, structured pass/fail reports | `tests/integration/test_grad_audit.py` |
| Cubic interpolation preserves its boundary and smoothness contracts | Exact synthetic identities, SciPy natural-spline parity, and FD-vs-AD checks | Cubic Hermite recovery, natural cubic spline coefficients/evaluation, clamped boundaries, PCHIP turning-point slopes, monotone bounds, PyTree/JAX transforms | `tests/unit/test_interpolation_shape_preserving.py`, `tests/validation/test_grad_checks.py` |
| Regular-grid interpolation recovers affine tables and differentiates inside grid cells | Exact synthetic identities, executable documentation, and FD-vs-AD checks | Bilinear/trilinear affine recovery, arbitrary trailing payload axes, whole-payload fill, eager axis/reject validation, clamp/fill/reject policies, JAX transforms | `tests/unit/test_regular_grid.py`, `tests/integration/test_regular_grid_docs.py`, `tests/validation/test_grad_checks.py` |
| Grid and stratified sampling helpers preserve explicit construction contracts | Exact grid identities, conservation checks, and FD-vs-AD checks where smooth | Log/geometric grids, bin centers, conservative rebin totals, one-sample-per-stratum uniforms | `tests/unit/test_grids.py`, `tests/unit/test_sampling.py`, `tests/validation/test_grad_checks.py` |
| Structured 1D mesh helpers preserve finite-volume geometry and conservation contracts | Exact geometry/stencil identities, conservation checks, and FD-vs-AD checks for remap values | Uniform edges, cell centers/widths/volumes, face geometry, neighbor stencils, divergence, face averaging, conservative remap | `tests/unit/test_meshes.py`, `tests/validation/test_grad_checks.py` |
| Random stream and resampling helpers preserve deterministic key, validation, and shape contracts | Exact fixed-key behavior, invalid eager-input rejection, uniform zero-total fallback, exact residual integer counts, executable documentation, and JIT shape checks | Key streams, folded-in streams, seed manifests, systematic/stratified/residual resampling | `tests/unit/test_random.py`, `tests/integration/test_random_docs.py` |
| B-spline basis, evaluation, calculus helpers, fitting, and tensor designs preserve local-basis invariants | Exact synthetic identities and FD-vs-AD checks | Partition of unity, nonnegativity, local support, Bernstein cubic, degree-1 parity with `interp1d`, de Boor parity, analytic derivatives, antiderivative/integral checks, roughness penalties, quantile knots, tensor-product designs, least-squares recovery, PyTree/JAX transforms | `tests/unit/test_splines.py`, `tests/validation/test_grad_checks.py` |
| Fixed-node quadrature and cumulative Simpson preserve their exactness contracts | Exact polynomial/moment identities and FD-vs-AD checks | Gauss-Legendre, Gauss-Laguerre, Gauss-Hermite, Clenshaw-Curtis, Hermite coefficients, cumulative Simpson panel sums | `tests/unit/test_quadrature.py`, `tests/unit/test_numerics.py`, `tests/validation/test_grad_checks.py` |
| Dense linear algebra helpers expose stable contracts for small fits and diagnostics | Exact synthetic identities, invalid eager-input rejection, executable documentation, and FD-vs-AD checks away from rank/cutoff boundaries | Weighted least squares, finite weights, positive covariance normalization, covariance/correlation guards, QR/SVD solves, and positive-definite jitter search | `tests/unit/test_linear_algebra.py`, `tests/integration/test_linear_algebra_docs.py`, `tests/validation/test_grad_checks.py` |
| Autodiff product helpers match explicit dense derivatives and independent finite-difference checks | Dense-Jacobian/Hessian parity and finite-difference gradient-direction checks | JVP, VJP, HVP, Jacobian-vector, vector-Jacobian, Gauss-Newton, and empirical Fisher-style products | `tests/unit/test_autodiff.py`, `tests/validation/test_grad_checks.py` |
| Distribution kernels preserve normalization, monotone CDFs, inverse-CDF round trips, and smooth interior gradients | Numerical integration tolerances, exact round trips within floating tolerance, analytic alpha=-1 limits, and central-FD-vs-AD checks through the removable singularity | Normal, lognormal, finite power law (including normalization/logpdf/CDF/PPF through alpha=-1), and truncated normal helpers | `tests/unit/test_distributions.py`, `tests/validation/test_grad_checks.py` |
| Geometry helpers preserve rotation, quaternion, and rigid-transform identities | Exact identity/round-trip checks and FD-vs-AD checks away from degenerate angles/vectors | Normalization, angular distance, axis-angle rotations, quaternions, rigid transforms, inverse and composition helpers | `tests/unit/test_geometry.py`, `tests/validation/test_grad_checks.py` |
| Optimization helpers expose differentiable objective diagnostics without owning an optimizer stack | Exact robust-loss identities, Armijo descent checks, and FD-vs-AD checks away from nonsmooth kinks | Squared/Huber/pseudo-Huber losses, weighted objective summaries, fixed-iteration Armijo backtracking, convergence diagnostics | `tests/unit/test_optimization.py`, `tests/validation/test_grad_checks.py` |
| Fixed-step ODE helpers preserve analytic toy-system behavior and scan-compatible AD paths | Analytic growth/decay tolerances, bounded harmonic-oscillator energy drift, and FD-vs-AD checks | Euler, midpoint/RK2, RK4, fixed-step dispatch, velocity-Verlet | `tests/unit/test_ode.py`, `tests/validation/test_grad_checks.py` |
| Linear operators preserve matrix algebra while remaining PyTrees | Dense parity for matvec/rmatvec/to_dense and FD-vs-AD checks through operator leaves | Dense, diagonal, scaled, sum, product, transpose, and block-diagonal operators | `tests/unit/test_operators.py`, `tests/validation/test_grad_checks.py` |
| Special-function kernels keep unit and normalization contracts explicit | Direct formula parity, limiting-case checks, recurrence identities, and FD-vs-AD checks | CGS Planck functions and log kernels, normalized log weights, Legendre/Chebyshev/Laguerre bases | `tests/unit/test_special.py`, `tests/validation/test_grad_checks.py` |
| Numerical-method trust reports render deterministic evidence summaries | Exact JSON ordering and Markdown table checks | Evidence dataclasses, JSON/Markdown renderers, default numerics trust report coverage | `tests/unit/test_provenance.py` |
| Runtime provenance manifests render deterministic evidence records | Exact schema, sorted JSON/Markdown checks, and missing-file behavior | Artifact hashes, environment snapshots, method manifests, JSON and Markdown renderers | `tests/unit/test_runtime_provenance.py` |
| The provenance-card registry resolves source symbols and assertion-bearing validation node IDs and renders fresh reference pages | Exact schema, symbol resolution, pytest collection, and byte-for-byte generated-page equality | Approved card families, unique card IDs, importable code references, collectable validation tests, assertion-bearing test bodies, and deterministic MyST output | `tests/validation/provenance_cards/test_registry.py::test_validation_references_collect_and_assert_behavior`, `tests/validation/provenance_cards/test_registry.py::test_generated_pages_equal_fresh_rendering` |
| FD-vs-AD audits classify gradient contracts conservatively | Existing audit tolerances | Smooth, known-zero, blocked, surrogate, and validation-only cases | `tests/integration/test_grad_audit.py` |
| Spatial candidate gathering and exact fixed-radius pairs preserve their distinct recall, symmetry, cutoff, capacity, and overflow contracts | Exact set containment or brute-force equality for small clouds; explicit overflow flags | Regular, boundary, clustered, cutoff-straddling, and undersized-capacity cases; Morton dense allocation is restricted to supported power-of-two grids | `tests/unit/test_spatial.py`, `tests/unit/test_spatial_grid_contract.py` |
| Generic spectral types preserve coordinate, point/bin sampling, physical semantic, provenance, conversion, resampling, and structured status contracts | Exact identities and focused floating tolerances | Wavelength/frequency and density transforms, point and conservative-bin remapping, fail-closed coverage | `tests/unit/test_spectra_types.py`, `tests/unit/test_spectra_transforms.py`, `tests/unit/test_spectra_resampling.py` |
| Prepared spectral stencils evaluate only inside host-selected complete cells or approved simplices | Exact affine/geometric fixtures, fixed shapes, and AD-vs-FD agreement | Linear and positive-log interpolation, `jit`, `vmap`, status arrays, and local derivatives | `tests/unit/test_spectra_stencils.py` |
| Source-specific atmosphere adapters convert every vertex to canonical surface `F_lambda` before parameter interpolation | Analytic conversion fixtures and strict product identity | NewEra `1e3`, BOSZ per-angstrom `10`, Sonora `1e-6`, TLUSTY `4 pi H_nu` plus density-coordinate conversion | `tests/unit/test_atmospheres_newera_backend.py`, `tests/unit/test_atmospheres_bosz.py`, `tests/unit/test_atmospheres_sonora.py`, `tests/unit/test_atmospheres_tlusty.py` |
| Atmosphere interpolation policies are selected from deterministic real-artifact holdouts without invented thresholds | Strict primary-metric win and no secondary regression | Positive-log accepted for NewEra; linear accepted for representative BOSZ/OSTAR; Sonora and BSTAR remain `POLICY_NOT_VALIDATED` | `tests/unit/test_spectral_validation.py`, `tests/validation/test_atmosphere_holdouts.py`, `docs/validation/atmosphere-interpolation.json` |
| Exact-product registry routing fails closed on unratified or unavailable science | Exact synthetic and real status codes | Product-scoped coverage, validated artifacts, `POLICY_NOT_VALIDATED` for Sonora/BSTAR, no compatibility aliases | `tests/unit/test_atmospheres_library.py`, `tests/validation/test_atmospheres_spectra.py` |
| Atmosphere coverage reports are deterministic | Exact Markdown/JSON ordering | Synthetic catalog rows summarized into stable coverage tables | `tests/unit/test_atmospheres_coverage.py`, `tests/unit/test_report_atmosphere_coverage_script.py` |
| Sonora and TLUSTY converters preserve raw semantic columns and archive provenance | Synthetic zip/tar readback with float32 storage checks | Sonora wavelength/`W/m2/m`; TLUSTY frequency/`H_nu`; source archives are not deleted | `tests/unit/test_sonora_conversion_script.py`, `tests/unit/test_tlusty_conversion_script.py` |
| Local processed atmosphere artifacts match measured coverage | Exact local counts and finite sampled flux | Sonora 1440 valid spectra + 4 skipped resource forks; TLUSTY 981/551/690 spectra with raw archives preserved and ragged-grid Zarr subgroups | `tests/validation/test_atmospheres_local_artifacts.py` |
| Cross-library overlap validation is diagnostic, not a strict model-equality claim | Shape/domain/finite checks and normalized SED difference only | Synthetic overlapping and non-overlapping spectra | `tests/unit/test_atmospheres_overlap.py` |
| Real prepared spectra are I/O-free after preparation and pass transform/performance gates | Finite 64-bin values, fixed shapes, AD-vs-FD tolerance, recorded timings without a fabricated performance ceiling | Supported NewEra/BOSZ/OSTAR paths, evidence-gated Sonora/BSTAR paths, `jit`, `vmap`, local gradient, host preparation, first JIT, cached and batched evaluation | `tests/validation/test_atmospheres_spectra.py`, `tests/unit/test_benchmark_spectra_script.py`, `docs/validation/spectra-performance.json` |

## Local evidence commands

Use the focused commands below when changing one subsystem:

```bash
uv run pytest tests/integration/test_grad_audit.py
uv run pytest tests/integration/test_grad_audit.py tests/unit/test_spatial.py
uv run pytest tests/unit/test_quantity_*.py tests/validation/test_quantity_*.py
uv run pytest tests/unit/test_spectra_*.py tests/unit/test_atmospheres_*.py
uv run pytest tests/unit/test_interpolation_shape_preserving.py tests/validation/test_grad_checks.py
uv run pytest tests/unit/test_regular_grid.py tests/validation/test_grad_checks.py
uv run pytest tests/unit/test_grids.py tests/unit/test_sampling.py tests/validation/test_grad_checks.py
uv run pytest tests/unit/test_meshes.py tests/validation/test_grad_checks.py
uv run pytest tests/unit/test_random.py
uv run pytest tests/unit/test_splines.py tests/validation/test_grad_checks.py
uv run pytest tests/unit/test_quadrature.py tests/unit/test_numerics.py tests/validation/test_grad_checks.py
uv run --no-sync pytest tests/unit/test_numerics.py tests/validation/test_grad_checks.py
uv run --no-sync python scripts/benchmark_rootfinding.py --check
uv run --no-sync pytest tests/unit/test_implicit_root.py tests/validation/test_implicit_root_gradients.py
uv run pytest tests/unit/test_linear_algebra.py tests/validation/test_grad_checks.py
uv run pytest tests/unit/test_autodiff.py tests/validation/test_grad_checks.py
uv run pytest tests/unit/test_distributions.py tests/validation/test_grad_checks.py
uv run pytest tests/unit/test_geometry.py tests/validation/test_grad_checks.py
uv run pytest tests/unit/test_optimization.py tests/validation/test_grad_checks.py
uv run pytest tests/unit/test_ode.py tests/validation/test_grad_checks.py
uv run pytest tests/unit/test_operators.py tests/validation/test_grad_checks.py
uv run pytest tests/unit/test_special.py tests/validation/test_grad_checks.py
uv run pytest tests/unit/test_provenance.py
uv run pytest tests/unit/test_runtime_provenance.py
uv run --extra data pytest tests/unit/test_atmospheres*.py tests/unit/test_*conversion_script.py
uv run --extra data pytest tests/validation/test_atmospheres_local_artifacts.py
uv run pytest tests/validation/test_atmospheres_spectra.py
```

Use the broader gate before publishing or handing a branch to downstream
packages:

```bash
uv run pytest
uv run ruff check src tests
uv run mypy src
```

The validation table is intentionally compact. Detailed numerical derivations
belong in [](../10-theory/index.md); this page records the executable anchors.
