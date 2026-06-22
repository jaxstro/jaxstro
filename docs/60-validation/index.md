---
title: Validation
description: >-
  Where quantitative claims meet their tests — Property | Tolerance | Measured |
  Anchor tables, FD-vs-AD grad audits, and convergence checks.
---

This section is where the docs earn trust. It will carry
`Property | Tolerance | Measured | Anchor` tables — the anchor being the test that
enforces each claim — alongside the FD-vs-AD gradient audit (principle
[1](../10-theory/index.md#p1-differentiability)) and convergence checks for the
numerical methods. Every quantitative claim elsewhere in these docs should resolve
to a row here.

## Validation anchors

| Property | Tolerance | Measured | Anchor |
| --- | --- | --- | --- |
| Constants match their recorded CGS values | Exact or documented floating tolerance per constant | Unit tests compare exported constants and derived values | `tests/unit/test_constants.py` |
| Unit-system conversions round-trip through CGS | Floating tolerance in the unit tests | Mass, length, time, velocity, and `G` conversions | `tests/unit/test_units.py` |
| `enable_high_precision()` configures JAX x64 before array creation | Exact config state | `jax_enable_x64=True` and highest matmul precision | `tests/unit/test_jaxconfig.py` |
| AD-safe numerical primitives avoid NaN gradients on guarded paths | Test-specific finite/close checks | Root-finding, interpolation, sampling, quadrature, and safe math | `tests/unit/test_numerics.py`, `tests/validation/test_grad_checks.py` |
| Root bracketing and monotone inverse tables expose explicit value/gradient contracts | Exact synthetic roots and FD-vs-AD checks where smooth | Fixed-count bracket expansion, independent bisection arrays, clamped monotone inverse interpolation, documented branchy-solver AD policy | `tests/unit/test_numerics.py`, `tests/validation/test_grad_checks.py` |
| Public finite-difference diagnostics report AD-vs-FD agreement with tolerance metadata | Exact analytic comparisons and custom-JVP mismatch detection | Central gradients, Jacobians, directional derivatives, structured pass/fail reports | `tests/integration/test_grad_audit.py` |
| Shape-preserving interpolation avoids monotone-table overshoot and differentiates inside stable limiter branches | Exact synthetic identities and FD-vs-AD checks | Cubic Hermite recovery, clamped boundaries, PCHIP turning-point slopes, monotone bounds, PyTree/JAX transforms | `tests/unit/test_interpolation_shape_preserving.py`, `tests/validation/test_grad_checks.py` |
| Regular-grid interpolation recovers affine tables and differentiates inside grid cells | Exact synthetic identities and FD-vs-AD checks | Bilinear/trilinear affine recovery, vector payloads, clamp/fill/reject policies, JAX transforms | `tests/unit/test_regular_grid.py`, `tests/validation/test_grad_checks.py` |
| Grid and stratified sampling helpers preserve explicit construction contracts | Exact grid identities, conservation checks, and FD-vs-AD checks where smooth | Log/geometric grids, bin centers, conservative rebin totals, one-sample-per-stratum uniforms | `tests/unit/test_grids.py`, `tests/unit/test_sampling.py`, `tests/validation/test_grad_checks.py` |
| B-spline basis, evaluation, calculus helpers, fitting, and tensor designs preserve local-basis invariants | Exact synthetic identities and FD-vs-AD checks | Partition of unity, nonnegativity, local support, Bernstein cubic, degree-1 parity with `interp1d`, de Boor parity, analytic derivatives, antiderivative/integral checks, roughness penalties, quantile knots, tensor-product designs, least-squares recovery, PyTree/JAX transforms | `tests/unit/test_splines.py`, `tests/validation/test_grad_checks.py` |
| Fixed-node quadrature and cumulative Simpson preserve their exactness contracts | Exact polynomial/moment identities and FD-vs-AD checks | Gauss-Legendre, Gauss-Laguerre, Gauss-Hermite, Clenshaw-Curtis, Hermite coefficients, cumulative Simpson panel sums | `tests/unit/test_quadrature.py`, `tests/unit/test_numerics.py`, `tests/validation/test_grad_checks.py` |
| Dense linear algebra helpers expose stable contracts for small fits and diagnostics | Exact synthetic identities and FD-vs-AD checks away from rank/cutoff boundaries | Weighted least squares, QR/SVD solves, covariance/correlation guards, positive-definite jitter search | `tests/unit/test_linear_algebra.py`, `tests/validation/test_grad_checks.py` |
| Optimization helpers expose differentiable objective diagnostics without owning an optimizer stack | Exact robust-loss identities, Armijo descent checks, and FD-vs-AD checks away from nonsmooth kinks | Squared/Huber/pseudo-Huber losses, weighted objective summaries, fixed-iteration Armijo backtracking, convergence diagnostics | `tests/unit/test_optimization.py`, `tests/validation/test_grad_checks.py` |
| Fixed-step ODE helpers preserve analytic toy-system behavior and scan-compatible AD paths | Analytic growth/decay tolerances, bounded harmonic-oscillator energy drift, and FD-vs-AD checks | Euler, midpoint/RK2, RK4, fixed-step dispatch, velocity-Verlet | `tests/unit/test_ode.py`, `tests/validation/test_grad_checks.py` |
| Special-function kernels keep unit and normalization contracts explicit | Direct formula parity, limiting-case checks, recurrence identities, and FD-vs-AD checks | CGS Planck functions and log kernels, normalized log weights, Legendre/Chebyshev/Laguerre bases | `tests/unit/test_special.py`, `tests/validation/test_grad_checks.py` |
| Numerical-method trust reports render deterministic evidence summaries | Exact JSON ordering and Markdown table checks | Evidence dataclasses, JSON/Markdown renderers, default numerics trust report coverage | `tests/unit/test_provenance.py` |
| FD-vs-AD audits classify gradient contracts conservatively | Existing audit tolerances | Smooth, known-zero, blocked, surrogate, and validation-only cases | `tests/integration/test_grad_audit.py` |
| Spatial candidate gathering excludes self and preserves exact-kNN recall when stencil/capacity settings make recall possible | Exact set containment for small clouds | Regular, boundary, and clustered cases | `tests/unit/test_spatial.py` |
| Atmosphere data indexing does not vendor raw PHOENIX data | Fixture size guard and parser-only tests | Synthetic tiny NewEra-like files only | `tests/unit/test_atmospheres.py` |
| Prepared atmosphere spectra interpolate inside the loaded cell and fail closed outside it | Exact synthetic bilinear values and status codes | Midpoint spectra, clamped out-of-grid status, wrong abundance-plane status | `tests/unit/test_atmospheres_spectra.py` |
| Processed NewEra artifacts can be opened without raw text files | Exact synthetic Zarr/Parquet fixture values | `NewEraBackend.open(...).spectrum(...)` returns the expected wavelength and flux | `tests/unit/test_atmospheres_newera_backend.py` |
| Catalog-first atmosphere selection preserves provenance and reports raw-only staged data without selecting unavailable backends | Exact synthetic coverage rows and status strings | Processed backend match, raw-only backend-unavailable match, and no-match reason | `tests/unit/test_atmospheres_library.py` |
| Atmosphere coverage reports are deterministic | Exact Markdown/JSON ordering | Synthetic catalog rows summarized into stable coverage tables | `tests/unit/test_atmospheres_coverage.py`, `tests/unit/test_report_atmosphere_coverage_script.py` |
| Sonora and TLUSTY converters preserve raw semantic columns and archive provenance | Synthetic zip/tar readback with float32 storage checks | Sonora wavelength/`W/m2/m`; TLUSTY frequency/`F_nu`; source archives are not deleted | `tests/unit/test_sonora_conversion_script.py`, `tests/unit/test_tlusty_conversion_script.py` |
| Local processed atmosphere artifacts match measured coverage | Exact local counts and finite sampled flux | Sonora 1440 valid spectra + 4 skipped resource forks; TLUSTY 981/551/690 spectra with raw archives preserved and ragged-grid Zarr subgroups | `tests/validation/test_atmospheres_local_artifacts.py` |
| Cross-library overlap validation is diagnostic, not a strict model-equality claim | Shape/domain/finite checks and normalized SED difference only | Synthetic overlapping and non-overlapping spectra | `tests/unit/test_atmospheres_overlap.py` |
| Prepared spectra run through JAX transform paths at moderate wavelength size | Shape and finite-output checks | `jit(vmap(...))` over 4096-wavelength spectra | `tests/validation/test_atmospheres_spectra.py` |

## Local evidence commands

Use the focused commands below when changing one subsystem:

```bash
uv run pytest tests/integration/test_grad_audit.py
uv run pytest tests/integration/test_grad_audit.py tests/unit/test_spatial.py
uv run pytest tests/unit/test_atmospheres.py tests/unit/test_atmospheres_spectra.py
uv run pytest tests/unit/test_interpolation_shape_preserving.py tests/validation/test_grad_checks.py
uv run pytest tests/unit/test_regular_grid.py tests/validation/test_grad_checks.py
uv run pytest tests/unit/test_grids.py tests/unit/test_sampling.py tests/validation/test_grad_checks.py
uv run pytest tests/unit/test_splines.py tests/validation/test_grad_checks.py
uv run pytest tests/unit/test_quadrature.py tests/unit/test_numerics.py tests/validation/test_grad_checks.py
uv run pytest tests/unit/test_numerics.py tests/validation/test_grad_checks.py
uv run pytest tests/unit/test_linear_algebra.py tests/validation/test_grad_checks.py
uv run pytest tests/unit/test_optimization.py tests/validation/test_grad_checks.py
uv run pytest tests/unit/test_ode.py tests/validation/test_grad_checks.py
uv run pytest tests/unit/test_special.py tests/validation/test_grad_checks.py
uv run pytest tests/unit/test_provenance.py
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
