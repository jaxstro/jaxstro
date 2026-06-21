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
| FD-vs-AD audits classify gradient contracts conservatively | Existing audit tolerances | Smooth, known-zero, blocked, surrogate, and validation-only cases | `tests/integration/test_grad_audit.py` |
| Spatial candidate gathering excludes self and preserves exact-kNN recall when stencil/capacity settings make recall possible | Exact set containment for small clouds | Regular, boundary, and clustered cases | `tests/unit/test_spatial.py` |
| Atmosphere data indexing does not vendor raw PHOENIX data | Fixture size guard and parser-only tests | Synthetic tiny NewEra-like files only | `tests/unit/test_atmospheres.py` |
| Prepared atmosphere spectra interpolate inside the loaded cell and fail closed outside it | Exact synthetic bilinear values and status codes | Midpoint spectra, clamped out-of-grid status, wrong abundance-plane status | `tests/unit/test_atmospheres_spectra.py` |
| Processed NewEra artifacts can be opened without raw text files | Exact synthetic Zarr/Parquet fixture values | `NewEraBackend.open(...).spectrum(...)` returns the expected wavelength and flux | `tests/unit/test_atmospheres_newera_backend.py` |
| Catalog-first atmosphere selection preserves provenance and reports raw-only staged data without selecting unavailable backends | Exact synthetic coverage rows and status strings | Processed backend match, raw-only backend-unavailable match, and no-match reason | `tests/unit/test_atmospheres_library.py` |
| Atmosphere coverage reports are deterministic | Exact Markdown/JSON ordering | Synthetic catalog rows summarized into stable coverage tables | `tests/unit/test_atmospheres_coverage.py`, `tests/unit/test_report_atmosphere_coverage_script.py` |
| Sonora and TLUSTY converters preserve raw semantic columns and archive provenance | Synthetic zip/tar readback with float32 storage checks | Sonora wavelength/`W/m2/m`; TLUSTY frequency/`F_nu`; source archives are not deleted | `tests/unit/test_sonora_conversion_script.py`, `tests/unit/test_tlusty_conversion_script.py` |
| Cross-library overlap validation is diagnostic, not a strict model-equality claim | Shape/domain/finite checks and normalized SED difference only | Synthetic overlapping and non-overlapping spectra | `tests/unit/test_atmospheres_overlap.py` |
| Prepared spectra run through JAX transform paths at moderate wavelength size | Shape and finite-output checks | `jit(vmap(...))` over 4096-wavelength spectra | `tests/validation/test_atmospheres_spectra.py` |

## Local evidence commands

Use the focused commands below when changing one subsystem:

```bash
uv run pytest tests/integration/test_grad_audit.py tests/unit/test_spatial.py
uv run pytest tests/unit/test_atmospheres.py tests/unit/test_atmospheres_spectra.py
uv run --extra data pytest tests/unit/test_atmospheres*.py tests/unit/test_*conversion_script.py
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
