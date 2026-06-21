# Spectra Hybrid Architecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a documented, tested hybrid spectra interface for local PHOENIX/NewEra atmosphere artifacts.

**Architecture:** Keep source ingestion host-side and optional-data-extra gated. Expose a small public boundary, `AtmosphereParams -> SpectrumResult`, with a lazy `NewEraBackend` convenience path and a prepared `PreparedSpectralGrid` path for JAX transforms.

**Tech Stack:** Python dataclasses, JAX PyTrees, optional `polars`/`zarr` from the `data` extra, MyST docs, pytest, ruff, mypy.

---

### Task 1: Document the Boundary

**Files:**
- Create: `docs/20-architecture/spectra-data-architecture.md`
- Modify: `docs/myst.yml`
- Modify: `docs/40-api/index.md`
- Modify: `docs/50-howto/index.md`
- Modify: `docs/60-validation/index.md`

**Step 1: Add the architecture page**

Document the exact public boundary:

```text
AtmosphereParams -> SpectrumResult
```

Explain ingestion artifacts, runtime layers, status policy, and dataset ownership.

**Step 2: Add the page to the MyST navigation**

Run: `cd docs && myst build`

Expected: the site builds with the spectra architecture page in the Architecture section.

### Task 2: Add Core Spectra Types

**Files:**
- Create: `src/jaxstro/atmospheres/spectra.py`
- Modify: `src/jaxstro/atmospheres/__init__.py`
- Test: `tests/unit/test_atmospheres_spectra.py`

**Step 1: Write tests for prepared interpolation**

Add tests covering:

```python
AtmosphereParams
Spectrum
SpectrumResult
PreparedSpectralGrid.spectrum(...)
```

Include midpoint bilinear interpolation, status for in-grid and out-of-grid
queries, `jax.jit`, `jax.vmap`, and `jax.grad` smoke coverage.

**Step 2: Implement the core**

Implement JAX PyTree dataclasses and bilinear interpolation over prepared
`teff`/`logg` axes.

**Step 3: Verify**

Run: `uv run pytest tests/unit/test_atmospheres_spectra.py`

Expected: all tests pass.

### Task 3: Add NewEra Backend

**Files:**
- Create: `src/jaxstro/atmospheres/newera.py`
- Modify: `src/jaxstro/atmospheres/__init__.py`
- Test: `tests/unit/test_atmospheres_newera_backend.py`

**Step 1: Write tests for a tiny processed artifact**

Build a synthetic converted artifact with four spectra on a 2x2 `teff`/`logg`
cell. Assert that `NewEraBackend.open(...).spectrum(...)` matches the prepared
interpolation result and that missing optional dependencies are not imported at
`jaxstro.atmospheres` import time.

**Step 2: Implement the backend**

Read `catalog.parquet`, open `newera_lowres_v3.zarr`, select an exact abundance
plane, load the enclosing local cell, and return `PreparedSpectralGrid`.

**Step 3: Verify**

Run: `uv run --extra data pytest tests/unit/test_atmospheres_newera_backend.py`

Expected: all tests pass.

### Task 4: Validate and Performance-Test

**Files:**
- Test: `tests/validation/test_atmospheres_spectra.py`
- Modify: `docs/60-validation/index.md`

**Step 1: Add validation/performance smoke tests**

Add a validation test that evaluates a prepared grid over a moderate wavelength
axis under `jit` and `vmap`, checking finite outputs and stable output shapes
without timing-sensitive assertions.

**Step 2: Run gates**

Run:

```bash
uv run --extra data pytest tests/unit/test_atmospheres.py tests/unit/test_atmospheres_spectra.py tests/unit/test_atmospheres_newera_backend.py tests/validation/test_atmospheres_spectra.py
uv run pytest
uv run ruff check src tests scripts/convert_newera_lowres.py
uv run ruff format --check src tests scripts/convert_newera_lowres.py
uv run mypy src
cd docs && myst build
```

Expected: all checks pass.
