# jaxstro Numerical Methods Roadmap Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Grow `jaxstro.numerics` into a compact, AD-safe, policy-light foundation for reusable scientific numerical methods across the jaxstro ecosystem.

**Architecture:** Keep method kernels in focused modules under `src/jaxstro/numerics/`, with public re-exports only after tests and docs exist. Each primitive must be JAX-native unless explicitly documented as host-side setup, and every differentiable public path needs finite-difference-vs-AD validation.

**Tech Stack:** JAX, `jax.numpy`, static JIT arguments for structural choices, `pytest`, Ruff, mypy, MyST, and existing `jaxstro.testing` gradient-validation patterns.

---

## Scope Rules

- Keep `jaxstro` domain-agnostic: no photometric filters, stellar-evolution policy, survey conventions, or population-model semantics.
- Prefer small, composable primitives over umbrella APIs.
- Add no new dependency unless a later design doc justifies it explicitly.
- Treat boundary behavior as a first-class API choice: clamp, fill, extrapolate, or reject must be named and tested.
- Update `docs/60-validation/index.md` whenever a new method family gains executable validation anchors.

## Chunk 1: B-spline Follow-Ups

**Goal:** Finish the evaluation-first spline foundation before adding broader interpolation families.

**Status:** Batch 1 implemented: design matrices, derivatives, fixed-knot
least-squares fitting, and `BSpline1D.derivative(...)`. Later spline batches
still cover antiderivatives, smoothing, adaptive knots, tensor products, and
optimized de Boor evaluation.

**Files:**
- Modify: `src/jaxstro/numerics/splines.py`
- Modify: `src/jaxstro/numerics/__init__.py`
- Modify: `tests/unit/test_splines.py`
- Modify: `tests/validation/test_grad_checks.py`
- Modify: `docs/10-theory/bsplines.md`
- Modify: `docs/40-api/index.md`
- Modify: `docs/60-validation/index.md`

**Tasks:**

1. Add `bspline_design_matrix(knots, x, degree=3)` as the explicit design-matrix spelling for basis evaluation on sample coordinates.
2. Add `bspline_derivative(knots, coeffs, x, degree=3, axis=-1)` using the standard derivative coefficient transform and degree-1 basis evaluation.
3. Add `fit_bspline_lstsq(knots, x, y, degree=3, sample_axis=0, rcond=None)` for fixed-knot least-squares coefficient fitting.
4. Extend `BSpline1D` with `.derivative(x)`.
5. Add tests for polynomial derivatives, AD parity, exact fixed-knot recovery, vector-valued fitting, exports, and eager validation.
6. Add FD-vs-AD checks for derivative wrt coefficients and least-squares fit wrt sample values.

**Deferrals:**
- Smoothing penalties, adaptive knots, tensor-product splines, custom VJPs, and optimized de Boor kernels.

## Chunk 2: Shape-Preserving Interpolation

**Goal:** Add monotone interpolators that avoid overshoot when the input data are monotonic.

**Status:** Implemented: cubic Hermite evaluation with supplied derivatives,
PCHIP-style slope construction, monotone cubic interpolation, and
`MonotoneTabulatedFunction1D`.

**Files:**
- Create or modify: `src/jaxstro/numerics/interpolation.py`
- Test: `tests/unit/test_interpolation.py`
- Validate: `tests/validation/test_grad_checks.py`
- Docs: `docs/10-theory/interpolation.md`

**Tasks:**

1. Add cubic Hermite evaluation with supplied derivatives.
2. Add monotone-slope construction for PCHIP-style interpolation.
3. Add `MonotoneTabulatedFunction1D` PyTree wrapper.
4. Validate monotonicity preservation, endpoint behavior, JAX transforms, and FD-vs-AD gradients away from knots.

**Deferrals:**
- Multidimensional monotone interpolation and extrapolation beyond named boundary policies.

## Chunk 3: Regular-Grid Interpolation

**Goal:** Support 2D/ND grid interpolation for atmosphere tables, tracks, and calibration surfaces without tying to any domain schema.

**Status:** Implemented: `regular_grid_interp(...)`, `bilinear_interp(...)`,
and `trilinear_interp(...)` with clamp/fill/reject boundary policies and
vector-valued payload support.

**Files:**
- Create: `src/jaxstro/numerics/regular_grid.py`
- Test: `tests/unit/test_regular_grid.py`
- Validate: `tests/validation/test_grad_checks.py`
- Docs: `docs/10-theory/regular-grid.md`

**Tasks:**

1. Add bilinear and trilinear interpolation helpers.
2. Add an ND regular-grid evaluator using static grid rank.
3. Support explicit boundary policy: clamp, fill, or reject.
4. Validate exact recovery of affine functions and gradients wrt values/interior coordinates.

**Deferrals:**
- Sparse grids, irregular triangulation, and scattered-data interpolation.

## Chunk 4: Numerical Differentiation And Diagnostics

**Goal:** Promote finite-difference diagnostics from test-local helpers into reusable public testing utilities.

**Status:** Implemented: central finite-difference gradients/Jacobians,
directional derivatives, AD-vs-FD comparison reports, and tolerance metadata in
`jaxstro.testing`.

**Files:**
- Modify: `src/jaxstro/testing/`
- Test: `tests/integration/test_grad_audit.py`
- Docs: `docs/60-validation/index.md`

**Tasks:**

1. Add central finite-difference gradient and Jacobian helpers.
2. Add directional derivative checks.
3. Add structured comparison reports with tolerance metadata.
4. Keep these utilities in `jaxstro.testing`, not `jaxstro.numerics`, because they diagnose code rather than participate in models.

## Chunk 5: Quadrature Expansion

**Goal:** Broaden fixed-node integration support while preserving AD flow through values rather than node generation.

**Status:** Implemented: Gauss-Laguerre nodes, Clenshaw-Curtis nodes, and
`cumulative_simpson(...)` panel-endpoint sums with explicit odd-sample shape
contract.

**Files:**
- Modify: `src/jaxstro/numerics/quadrature.py`
- Test: `tests/unit/test_quadrature.py`
- Validate: `tests/validation/test_grad_checks.py`
- Docs: `docs/10-theory/quadrature.md`

**Tasks:**

1. Add Gauss-Laguerre nodes.
2. Add fixed-order Clenshaw-Curtis nodes.
3. Add cumulative Simpson helpers where shape constraints are explicit.
4. Validate exact polynomial moments and gradients through integrands.

**Deferrals:**
- Adaptive quadrature and convergence-loop APIs.

## Chunk 6: Root-Finding And Monotone Inversion

**Goal:** Fill gaps around bracket discovery and monotone inverse interpolation.

**Status:** Implemented: `bracket_expand(...)`, `bisect_many(...)`, and
`monotone_inverse_interp(...)`, with explicit docs separating forward bracketing
from differentiable Newton-style solves.

**Files:**
- Modify: `src/jaxstro/numerics/rootfinding.py`
- Test: `tests/unit/test_rootfinding.py`
- Validate: `tests/validation/test_grad_checks.py`
- Docs: `docs/10-theory/rootfinding.md`

**Tasks:**

1. Add bracket-search utilities with static maximum expansion steps.
2. Add vectorized root solving over independent brackets.
3. Add monotone inverse interpolation for table-defined CDF-like functions.
4. Document AD policy for Brent-like methods before implementing any hybrid branchy solver.

**Deferrals:**
- Brent-like hybrid solvers until they have an explicit value-only or custom-AD
  policy.

## Chunk 7: Linear Algebra Primitives

**Goal:** Add small stable wrappers for weighted fits and matrix diagnostics.

**Status:** Implemented: `weighted_lstsq(...)`, `qr_solve(...)`,
`svd_solve(...)`, covariance/correlation helpers, and positive-definite jitter
utilities for small dense matrices.

**Files:**
- Modify: `src/jaxstro/numerics/linear_algebra.py`
- Test: `tests/unit/test_linear_algebra.py`
- Validate: `tests/validation/test_grad_checks.py`
- Docs: `docs/10-theory/linear-algebra.md`

**Tasks:**

1. Add weighted least squares.
2. Add QR/SVD solve wrappers for ill-conditioned systems.
3. Add covariance/correlation helpers.
4. Add positive-definite checks and jitter utilities.

**Deferrals:**
- Sparse, iterative, matrix-free, and custom-implicit-differentiation solvers;
  those remain solver-library territory.

## Chunk 8: Astro-Relevant Special Functions

**Goal:** Provide generic mathematical kernels often needed in astronomy without owning domain interpretation.

**Status:** Implemented: explicit-CGS Planck radiance kernels and log kernels,
normalized log-weight helpers, and Legendre/Chebyshev/Laguerre basis evaluators.

**Files:**
- Create: `src/jaxstro/numerics/special.py`
- Test: `tests/unit/test_special.py`
- Validate: `tests/validation/test_grad_checks.py`
- Docs: `docs/10-theory/special-functions.md`

**Tasks:**

1. Add stable Planck-function kernels with explicit unit assumptions.
2. Add normalized log-weight helpers.
3. Add orthogonal polynomial bases: Legendre, Chebyshev, Laguerre.
4. Consider spherical Bessel functions only if downstream requirements appear.

**Deferrals:**
- Spherical Bessel functions until a downstream package supplies a concrete
  normalization and stability contract.

## Chunk 9: Sampling And Grid Utilities

**Goal:** Add reusable grid and deterministic sampling helpers with explicit differentiability boundaries.

**Files:**
- Modify: `src/jaxstro/numerics/sampling.py`
- Create: `src/jaxstro/numerics/grids.py`
- Test: `tests/unit/test_sampling.py`, `tests/unit/test_grids.py`
- Docs: `docs/10-theory/grids.md`

**Tasks:**

1. Add log grids, geometric bin centers/edges, and conservative binning helpers.
2. Add stratified sampling helpers.
3. Add Sobol/Halton only if implementable without a dependency and with clear validation.
4. Document nondifferentiable histogram/binning boundaries.

## Chunk 10: Provenance And Trust Reports

**Goal:** Make numerical method trust easier to audit without binding to a specific data product.

**Files:**
- Create: `src/jaxstro/provenance.py` or `src/jaxstro/testing/reports.py`
- Test: `tests/unit/test_provenance.py`
- Docs: `docs/60-validation/index.md`

**Tasks:**

1. Add lightweight provenance dataclasses for method evidence.
2. Add deterministic JSON/Markdown report helpers.
3. Add a numerical-method trust report summarizing implemented primitives and validation anchors.

## Global Verification Gate

Run before merging any chunk:

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest -q
env -u VIRTUAL_ENV uv run --no-sync ruff check src tests
env -u VIRTUAL_ENV uv run --no-sync ruff format --check src tests
env -u VIRTUAL_ENV uv run --no-sync mypy src/jaxstro
cd docs && myst build
```
