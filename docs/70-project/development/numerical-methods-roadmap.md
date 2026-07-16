---
title: Numerical methods roadmap
description: >-
  A development checklist for growing jaxstro's AD-safe numerical foundation in
  small, validated chunks.
---

This page tracks numerical methods that belong in `jaxstro` because they are
generic, JAX-native, and broadly useful across downstream astronomy packages.
The boundary is deliberate: `jaxstro` provides trustworthy numerical primitives;
Fluxax owns filters and photometry workflows, Progenax owns population and
stellar-model policy, and data packages own their archive-specific schemas.

## Implementation Checklist

- [x] **Core 1D B-spline evaluation.** Basis evaluation, spline evaluation,
  clamped open-uniform knots, and a PyTree wrapper for fixed knots and
  coefficients.
- [x] **B-spline follow-ups, batch 1.** Design matrices, derivatives, and
  fixed-knot least-squares fitting.
- [x] **B-spline follow-ups, later batches.** de Boor evaluation,
  antiderivatives, smoothing penalties, adaptive knots, and tensor-product
  construction.
- [x] **Shape-preserving interpolation.** Cubic Hermite and monotone PCHIP-style
  interpolation for table-like functions where overshoot is unacceptable.
- [x] **Natural cubic spline interpolation.** Coefficient solve, clamped
  evaluation, SciPy parity tests, PyTree wrapper, and FD-vs-AD validation for
  smooth anchor-table interpolation.
- [x] **Regular-grid interpolation.** Bilinear, trilinear, and static-rank ND
  interpolation for gridded models with explicit clamp/fill/reject policies.
- [x] **Numerical differentiation diagnostics.** Public finite-difference,
  directional-derivative, and Jacobian-check utilities in `jaxstro.testing`.
- [x] **Complete fixed quadrature.** Canonical sampled-data rules, one shared
  Gaussian recurrence engine, Legendre, Jacobi, generalized Laguerre, both
  Hermite conventions, Clenshaw-Curtis, Fejer I and II, fixed tanh-sinh,
  explicit domains and measures, and a JAX-transformable fixed evaluator.
- [x] **Adaptive one-dimensional quadrature.** Gauss-Kronrod pairs, nested
  Clenshaw-Curtis, adaptive tanh-sinh, Romberg variants, deterministic status
  precedence, fixed-capacity work evidence, and an analytic and failure-envelope
  validation suite. Replay derivatives and quantity inputs remain later work.
- [x] **Root-finding and monotone inversion.** Bracket discovery, vectorized
  independent solves, and monotone inverse interpolation for CDF-like tables.
- [x] **Linear algebra primitives.** Weighted least squares, QR/SVD solve
  wrappers, covariance/correlation helpers, and positive-definite jitter tools.
- [x] **Astro-relevant special functions.** Stable Planck kernels, normalized
  log-weight helpers, and orthogonal polynomial bases without owning downstream
  domain semantics.
- [x] **Sampling and grid utilities.** Log grids, geometric bin centers/edges,
  conservative binning, and stratified sampling.
- [x] **Provenance and trust reports.** Deterministic JSON/Markdown summaries
  that connect numerical methods to their validation anchors.
- [x] **Optimization helpers, first slice.** Robust residual losses,
  objective summaries, fixed-iteration Armijo backtracking, and convergence
  diagnostics without becoming an optimizer stack.
- [x] **ODE helpers, first slice.** Fixed-step Euler, midpoint/RK2, RK4, and
  leapfrog or velocity-Verlet with scan-friendly APIs and analytic validation.
- [x] **Linear-operator helpers, first slice.** Dense, diagonal, scaled,
  sum/product, transpose/rmatvec, and block composition as small PyTrees.
- [x] **Distribution kernels, first slice.** Stable logpdf/CDF/inverse-CDF
  helpers for generic families and truncated variants.
- [x] **Geometry helpers, first slice.** Vector normalization, angular
  distance, rotations, quaternions, rigid transforms, and composition helpers.
- [x] **Autodiff helpers, first slice.** JVP/VJP/HVP, Jacobian-vector,
  vector-Jacobian, Gauss-Newton, and generic Fisher-style products.
- [x] **Runtime provenance, first slice.** Artifact hashing, environment
  snapshots, method manifests, and deterministic JSON/Markdown rendering.
- [x] **Random helpers, first slice.** Explicit key streams, resampling
  methods, seed manifests, and shape-stable APIs.
- [x] **Structured 1D mesh helpers, first slice.** Cell centers/volumes,
  face geometry, finite-volume stencils, and conservative remap.

## Acceptance Standard

Each checked item needs three pieces of evidence:

1. A public API with explicit boundary behavior.
2. Unit tests for mathematical identities and transform compatibility.
3. FD-vs-AD validation for differentiable paths, linked from
   [](../../60-validation/validation.md).

The detailed chunk plan lives in
`docs/plans/2026-06-22-jaxstro-numerics-roadmap.md`.
