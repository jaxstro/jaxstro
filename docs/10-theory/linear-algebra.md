---
title: Linear algebra helpers
description: >-
  Weighted least squares, QR/SVD solves, covariance/correlation helpers, and
  positive-definite jitter utilities for small AD-safe scientific workflows.
---

Linear algebra is where many scientific workflows quietly become numerically
fragile. A fit is "just" a matrix solve until the design matrix is ill-conditioned.
A covariance matrix is "just" centered data until one variable has zero variance.
A kernel matrix is "just" positive definite until roundoff nudges its smallest
eigenvalue below zero.

jaxstro's linear algebra helpers are intentionally small. They wrap common JAX
operations with explicit contracts and validation, while leaving large sparse
systems, iterative solvers, and optimizer-grade linear algebra to dedicated
libraries such as Lineax.

## Weighted least squares

`weighted_lstsq(design, y, weights=None, rcond=None)` solves

```{math}
\min_\beta \sum_i w_i\,\left\lVert (X\beta)_i - y_i \right\rVert^2.
```

The implementation applies $\sqrt{w_i}$ to each row of the design matrix and to
the matching observation, then calls JAX's least-squares routine. That keeps the
mathematics ordinary: zero weight means the row contributes nothing, vector-valued
responses solve multiple right-hand sides at once, and gradients flow through the
response values.

Use this when the design matrix is fixed or slowly varying and you want the fitted
coefficients inside a differentiable calculation. If the rank of the design matrix
changes during differentiation, the map is not smooth; use the result as a
diagnostic or regularize the problem before treating the gradient as scientific
evidence.

## QR and SVD solve wrappers

`qr_solve(A, b)` solves square or tall full-rank systems through a reduced QR
factorization. For a tall matrix it returns the least-squares solution. This is
the good default when you expect a well-conditioned design matrix and want a
stable solve without forming normal equations.

`svd_solve(A, b, rcond=None)` solves with an explicit truncated pseudoinverse:
singular values at or below `rcond * max(s)` are discarded. This is slower than QR
but makes the ill-conditioned direction policy visible. The truncation cutoff is a
non-smooth boundary, so gradients are trustworthy only away from singular values
crossing that threshold.

## Covariance and correlation

`covariance_matrix(samples, weights=None, rowvar=False, ddof=1)` treats rows as
observations and columns as variables by default. Weighted covariance uses
per-observation weights and an explicit denominator `sum(weights) - ddof`.

`correlation_from_covariance(covariance)` divides by the standard deviations on
both axes. Zero variance is guarded before division, so a constant variable
produces finite zeros instead of `NaN`. `correlation_matrix(...)` composes the two
steps from samples.

That zero-variance guard is a value contract, not magic. A variable with no
variance has no meaningful correlation with anything else; returning finite zeros
lets downstream code fail closed or mask explicitly.

## Positive-definite jitter

`is_positive_definite(A, tol=0)` symmetrizes the matrix and checks the eigenvalues.
`add_diagonal_jitter(A, jitter)` adds $jI$ without touching off-diagonal entries.
`positive_definite_jitter(A, initial_jitter, growth, max_steps)` searches a fixed
geometric sequence of diagonal shifts and returns:

```python
A_shifted, jitter, success
```

Already-positive-definite matrices return zero jitter. Otherwise the scan chooses
the first tested diagonal shift that makes the symmetrized matrix positive
definite. The fixed scan count is deliberate: the result composes with JAX
transforms and reports failure through `success` rather than hiding it behind a
Python exception.

## What is deliberately not here

jaxstro does not own sparse solves, iterative Krylov methods, matrix-free linear
operators, or custom implicit differentiation rules. Those deserve a solver
library. The foundation layer provides the small dense operations that sibling
packages repeatedly need, with validation anchors in [](../60-validation/index.md)
and call signatures in [](../40-api/index.md).
