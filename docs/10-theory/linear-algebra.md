---
title: Linear algebra helpers
description: >-
  Weighted least squares, QR/SVD solves, covariance/correlation helpers, and
  positive-definite jitter with explicit numerical and differentiation contracts.
---

Linear algebra is where many scientific workflows quietly become numerically
fragile. A fit is "just" a matrix solve until the design loses rank. A covariance
is "just" centered data until its normalization denominator vanishes. A kernel
matrix is "just" positive definite until roundoff moves an eigenvalue through
zero.

jaxstro keeps this layer deliberately small. The following example exercises its
four main responsibilities without hiding the numerical policy:

```python
from jaxstro.jaxconfig import enable_high_precision

enable_high_precision()  # before creating JAX arrays

import jax.numpy as jnp

from jaxstro.numerics.linear_algebra import (
    correlation_from_covariance,
    covariance_matrix,
    positive_definite_jitter,
    qr_solve,
    svd_solve,
    weighted_lstsq,
)

x = jnp.array([0.0, 1.0, 2.0, 3.0])
design = jnp.stack([jnp.ones_like(x), x], axis=1)
observations = jnp.array([1.0, 3.0, 5.0, 20.0])
weights = jnp.array([1.0, 1.0, 1.0, 0.0])

unweighted_coeffs = weighted_lstsq(design, observations)
weighted_coeffs = weighted_lstsq(design, observations, weights)
qr_coeffs = qr_solve(design[:3], observations[:3])
svd_coeffs = svd_solve(design[:3], observations[:3])

samples = jnp.array(
    [
        [1.0, 2.0, 5.0],
        [2.0, 4.0, 5.0],
        [3.0, 6.0, 5.0],
        [4.0, 8.0, 5.0],
    ]
)
covariance = covariance_matrix(samples)
correlation = correlation_from_covariance(covariance)

matrix = jnp.diag(jnp.array([-0.03, 2.0]))
shifted, jitter, success = positive_definite_jitter(
    matrix,
    initial_jitter=1.0e-3,
    growth=10.0,
    max_steps=4,
)

assert jnp.allclose(unweighted_coeffs, jnp.array([-1.6, 5.9]))
assert jnp.allclose(weighted_coeffs, jnp.array([1.0, 2.0]))
assert jnp.allclose(qr_coeffs, svd_coeffs)
assert jnp.all(jnp.isfinite(correlation))
assert jnp.allclose(correlation[2], 0.0)  # constant third variable
assert success and jnp.isclose(jitter, 0.1)
assert jnp.linalg.eigvalsh(shifted).min() > 0.0
```

The zero weight is a declared modeling choice: it removes the final observation
from the objective. It is not an automatic outlier detector. Likewise, the
returned jitter is the first successful member of the requested geometric
sequence, not the smallest possible matrix perturbation.

:::{figure} ./figures/linear-algebra-contracts.webp
:name: fig-linear-algebra-contracts
:alt: Four regression observations with an outlier and measured weighted and unweighted fits, beside matrix eigenvalues before and after selected diagonal jitter

Both panels are computed from public jaxstro APIs. The weighted fit ignores the
declared zero-weight observation and recovers `(1, 2)`. The diagonal shift moves
the smallest eigenvalue from `-0.03` to `0.07`. This fixed example demonstrates
API policy; it is not a robust-regression or nearest-matrix benchmark.
:::

## Learning objectives

After this chapter, you should be able to state the objective solved by weighted
least squares, recognize conditioning evidence, and interpret diagonal jitter
as an explicit numerical intervention rather than hidden model physics.

### Concept check: what did the weights change?

Predict which observations dominate a weighted fit. Compute weighted and
unweighted solutions, then audit residuals, matrix conditioning, and units
before claiming one fit is scientifically preferable.

## Weighted least squares

`weighted_lstsq(design, y, weights=None, rcond=None)` solves

```{math}
\min_\beta \sum_i w_i\,\left\lVert (X\beta)_i-y_i\right\rVert^2.
```

The implementation multiplies each design row and matching observation by
`sqrt(weights[i])`, then calls JAX's least-squares routine. This is the standard
weighted reduction to ordinary least squares; QR, SVD, conditioning, and
full-rank least-squares foundations are treated by
{cite:t}`GolubVanLoan2013`.

Concrete weights must be one-dimensional, finite, nonnegative, and aligned with
the observations. A zero weight gives that observation exactly zero local
sensitivity. A negative or non-finite value is not silently converted into a
statistical policy.

## QR and SVD solve wrappers

`qr_solve(A, b)` solves square or tall full-rank systems through reduced QR. For
a tall matrix it returns the least-squares solution without forming normal
equations.

`svd_solve(A, b, rcond=None)` constructs a truncated pseudoinverse. Singular
values at or below `rcond * max(s)` are discarded. This makes the regularization
boundary explicit, but it also makes the solve only piecewise smooth: a singular
value crossing the cutoff changes the retained subspace.

Neither wrapper estimates whether its mathematical preconditions are
scientifically appropriate. `qr_solve` assumes full column rank. `svd_solve`
requires the caller to interpret its cutoff rather than treating a returned
array as proof that the inverse problem is identifiable.

## Covariance and correlation

`covariance_matrix(samples, weights=None, rowvar=False, ddof=1)` treats rows as
observations and columns as variables by default. The unweighted normalization is
`n_observations - ddof`; the weighted normalization is `sum(weights) - ddof`.
Concrete calls reject a nonpositive denominator instead of returning a plausible
zero matrix from an undefined estimate.

`correlation_from_covariance(covariance)` divides by the standard deviations on
both axes. Its public boundary rejects a non-square matrix, non-finite elements,
or a negative diagonal. A zero variance is guarded before division, so the
corresponding correlation row and column are finite zeros. This guard does not
certify that an arbitrary input is symmetric or positive semidefinite; callers
still own those covariance semantics.

These value-dependent eager checks cannot execute on unknown tracers. In all
three weighted/correlation surfaces, value-dependent eager validation is skipped while inputs are traced; compiled callers must supply finite weights, a positive
normalization denominator, and a valid covariance diagonal.

## Positive-definite jitter

`positive_definite_jitter(A, initial_jitter, growth, max_steps)` evaluates a
fixed geometric sequence of diagonal shifts and returns
`(A + jitter I, jitter, success)`. An already-positive-definite matrix returns
zero jitter. Otherwise, `success` identifies whether any tested shift made the
symmetrized matrix positive definite.

Diagonal perturbations are one broad strategy for obtaining a positive-definite
matrix. Modified Cholesky methods instead construct and control a matrix
perturbation during factorization; {cite:t}`ChengHigham1998` analyze that more
sophisticated problem. jaxstro's helper is intentionally narrower: a transparent
fixed candidate search, not a modified-Cholesky implementation and not a nearest
positive-definite projection.

## Differentiation and execution boundaries

Dense linear algebra does not have one universal derivative contract:

```{list-table} Linear-algebra differentiation contracts
:header-rows: 1
:label: tbl-linear-algebra-contracts

* - Operation
  - Contract
  - Supported claim
  - Boundary
* - Norm and projection at regular points
  - `smooth_pathwise`
  - AD agrees with central finite differences away from zero norm/denominator.
  - The active formula remains nonsingular.
* - Weighted least squares with fixed full-rank design
  - `smooth_pathwise`
  - AD through observations agrees with finite differences.
  - Rank and the declared weight pattern remain fixed.
* - Zero-weight observation
  - `known_zero`
  - The fitted coefficients are locally insensitive to that observation.
  - This records an explicit modeling exclusion, not robust inference.
* - QR/SVD solve inside a fixed full-rank/cutoff regime
  - `smooth_pathwise`
  - Right-hand-side sensitivities are meaningful while the retained subspace is
    unchanged.
  - The matrix remains away from rank and singular-value cutoff transitions.
* - Rank changes, SVD cutoff crossings, and condition numbers
  - `validation_only`
  - Values, residuals, retained ranks, and condition diagnostics can be checked.
  - No universal derivative is claimed at degeneracy or a cutoff crossing.
* - Zero variance and jitter selection
  - `validation_only`
  - Finite guarded correlations and the selected jitter/success state are tested.
  - Zero-variance guards and first-success selection are branch boundaries.
```

The validation suite checks smooth `norm2`, projection, weighted least squares,
SVD right-hand-side, covariance, and positive-variance correlation cases against
independent central finite differences. `condition_number(...)` remains a
diagnostic: exact singularity returns positive infinity, and coincident singular
values are not an inference gradient.

## What is deliberately not here

jaxstro does not own sparse solves, iterative Krylov methods, robust-regression
policy, nearest-correlation projection, matrix-free operators, or custom implicit
differentiation. Those deserve specialized libraries. This module provides small
dense primitives whose numerical boundaries can be stated and tested.

## From explanation to evidence

For signatures and ownership, see
[](../40-api/index.md#jaxstro-numerics-linear-algebra). The assertion-bearing
evidence map is in [](../60-validation/index.md). The five package-wide gradient
labels are defined in [](./index.md#gradient-contracts).
