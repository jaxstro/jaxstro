---
title: Linear algebra helpers
description: >-
  Covariance, least squares, conditioning, and positive-definite repair with
  explicit numerical and differentiation contracts.
---

## The question this method answers

How can a researcher summarize coupled variation, fit a linear model, or solve
a dense system without hiding rank and conditioning assumptions? Linear algebra
turns vectors of measurements into geometric questions about directions,
projections, and sensitivity.

:::{important}
A returned coefficient vector is evidence that an algorithm ran. It is not by
itself evidence that the design identifies the coefficients or that the fitted
model is scientifically appropriate.
:::

## Before computation: what should be true?

State the shapes and units first. For $n$ observations and $p$ features, a
design matrix has shape $(n,p)$ and a response has leading shape $(n,)$. A
weighted fit needs finite nonnegative weights of shape $(n,)$. Covariance needs
`n_obs - ddof > 0` for unweighted covariance and
`sum(weights) - ddof > 0` for weighted covariance. The weighted denominator is
the current frequency-weight-style runtime semantics, not an
effective-sample-size correction. A solve also needs a rank assumption or an
explicit singular-value cutoff.

:::{warning}
`weighted_lstsq` and covariance helpers reject invalid concrete weights, but
value-dependent eager validation is skipped while inputs are traced. A compiled
caller still owns finiteness, nonnegativity, and a positive covariance
normalization denominator.
:::

## Define the mathematical objects

A vector is an ordered collection of scalars. Its Euclidean norm is
$\lVert x\rVert_2=(\sum_j x_j^2)^{1/2}$. A matrix is a linear map between
vector spaces. Its singular values measure how strongly it stretches different
directions; the 2-norm condition number is
$\kappa_2(A)=\sigma_{\max}/\sigma_{\min}$. Large conditioning means small input
changes can produce large solution changes.

Covariance measures joint centered variation. If row $i$ is observation $x_i$
and $\bar{x}$ is the sample mean, the unweighted estimator is

```{math}
C=\frac{1}{n-\mathrm{ddof}}\sum_{i=1}^{n}(x_i-\bar{x})(x_i-\bar{x})^\mathsf{T}.
```

The diagonal contains variances. Correlation divides $C_{jk}$ by the two
standard deviations, making it dimensionless. A zero variance means that
normalization is undefined; Jaxstro returns finite zeros for that row and
column rather than inventing a correlation.

## Derive the method

The covariance estimator follows directly from centered outer products:

```{math}
:label: eq-covariance-estimator
C=\frac{1}{n-\mathrm{ddof}}\sum_{i=1}^{n}(x_i-\bar{x})(x_i-\bar{x})^\mathsf{T}.
```

Weighted least squares chooses coefficients $\beta$ to minimize squared
residuals $r=X\beta-y$:

```{math}
:label: eq-weighted-least-squares
J(\beta)=r^\mathsf{T}Wr,
\qquad
\nabla_\beta J=2X^\mathsf{T}W(X\beta-y)=0,
\qquad
(X^\mathsf{T}WX)\widehat{\beta}=X^\mathsf{T}Wy.
```

The implementation does not form the normal equations. It multiplies each row
of $X$ and $y$ by $\sqrt{w_i}$ and delegates to JAX least squares, avoiding the
extra conditioning penalty of explicitly forming $X^\mathsf{T}WX$. The normal
equation is still a useful audit: $X^\mathsf{T}W\widehat{r}$ should be near zero.

QR factors a tall matrix as $A=QR$ with orthonormal columns in $Q$, then solves
$R\beta=Q^\mathsf{T}y$. SVD writes $A=U\Sigma V^\mathsf{T}$ and inverts only
singular values above the chosen cutoff.

## What the algorithm actually does

`covariance_matrix(samples, weights=None, rowvar=False, ddof=1)` treats rows as
observations by default. `rowvar` and `ddof` are static in its compiled core.
Under the current weighted rule, one observation with weight 2 and `ddof=1`
passes because its denominator is one and returns a zero covariance matrix. This
illustrates the normalization convention; it is not evidence for two independent
observations.
`weighted_lstsq` accepts scalar or array-valued responses with the same leading
sample axis. `qr_solve` requires rows greater than or equal to columns;
`svd_solve` zeroes inverse singular values at or below `rcond * max(s)`.

`positive_definite_jitter` scans a fixed geometric sequence of diagonal shifts
and returns `(shifted, jitter, success)`. It returns the first successful tested
shift, not the smallest possible perturbation or a nearest positive-definite
matrix {cite:t}`ChengHigham1998`.

## What JAX differentiates

JAX differentiates the executed dense algebra. Smooth derivatives are useful
while rank, SVD cutoff membership, and the declared zero-weight pattern stay
fixed. A zero weight gives that observation known-zero local sensitivity.
Rank changes, cutoff crossings, zero-norm branches, zero variance, first-success
jitter selection, and coincident singular values are nonsmooth boundaries.
`condition_number` is a diagnostic, not an inference objective; exact
singularity returns positive infinity.

```{list-table} Linear-algebra differentiation contracts
:header-rows: 1
:label: tbl-linear-algebra-contracts

* - Operation
  - Supported derivative claim
  - Boundary
* - Norm and projection at regular points
  - `smooth_pathwise`
  - The active norm or denominator stays nonzero.
* - Weighted least squares with fixed full-rank design
  - `smooth_pathwise`
  - Rank and the weight pattern stay fixed.
* - Zero-weight observation
  - `known_zero`
  - This is declared exclusion, not robust inference.
* - QR/SVD solve inside a fixed full-rank/cutoff regime
  - `smooth_pathwise`
  - No rank or retained-subspace transition occurs.
* - Rank changes, SVD cutoff crossings, and condition numbers
  - `validation_only`
  - Residuals, ranks, and condition diagnostics are audited instead.
* - Zero variance and jitter selection
  - `validation_only`
  - Guards and first-success selection are branch boundaries.
```

## Using it in Jaxstro

```python
from jaxstro.jaxconfig import enable_high_precision

enable_high_precision()

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
    [[1.0, 2.0, 5.0], [2.0, 4.0, 5.0], [3.0, 6.0, 5.0], [4.0, 8.0, 5.0]]
)
covariance = covariance_matrix(samples)
correlation = correlation_from_covariance(covariance)

matrix = jnp.diag(jnp.array([-0.03, 2.0]))
shifted, jitter, success = positive_definite_jitter(
    matrix, initial_jitter=1.0e-3, growth=10.0, max_steps=4
)

assert jnp.allclose(unweighted_coeffs, jnp.array([-1.6, 5.9]))
assert jnp.allclose(weighted_coeffs, jnp.array([1.0, 2.0]))
assert jnp.allclose(qr_coeffs, svd_coeffs)
assert jnp.all(jnp.isfinite(correlation))
assert jnp.allclose(correlation[2], 0.0)
assert success and jnp.isclose(jitter, 0.1)
assert jnp.linalg.eigvalsh(shifted).min() > 0.0
```

## How to audit the result

1. Check shapes, units, weight domains, and the effective covariance denominator.
2. Compare $X^\mathsf{T}W\widehat{r}$ with zero and report residuals by observation.
3. Compare QR and SVD in a well-conditioned full-rank fixture {cite:t}`GolubVanLoan2013`.
4. Perturb $y$ independently and compare central differences with AD while rank is fixed.
5. Report singular values, the SVD cutoff, selected jitter, and `success`.

:::{tip}
Scale columns to comparable numerical ranges before interpreting a condition
number. Record the scaling so the audit is reproducible.
:::

:::{figure} ../../10-theory/figures/linear-algebra-contracts.webp
:name: fig-linear-algebra-contracts
:alt: Four regression observations with an outlier and measured weighted and unweighted fits, beside matrix eigenvalues before and after selected diagonal jitter

The public APIs produce both panels. This fixed fixture demonstrates declared
weighting and jitter policy; it is not a robust-regression benchmark.
:::

[](#fig-linear-algebra-contracts) ties the reported fit and jitter diagnostics
to the concrete fixture used in the audit.

## Where the claim stops

These helpers do not establish model adequacy, identifiability, robust outlier
policy, or uncertainty calibration. Jaxstro does not own sparse or iterative
linear solves here; the delegated Lineax guide remains separate. A finite
correlation matrix is not proof that an arbitrary input was symmetric or
positive semidefinite.

## Connected ideas

:::{seealso}
Build the geometric language in
[](../../10-foundations/mathematical-objects/linear-algebra-language-of-change.md),
represent array state with
[](../../30-representations/parameters-state/pytrees-as-scientific-state.md), and
audit sensitivities through
[](../../40-workflows/differentiable-research/auditing-derivatives.md).
Signatures live in [](../../50-api/linear-structure/linear-algebra.md), while
assertion-bearing evidence belongs in [](../../60-validation/validation.md).
The shared gradient labels are in [](../methods.md#gradient-contracts).
:::
