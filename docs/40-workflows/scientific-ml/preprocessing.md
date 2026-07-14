# Preprocessing without data leakage

Use this page when a model needs standardized or whitened inputs and you need
the fitted transform to remain reproducible, unit-aware, and isolated from
validation and test data.

:::{important} Planned Jaxstro capability
`jaxstro.ml` does not exist and has no implementation schedule. This page
defines evidence and ownership requirements for a possible future capability;
it is not an API reference or implementation commitment.
:::

## The scientific question

How can a numerical model receive well-scaled inputs without allowing held-out
observations to influence the fitted transform? Preprocessing is part of the
scientific model because its fitted state changes every downstream prediction.

## Prerequisites

Review [](../../10-foundations/mathematical-objects/functions-units-scales.md)
and [](../../10-foundations/models-and-computation/what-is-a-model.md). The
conditioning consequences connect to
[](../../10-foundations/models-and-computation/sensitivity-conditioning-identifiability.md).

## Mathematical objects

Let the training matrix be $X_{\mathrm{train}}\in\mathbb{R}^{n\times p}$.
The fit stage produces means $\mu$, scales $s$, and, for whitening, a covariance
factor. The apply stage maps arrays using only those frozen values.

## Core derivation

For feature $j$, standardization is

```{math}
:label: eq-ml-standardization

z_{ij}=\frac{x_{ij}-\mu_j}{s_j},
\qquad
\mu_j=\frac{1}{n}\sum_{i\in I_{\mathrm{train}}}x_{ij}.
```

If the declared covariance convention is the sample covariance
$C=(n-1)^{-1}(X-\mu)^{\mathsf T}(X-\mu)$ and $C=LL^{\mathsf T}$ is a Cholesky
factorization, one whitening convention is

```{math}
:label: eq-ml-whitening

z_i=L^{-1}(x_i-\mu),
\qquad
\operatorname{Cov}(z)\approx I.
```

The factor orientation and covariance normalization are part of the artifact;
another factorization is valid only when its convention is declared.

## Assumptions and failure boundaries

A transform fitted on all data is data leakage. Fit on the training partition
only, then apply unchanged to validation, test, and production inputs. A future
contract must define zero-scale behavior rather than silently divide by zero;
missing values must be rejected, masked, or imputed by a declared policy. Dtype
and accumulation precision are explicit. Dimensionful columns must either be
converted to declared units before fitting or retain compatible unit metadata.
Inverse transforms must use the same fitted state and report when whitening is
rank-deficient or regularized.

## Worked conceptual example

Suppose temperature and luminosity are measured for 100 objects. Split indices
first. Compute $\mu$, $s$, and $L$ from the 70 training rows, serialize those
values with units and dtype, and apply them to all three partitions. Re-fitting
on the 15 validation rows would define a different model and invalidate a fair
validation comparison.

## Ownership boundary

Host-side code owns data inspection, partition selection, fitting, and artifact
serialization. JAX-side code owns a pure, fixed-shape apply function suitable
for `jit`, `vmap`, JVP, and VJP. Jaxstro could own domain-agnostic contracts for
this split; model construction remains with Equinox and domain semantics remain
with the downstream project.

## Proposed interface

Any future interface must distinguish immutable fitted state from the pure
apply operation. Names shown in design discussions are non-executable sketches,
not importable symbols.

## Evidence required before implementation

Evidence must include train-only fit tests, analytic forward and inverse checks,
reference comparisons, zero-scale and rank-deficiency failures, dtype and unit
round trips, JIT/VMAP/AD checks for apply, deterministic serialization, and a
leakage test that changes held-out values without changing fitted state.

## Where the claim stops

Standardized or whitened inputs do not prove that features are informative,
that the model is identifiable, or that a trained model generalizes.

## Connected ideas

Continue to [](./data-plans.md),
[](../../20-methods/linear-structure/linear-algebra.md), and
[](../reproducible-research/evidence-and-claim-boundaries.md).
