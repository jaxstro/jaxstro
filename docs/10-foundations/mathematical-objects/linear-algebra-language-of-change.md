---
title: Linear algebra as the language of change
description: Vectors, linear maps, geometry, and conditioning before matrix recipes.
---

# Linear algebra as the language of change

Use this page when vectors, matrices, or conditioning need to be interpreted as
scientific maps rather than calculation recipes.

Linear algebra is the language we use to describe many simultaneous changes.
Begin with **vectors as perturbations** and a **linear map** that transforms
them. A matrix is the coordinate representation of that map after choosing a
basis; it is not the underlying idea.

## Vectors and linear maps

A vector can represent a position, but it can also represent a small change in
stellar parameters, a residual spectrum, a velocity, or a direction through
parameter space. A map $A$ is linear when it preserves addition and scaling:

```{math}
A(a\,u+b\,v)=a\,A(u)+b\,A(v).
```

A **basis** supplies coordinate vectors. Changing basis changes the matrix and
coordinates, not the abstract vector or map. Units and scaling remain part of
the scientific interpretation of those coordinates.

## Geometry: dot products, norms, and projection

A dot product defines angles and lengths; a norm measures magnitude; a
projection keeps the component lying in a chosen subspace. These choices are
not always innocent. Standard Euclidean distance treats every coordinate as
commensurate, while covariance-weighted geometry accounts for different scales
and correlated measurement uncertainty.

## Directions a map reveals or hides

Eigenvectors are directions preserved up to scaling by a square map. Singular
vectors describe input and output directions of any rectangular map. A **null
space** contains changes the map cannot see. In inference, a null direction can
represent a parameter combination that leaves predictions unchanged locally.

The ratio between strongly and weakly transmitted directions contributes to a
**condition number**. A large condition number warns that small perturbations in
data or arithmetic may cause large changes in an inferred solution. It is a
property of the represented problem and scaling, not merely of the solver.

## Jacobians, covariance, and curvature

A Jacobian is the linear map carrying small input perturbations to small output
perturbations. Covariance describes the second-moment geometry of variation.
A quadratic form $v^T A v$ assigns a direction-dependent magnitude. A
**Hessian** describes local second-order curvature of a scalar function. These
objects connect linear algebra to derivatives, uncertainty, optimization, and
identifiability.

## Predict

Before solving, predict the map's shape, units, rank, visible and null
directions, and which scaling choices may control its condition number.

## Compute

Use a representation appropriate to the question: matrix-vector products when
possible, QR for stable least squares, SVD when rank and null directions matter,
and covariance-aware weights when the measurement model warrants them.

## Audit

Check residuals, reconstruction identities, rank assumptions, singular values,
and sensitivity to rescaling. Compare a structured computation with a small
explicit dense case.

## State the warranted claim

"This solve is stable for the tested matrix and scaling" does not imply that the
physical parameters are globally identifiable. State the represented map,
domain, rank tolerance, and observed conditioning.

## Misconception check

> A vector need not be an arrow in physical space. A matrix is not automatically
> a model, and an invertible matrix in floating-point arithmetic is not
> automatically a well-conditioned scientific inverse.

Continue to [](./what-is-a-derivative.md). For Jaxstro's numerical helpers, see
[](../../20-methods/linear-structure/linear-algebra.md).
