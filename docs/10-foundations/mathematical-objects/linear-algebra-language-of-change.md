---
title: Linear algebra as the language of change
description: Vectors, linear maps, geometry, and conditioning before matrix recipes.
---

# Linear algebra as the language of change

Use this page when vectors, matrices, or conditioning need to be interpreted as
scientific maps rather than calculation recipes.

Two source parameters can move two measured channels at the same time. Treat
vectors as perturbations, then ask which combinations the data see, which they
miss, and how numerical coordinates alter that diagnosis. A matrix is a coordinate
representation of the map after choosing a basis; the map and its visible
directions are the scientific objects.

## Vectors, coordinates, and linear maps

A vector can represent a position, but it can also represent a small change in
model parameters, a residual spectrum, a velocity, or a direction through
parameter space. A map $A$ is linear when it preserves addition and scaling:

```{math}
A(a\,u+b\,v)=a\,A(u)+b\,A(v).
```

A **basis** supplies coordinates for a vector. In a basis whose vectors are
the columns of $B$, the same abstract vector has coordinates $[v]_B$ through

```{math}
v = B[v]_B.
```

Changing basis changes coordinates and the matrix representing a map, not the
underlying vector or map. Units and scaling remain part of the scientific
meaning of those coordinates: a parameter measured in kelvin and one measured
in dex should not be treated as interchangeable numerical axes simply because
both appear in one array.

## Geometry: dot products, norms, and projection

The Euclidean dot product and norm are

```{math}
\langle u, v \rangle = u^{\mathsf{T}}v,
\qquad
\lVert v \rVert_2 = \sqrt{v^{\mathsf{T}}v}.
```

They define the angle through

```{math}
\cos\phi = \frac{\langle u,v\rangle}
{\lVert u\rVert_2\lVert v\rVert_2}.
```

If the columns of $Q$ are an orthonormal basis for a subspace
$\mathcal{S}$, its Euclidean projection is

```{math}
P_{\mathcal{S}} = QQ^{\mathsf{T}},
\qquad
P_{\mathcal{S}}v \in \mathcal{S}.
```

This geometry is a choice. When a residual $r$ has covariance $C$ that is
known, symmetric, and positive-definite, its covariance-weighted squared norm
is

```{math}
\lVert r\rVert_{C^{-1}}^2 = r^{\mathsf{T}}C^{-1}r.
```

It gives less influence to uncertain directions and accounts for correlated
errors. It is warranted only when $C$ represents the measurement model; it is
not a generic way to make a fit look better.

## Projection, residuals, and least squares

Suppose a linearized prediction is $X\beta$ for observations $y$. Weighted
least squares chooses a parameter change by

```{math}
\widehat\beta = \underset{\beta}{\arg\min}\;
\lVert X\beta-y\rVert_W^2,
\qquad
\lVert r\rVert_W^2 = r^{\mathsf{T}}Wr.
```

At an interior optimum with appropriate differentiability, the fitted residual
$\widehat r=X\widehat\beta-y$ is orthogonal to the model's visible directions:

```{math}
X^{\mathsf{T}}W\widehat r = 0.
```

This is the geometric content of the normal equations. Do not form the inverse
of $X^{\mathsf{T}}WX$ merely because it appears in an algebraic derivation:
QR or SVD reveals rank loss and is usually the more stable computational
representation. The method page explains the supported numerical routes.

## Directions a map reveals or hides

For a square map, an eigenpair satisfies

```{math}
Av=\lambda v.
```

Eigenvectors describe directions preserved up to scaling. For any rectangular
map, the singular-value decomposition is more broadly useful:

```{math}
A = U\Sigma V^{\mathsf{T}}.
```

The right singular vectors in $V$ are input directions; the diagonal entries
of $\Sigma$ say how strongly $A$ transmits them; $U$ gives the corresponding
output directions. A **null space** contains changes $v$ for which $Av=0$.
In inference, it can represent a parameter combination that leaves predictions
unchanged to first order.

The number of meaningfully nonzero singular values is the numerical rank. It
depends on an explicitly stated **rank tolerance**, data precision, and the
units/scaling of the representation, not a magic count returned by a library.

## Conditioning is a scientific question about scale

For a full-rank matrix in the Euclidean norm, the spectral condition number is

```{math}
\kappa_2(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}.
```

A large value warns that small perturbations in data or arithmetic can produce
large changes in a solution. Rescaling coordinates replaces $A$ by, for
example, $AD^{-1}$; it can improve arithmetic but it also changes the metric in
which parameter steps are described. Report the scaling and condition number
together. A well-conditioned representation does not remove a physical
degeneracy; it can only make the numerical consequences easier to see.

## Jacobians, covariance, and curvature

A Jacobian is the local linear map from parameter perturbations to prediction
perturbations:

```{math}
\delta y \approx J\,\delta\theta + \varepsilon.
```

Under a local linear model with a fixed covariance for the parameter changes
and errors, covariance propagation gives

```{math}
C_y \approx J C_\theta J^{\mathsf{T}} + C_\varepsilon.
```

This is a local statement. It does not establish that a nonlinear posterior is
Gaussian, that errors are independent, or that the parameters are globally
identifiable.

A quadratic form $v^{\mathsf{T}}Av$ assigns a direction-dependent magnitude.
For a scalar objective $f$, a Hessian $H$ gives the leading curvature term
near a reference point:

```{math}
f(\theta+\delta\theta)
\approx f(\theta) + \nabla f(\theta)^{\mathsf{T}}\delta\theta
+ \tfrac12\delta\theta^{\mathsf{T}}H\delta\theta.
```

Curvature can diagnose a locally weak direction, but it is not global
identifiability. Read [](../models-and-computation/sensitivity-conditioning-identifiability.md)
for the distinction.

## Read a small measurement problem geometrically

Imagine two measured features responding to two parameters. At a reference
model, the Jacobian $J$ maps a proposed parameter shift into a predicted change
in those features. Before computing a solve, ask: are its two columns nearly
parallel? If so, two physically distinct parameter changes create nearly the
same observable change. The smaller singular value will be small, and the
corresponding right singular vector names the locally confounded combination.

The useful response is not automatically to add a numerical regularizer. First
decide whether a new observable, a justified prior, a reparameterization, or a
more restricted scientific claim resolves the actual ambiguity.

## Try the running case

For the two-channel measurement, suppose two parameter changes have the local
map

```{math}
J = \begin{bmatrix} 1 & 1 \\ 1 & 1.01 \end{bmatrix}.
```

Before solving for a parameter update, predict which combination is weakly
visible: the common change in both parameters or their difference.

## Worked audit

The columns of $J$ are nearly parallel, so their difference is weakly visible:
one nearly null right-singular direction trades one parameter against the
other. Solving the square system can still produce a finite answer. The useful
audit is its singular values and the physical meaning of that weak direction,
not whether a generic inverse exists.

:::{figure} ../figures/linear-weak-direction.svg
:name: fig-linear-weak-direction
:alt: Two nearly parallel Jacobian columns map a weak difference direction in parameter space into a long uncertainty region, showing that the two measurements barely distinguish that parameter combination.

The long axis is not a solver failure. It identifies the combination that the
specified measurements leave weakly constrained.
:::

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
