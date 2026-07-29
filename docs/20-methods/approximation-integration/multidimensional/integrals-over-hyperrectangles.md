---
title: Integrals over hyperrectangles
description: Geometry, orientation, units, and the common interface for multidimensional integration.
---

# Integrals over hyperrectangles

## Scientific question

How do we turn a scientific integral with several bounded coordinates into a
problem that numerical methods can share? Jaxstro's Phase B methods integrate
over a finite Cartesian product

```{math}
:label: eq-multidim-hyperrectangle
\Omega=\prod_{j=1}^{d}[a_j,b_j].
```

The coordinates may represent different physical ideas. A stellar-population
integral, for example, can combine mass, metallicity, age, and distance.

:::{note}
A hyperrectangle describes coordinate bounds, not a claim that the physical
geometry is Euclidean. Jacobians such as \(r\), \(r^2\sin\theta\), or a
selection density belong in the integrand or measure.
:::

## Geometric picture

Each axis contributes an interval. Their Cartesian product is a rectangle in
two dimensions, a box in three, and a hyperrectangle in higher dimensions.
Reversing one axis reverses the orientation; reversing two restores it.

## Derivation

Map the unit cube coordinate \(u_j\in[0,1]\) to each physical coordinate:

```{math}
:label: eq-multidim-affine-map
x_j=a_j+(b_j-a_j)u_j,
\qquad
d^d x=\prod_{j=1}^{d}(b_j-a_j)\,d^d u.
```

Therefore

```{math}
:label: eq-multidim-unit-cube
\int_{\Omega}f(\boldsymbol{x})\,d^d x
=
\left[\prod_{j=1}^{d}(b_j-a_j)\right]
\int_{[0,1]^d}
f\!\left(\boldsymbol{a}+(\boldsymbol{b}-\boldsymbol{a})
\odot\boldsymbol{u}\right)d^d u.
```

The signed product of widths preserves orientation.

## Computational cost

Domain mapping costs \(O(Nd)\) for \(N\) points. The integration method, not the
affine map, usually determines the dominant cost.

## What the estimator means

The domain supplies bounds and a signed volume factor. Error evidence comes
from the chosen formula: embedded rules, refinement differences, sparse
surpluses, or randomized replicates.

## JAX and differentiation

Finite bounds can be dynamic JAX values. With `gradient="replay"`, accepted
formula points are reconstructed and the first derivative includes smooth
bound motion. Method choice, dimension, and capacities remain static.

## Quantities and units

`Hyperrectangle.from_axes` accepts an `Axis` per coordinate. Each axis owns one
static unit, so heterogeneous coordinates do not need to be stacked into a
single artificial unit. If \(f\) has unit \(U_f\), the result has unit

```{math}
:label: eq-multidim-result-unit
U_I=U_f\prod_{j=1}^{d}U_{x_j}.
```

Quantity mode is alpha and opt-in; raw numeric domains remain supported.

## Worked astrophysical example

For a projected Plummer profile with scale \(a\), integrate radius and angle:

```{math}
:label: eq-multidim-plummer
\int_0^{2\pi}\int_0^R
\frac{r}{\pi a^2[1+(r/a)^2]^2}\,dr\,d\phi
=\frac{R^2}{R^2+a^2}.
```

Here the polar Jacobian \(r\) is explicit in the integrand.

## Failure modes

:::{warning}
Coincident bounds produce a zero-volume result. Nonfinite bounds are invalid.
A curved, triangular, simplex, or implicitly constrained region is not a
hyperrectangle and is outside the current geometry contract.
:::

## Audit recipe

Record axis order, bounds, units, orientation, Jacobians, method, tolerances,
capacities, status, work, and an independent truth or refinement check.

## Warranted claim

Jaxstro supports differentiable finite hyperrectangles with homogeneous raw
coordinates or heterogeneous opt-in quantity axes. Phase C geometries remain
future work.
