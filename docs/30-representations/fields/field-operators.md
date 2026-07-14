---
title: Field operators
description: >-
  Discrete gradient and divergence contracts with orientation, metric, boundary,
  units, adjoint, and conservation evidence made explicit.
---

# Field operators

Use this page when a discrete gradient, divergence, flux, or related operator is
applied to field values and you need to audit its orientation, spacing, units,
boundary treatment, and conservation meaning.

:::{important} Deferred abstraction
`jaxstro.fields` does not exist. Common field operators remain deferred until at
least two real consumers demonstrate shared domain, topology, and operator
contracts. No implementation schedule is promised by this guide.
:::

## The scientific question

How should a continuum differential relation be represented on finite entities, and
which identities survive the discretization? An operator is not defined by a stencil
alone. Its input and output locations, orientation, spacing or metric, boundary
conditions, component basis, and units determine its scientific meaning.

For example, a discrete gradient may map node scalars to oriented edge differences,
while a finite-volume divergence may map oriented face fluxes to cell-centered rates.
Calling both arrays "derivatives" hides their distinct domains and conservation
contracts.

## Mathematical objects

Let scalar samples $\phi_i$ live at ordered one-dimensional nodes $x_i$. A forward
difference gradient lives on the interval between nodes $i$ and $i+1$. Its
orientation points from $i$ to $i+1$, and its units are $[\phi]/[x]$.

For a finite-volume cell $c$ with volume $V_c$, let face $f$ have outward signed
orientation $s_{cf}\in\{-1,+1\}$ relative to the cell, area $A_f$, and normal flux
$q_f$. A discrete divergence lives at the cell and has units $[q] [A]/[V]$.
Boundary faces require a supplied flux or a rule derived from the boundary
conditions. Interior faces must appear with opposite signs in adjacent cells.

Matrices can represent these maps, but the matrix alone does not say whether an
inner product includes mass, area, or volume weights. Therefore an algebraic
transpose is not automatically the physically meaningful adjoint.

## Core derivation

A one-dimensional edge gradient and a finite-volume cell divergence can be written
as

```{math}
:label: eq-discrete-field-operators
(G\phi)_{i+1/2}=\frac{\phi_{i+1}-\phi_i}{x_{i+1}-x_i},
\qquad
(Dq)_c=\frac{1}{V_c}\sum_{f\in\partial c}s_{cf}A_f q_f.
```

In [](#eq-discrete-field-operators), reversing an edge orientation reverses both the
numerator ordering and the represented component. For divergence, summing
$V_c(Dq)_c$ over cells cancels interior face contributions because the same face has
opposite orientation in adjacent cells. What remains is boundary flux:

```{math}
:label: eq-discrete-divergence-balance
\sum_c V_c(Dq)_c=\sum_{f\in\partial\Omega}s_f A_f q_f.
```

Equation [](#eq-discrete-divergence-balance) is the discrete conservation statement
to test. It depends on consistent incidence, geometry, and boundary accounting; it
does not follow merely because a stencil resembles a continuum derivative.

An algebraic adjoint $G^{\mathsf{T}}$ is defined by an unweighted Euclidean inner
product. A discrete adjoint under mass matrices $\mathbf{M}_0$ and $\mathbf{M}_1$
instead satisfies
$\langle G\phi,q\rangle_{\mathbf{M}_1}=\langle\phi,G^{*}q\rangle_{\mathbf{M}_0}$,
including boundary terms. Continuum identities such as integration by parts require
discrete summation-by-parts evidence under the declared inner products.

## Failure modes and interpretation limits

- Uniform-spacing stencils applied to nonuniform coordinates return incorrect units
  and scale even when shapes match.
- Inconsistent face orientation destroys interior-flux cancellation.
- Ghost cells or padding without a named boundary condition hide external data.
- Applying a node operator to cell-centered values introduces an unstated
  interpolation.
- Coordinate-basis derivatives can omit metric or connection terms on curvilinear
  domains.
- An algebraic adjoint can differ from the continuum adjoint because discrete inner
  products and boundary terms differ.
- Conservation of a discrete sum does not prove accuracy, convergence, stability,
  or conservation of every physical invariant.
- Differentiating field values through a fixed operator does not differentiate a
  limiter switch, topology choice, or adaptive remeshing decision.

## What Jaxstro may add

Current `jaxstro.spatial`, grid utilities, mesh utilities, geometry, and operator
utilities remain narrower owners. Existing numerical operator utilities own generic
algebraic mechanics; they do not claim field location, metric, boundary, or
conservation semantics. `jaxstro.fields` does not exist, and no common operator
package is imminent.

If two real consumers establish a shared contract, Jaxstro may later represent
input and output entity locations, signed incidence, metric weights, boundary
policies, unit transforms, and operator evidence. The abstraction would wrap
scientific semantics around reusable linear maps without taking ownership of domain
equations, Riemann solvers, time integrators, or application-specific closures.

## Evidence required before implementation

Analytic tests must recover constant and linear-field derivatives on uniform and
nonuniform domains with the expected convergence order. Unit tests must verify that
gradient and divergence outputs carry the derived units. Orientation reversal and
entity reordering must preserve the represented result. Boundary tests must cover
periodic, prescribed-value, prescribed-flux, and unsupported policies explicitly.

Conservation evidence must verify [](#eq-discrete-divergence-balance) on structured
and unstructured examples, including boundary flux. Adjoint claims require
inner-product tests with the actual mass or metric matrices. Summation-by-parts
claims require the complete discrete identity, including boundary terms. JAX tests
must distinguish derivatives with respect to dynamic field values and coordinates
from nondifferentiable topology, limiter, and boundary-policy choices.

## Claim boundary

These examples define audit questions, not a universal operator API. A passing
conservation identity does not establish accuracy or stability, and an algebraic
transpose does not establish a continuum adjoint. This documentation promises no
module or schedule. Shared ownership remains deferred until two consumers support
the same field-location, topology, metric, boundary, unit, and evidence contract.

## Connected representations, foundations, and methods

- Read [](fields-and-domains.md) and [](topology-and-discretization.md) for the
  objects on which these operators act.
- Return to [](../representations.md) for current ownership boundaries.
- Review [](../../10-foundations/mathematical-objects/functions-units-scales.md) for
  dimensional checks and [](../../10-foundations/mathematical-objects/linear-algebra-language-of-change.md)
  for adjoints and inner products.
- Compare [](../../20-methods/linear-structure/operators.md),
  [](../../20-methods/discrete-space/grids.md), and
  [](../../20-methods/discrete-space/meshes.md) for current narrower methods.
