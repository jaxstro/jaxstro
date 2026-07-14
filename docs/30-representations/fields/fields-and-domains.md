---
title: Fields and domains
description: >-
  Conceptual contracts for values attached to scientific domains, including
  coordinates, components, units, and boundary conditions.
---

# Fields and domains

Use this page when an array is described as a physical field and you need to state
the domain, value space, units, components, and boundary conditions that give the
array scientific meaning.

:::{important} Deferred abstraction
`jaxstro.fields` does not exist. A common field runtime is deferred until at least
two real consumers demonstrate shared domain, topology, and operator contracts. No
implementation schedule is promised by this guide.
:::

## The scientific question

What quantity is attached to each point of which domain, and how is that continuous
or discrete object represented in a computation? A temperature field on a volume,
a velocity field on mesh faces, and a spectral intensity field on directions and
wavelengths may all be arrays, but they have different domains, components, units,
and boundary semantics.

A sampled array is not by itself a field contract. Shape does not say whether an
axis indexes coordinates, vector components, cells, faces, time, or independent
systems. Values cannot be interpreted or differentiated responsibly until those
roles and their conventions are explicit.

## Mathematical objects

Let $\Omega$ be a domain and $V$ a value space. A scalar field has one value per
domain point; a vector or tensor field has components defined relative to a basis or
frame. Domain points are geometric objects. Coordinates are labels assigned by a
chart and can change without changing the underlying point. Field values may carry
physical units distinct from coordinate units.

A computational representation also needs a sampling location: nodes, cell
centers, faces, or edges. It needs component ordering and basis conventions,
including whether vector components are global Cartesian components or local basis
components. It needs boundary conditions, such as periodic, Dirichlet, Neumann,
inflow, outflow, reflective, or domain-specific constraints. A boundary condition
is part of the operator problem, not padding metadata.

The domain may be continuous while its discretization is finite. Conversely, a
graph or catalog can be a fundamentally discrete domain. The contract must not
invent a continuous geometry when only adjacency is meaningful.

## Core derivation

The defining object is a map from domain points to values:

```{math}
:label: eq-field-map
\phi:\Omega\rightarrow V,
\qquad
p\mapsto\phi(p).
```

Equation [](#eq-field-map) separates three things that array notation often
collapses: the point $p\in\Omega$, its coordinate representation in a chosen chart,
and the value $\phi(p)\in V$. If coordinates change from $x$ to $x'$, a scalar
value at the same point is unchanged, while vector and tensor components transform
according to their basis convention.

Sampling at points $p_i$ produces values $\phi_i=\phi(p_i)$, but the finite list
$\{\phi_i\}$ does not reconstruct the field without the points, sampling locations,
interpolation or basis rule, domain boundary, and component convention. Even on a
regular grid, transposing axes while retaining shape can change the physical map.

Units provide a useful audit. If $p$ uses length coordinates and $\phi$ is density,
then $\phi$ has mass per volume regardless of array dtype. Integrating it requires a
measure or cell volumes with volume units; summing samples alone does not produce a
mass.

## Failure modes and interpretation limits

- Treating array indices as physical coordinates silently assumes origin, spacing,
  ordering, and chart.
- Omitting component basis information makes vector or tensor values ambiguous.
- Mixing node-centered and cell-centered values can introduce half-cell shifts and
  invalid operator stencils.
- Ignoring boundary conditions produces under-specified gradients, fluxes, and
  evolution problems.
- Interpolating across coordinate singularities or domain cuts can violate the
  intended geometry.
- Reusing one array for values with incompatible units makes reductions and losses
  scientifically meaningless.
- Assuming that a finite sample uniquely identifies a continuous field hides the
  reconstruction model and its approximation error.

## What Jaxstro may add

Current `jaxstro.spatial`, grid utilities, mesh utilities, geometry, and operator
utilities remain narrower owners of their existing contracts. They do not compose
into a general field abstraction merely because each works with arrays in space.
`jaxstro.fields` does not exist, and these pages do not propose one as imminent.

Only after two real consumers demonstrate the same domain, topology, value-location,
component, unit, boundary, and transform needs could Jaxstro consider a common
domain-agnostic representation. Such a representation might separate static domain
metadata from dynamic field values and make sampling location explicit. It must not
absorb solver policy, domain equations, or downstream physical interpretation.

## Evidence required before implementation

Two independent consumers must first supply concrete use cases and show that one
contract serves both without optional-field sprawl. Evidence must include scalar,
vector, and where justified tensor values; coordinate changes; unit checks;
node-, cell-, and face-located data; and multiple boundary types. Round-trip tests
must show that serialization preserves domain identity, coordinates, component
basis, sampling location, units, and boundary metadata.

JAX evidence must distinguish static topology and convention data from dynamic
value leaves under JIT, VMAP, and differentiation. Analytic fields should verify
sampling and coordinate transformation. Failure tests must reject axis, unit,
location, and boundary mismatches before an operator returns plausible numbers.

## Claim boundary

This page defines a field concept, not a package, base class, mesh standard, solver,
or schedule. It does not claim that existing arrays are fields or that all domains
share one useful runtime representation. The two-consumer rule is an evidence gate:
shared abstractions follow demonstrated common structure rather than anticipated
reuse.

## Connected representations, foundations, and methods

- Return to [](../representations.md) for current representation families and their
  runtime owners.
- Review [](../../10-foundations/mathematical-objects/functions-units-scales.md) for
  maps, domains, codomains, and units.
- Compare [](../geometry-coordinates/geometry.md) for current geometric conventions.
- Continue to [](topology-and-discretization.md) and
  [](../../20-methods/discrete-space/grids.md) to separate fields from the discrete
  structures used to store them.
