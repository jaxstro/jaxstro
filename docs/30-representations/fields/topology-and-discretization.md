---
title: Topology and discretization
description: >-
  Separation of coordinates, connectivity, orientation, boundaries, and refinement
  in structured and unstructured scientific domains.
---

# Topology and discretization

Use this page when field values are sampled on a grid, mesh, or graph and you need
to distinguish geometric coordinates from connectivity and orientation.

:::{important} Deferred abstraction
`jaxstro.fields` does not exist. Shared topology and discretization records remain
deferred until at least two real consumers establish the same structural contract.
No implementation schedule is promised by this guide.
:::

## The scientific question

Which discrete entities are connected, how are they oriented, where is the
boundary, and how does that structure relate to coordinates? Coordinates do not
define topology. Two node sets can have identical coordinates but different edges
or cells; one connectivity can also be embedded in different coordinate systems.

This distinction controls valid neighborhoods, interpolation, flux orientation,
conservation, and refinement. Treating topology as an incidental array of indices
can make a computation depend on storage order instead of the represented domain.

## Mathematical objects

A cell complex or mesh contains nodes, cells, and faces, often with edges as an
additional entity type. Connectivity maps record which lower-dimensional entities
bound each higher-dimensional entity. Orientation assigns signs so neighboring
cells agree about shared-face flux direction. Boundary entities are those not
paired in the same way as interior entities, together with any periodic or identified
relations.

Coordinates attach geometric positions to nodes or other entities. Metrics add
lengths, areas, volumes, angles, or metric tensors. Topology can exist without an
embedding, while coordinates without connectivity do not define cells.

Structured and unstructured layouts encode the same ideas differently. A structured
layout may derive neighbors from multidimensional indices and fixed strides. An
unstructured layout stores explicit connectivity. Refinement can split entities,
change adjacency, and introduce parent-child relations; it is more than changing
coordinate resolution.

## Core derivation

Let $\mathbf{B}_k$ be the signed incidence matrix mapping oriented $k$-entities to
their oriented $(k-1)$-entity boundaries. The boundary of a boundary is empty, so

```{math}
:label: eq-topology-boundary-composition
\mathbf{B}_{k-1}\mathbf{B}_{k}=\mathbf{0}.
```

Equation [](#eq-topology-boundary-composition) is a topological identity independent
of coordinates or metric. For a two-dimensional cell, a column of $\mathbf{B}_2$
lists its oriented boundary edges; applying $\mathbf{B}_1$ then cancels edge
endpoints in pairs. Reordering or reversing an entity changes corresponding signs
but must preserve the zero composition.

Discrete differential operators often combine incidence with metric-dependent
maps. This separation is valuable: connectivity determines which values interact
and their orientation, while lengths, areas, and volumes determine physical scale.
It also identifies what JAX can differentiate. Coordinates and field values may be
dynamic leaves under a fixed connectivity, but changing the number or identity of
entities changes shapes and program structure.

## Failure modes and interpretation limits

- Inferring connectivity from nearest coordinates can connect distinct surfaces or
  miss domain-specific adjacency.
- Unrecorded orientation can make neighboring fluxes add instead of cancel.
- Confusing cell order with geometric orientation can flip signed areas or normals.
- Treating periodic boundaries as ordinary exterior boundaries changes topology.
- Refinement without parent-child and conservation rules can change integrated
  quantities.
- Degenerate or inverted cells can make metric factors singular even when incidence
  remains valid.
- Topology changes are structural and nondifferentiable even when field values are
  dynamic leaves. Gradients through a fixed chosen topology do not differentiate the
  discrete choice that created it.

## What Jaxstro may add

Current `jaxstro.spatial`, grid utilities, mesh utilities, geometry, and operator
utilities remain narrower owners. Their current candidate search, binning,
coordinate, and algebraic contracts should not be widened into a general field
runtime without evidence. `jaxstro.fields` does not exist and is not promised by
this documentation.

After two real consumers demonstrate shared needs, a common abstraction might store
entity counts, signed connectivity, boundaries, sampling locations, coordinates,
and metric data with clear static versus dynamic roles. It would need to support
structured and unstructured cases without forcing either into an unnatural layout.
Refinement policy, mesh generation, domain equations, and solver schedules would
remain outside a domain-agnostic representation.

## Evidence required before implementation

Evidence must include at least two consumer-derived topologies, not only toy grids.
Tests should cover oriented lines, surfaces, and volumes where relevant; boundary
identification; periodic connectivity; structured and unstructured parity; entity
reordering; and orientation reversal. The incidence identity in
[](#eq-topology-boundary-composition) must hold exactly for integer connectivity.

Geometry tests must separately verify lengths, areas, volumes, normals, and
degenerate-cell failures. JAX tests must show which arrays are static structure and
which are differentiable coordinates or field values. Serialization must preserve
entity identity and orientation. Refinement evidence must demonstrate conservation
and stable parent-child maps before it enters any shared contract.

## Claim boundary

This page does not select a mesh format, promise adaptive refinement, define a field
runtime, or make topology differentiable. The incidence identity is a structural
check, not validation of geometric quality or a physical discretization. A shared
Jaxstro owner remains deferred until the same abstraction succeeds for two real
consumers.

## Connected representations, foundations, and methods

- Start with [](fields-and-domains.md) for the distinction between domain points,
  coordinates, and values.
- Return to [](../representations.md) for existing representation owners.
- Review [](../../10-foundations/mathematical-objects/linear-algebra-language-of-change.md)
  for sparse linear maps and null spaces.
- Compare [](../../20-methods/discrete-space/grids.md),
  [](../../20-methods/discrete-space/meshes.md), and
  [](../../20-methods/discrete-space/spatial.md) for narrower current methods.
