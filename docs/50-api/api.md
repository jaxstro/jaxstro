---
title: API reference
description: Owner-qualified reference for the current importable Jaxstro surface.
---

# API reference

Use this reference when you know the operation you need and want its current
Python owner, records, transform behavior, failure contract, and evidence. The
groups below are documentation routes; they do not create Python namespaces.

::::{grid} 1 1 3 3

:::{card} Which method owns the operation?
:link: ./change-constraints/rootfinding.md

Find change, approximation, linear, random, and discrete-space owners by the
kind of numerical operation they perform.
:::

:::{card} Which representation owns the data?
:link: ./physical-representations/units.md

Find physical representations and scientific-data owners without hiding
units, coordinates, sampling semantics, or provenance.
:::

:::{card} Which infrastructure records trust?
:link: ./research-infrastructure/contracts.md

Find checks, configuration, scientific contracts, evidence, provenance, and
testing owners.
:::

::::

:::{note}
API pages answer what is importable now. Concept pages and ecosystem guides do
not become runtime promises merely because they are linked from the same site.
Release-qualified platform support and explicit non-claims are stated in the
[](../70-project/release/support.md).
:::

| Surface | Status | Documentation rule |
| --- | --- | --- |
| Owner-qualified module APIs | Current | Canonical reference path |
| Flat `jaxstro.numerics` callable re-exports | Current legacy inventory | Remain importable until the Project 2 hard cut |
| Proposed or delegated capabilities | Planned or ecosystem-owned | Remain outside API reference until an owner and contracts exist |

## Canonical import policy

Import a coherent module and qualify the call:

```python
from jaxstro.numerics import rootfinding

result = rootfinding.safeguarded_bracketed_root(...)
```

Or import a symbol directly from the module that owns it:

```python
from jaxstro.numerics.rootfinding import safeguarded_bracketed_root
```

Flat callable re-exports from `jaxstro.numerics` are legacy inventory awaiting Project 2.
They remain importable for now, but they are not the canonical documentation path.
This reference does not change runtime exports.

## Method owners

- **Change and constraints:** [](./change-constraints/autodiff.md),
  [](./change-constraints/rootfinding.md), [](./change-constraints/kepler.md),
  [](./change-constraints/optimization.md), and [](./change-constraints/ode.md).
- **Approximation and integration:**
  [](./approximation-integration/interpolation.md),
  [](./approximation-integration/regular-grid.md),
  [](./approximation-integration/splines.md), and
  [`jaxstro.quad`](./approximation-integration/quad.md). The
  [sampled-integration](./approximation-integration/integration.md) and
  [fixed-quadrature](./approximation-integration/quadrature.md) pages document
  temporary compatibility paths, not current owners.
- **Linear structure:** [](./linear-structure/linear-algebra.md),
  [](./linear-structure/compensated.md), [](./linear-structure/operators.md), and
  [](./linear-structure/special.md).
- **Randomness:** [](./randomness/distributions.md), [](./randomness/rng.md),
  [](./randomness/random.md), [](./randomness/sampling.md), and
  [](./randomness/stats.md).
- **Discrete space:** [](./discrete-space/grids.md),
  [](./discrete-space/meshes.md), and [](./discrete-space/spatial.md).

## Representation and data owners

- **Physical representations:** [](./physical-representations/constants-api.md),
  [](./physical-representations/units.md),
  [](./physical-representations/quantity.md),
  [](./physical-representations/coords.md),
  [](./physical-representations/geometry.md),
  [](./physical-representations/astrometry.md), and
  [](./physical-representations/params.md).
- **Scientific data:** [](./scientific-data/spectra.md) and
  [](./scientific-data/atmospheres-api.md).

## Research infrastructure owners

See [](./research-infrastructure/checks.md),
[](./research-infrastructure/types.md),
[](./research-infrastructure/jaxconfig.md),
[](./research-infrastructure/contracts.md),
[](./research-infrastructure/evidence.md),
[](./research-infrastructure/provenance.md), and
[](./research-infrastructure/testing.md).

Only current importable surfaces appear here. Proposed capabilities remain in
the Methods, Representations, Workflows, and Project roadmaps until runtime
owners and executable contracts exist.
