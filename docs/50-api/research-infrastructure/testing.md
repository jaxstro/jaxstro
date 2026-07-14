---
title: Validation and provenance tooling
---

# Validation and provenance tooling

## Owner import path

`jaxstro.testing`

## Purpose

Gradient audits, numerical comparisons, evidence reports, ratchets, and
source-backed provenance-card validation and rendering.

## Public records and callables

The owner exports gradient-contract records and audits; finite-difference and
directional-derivative comparisons; documentation ratchets; numerical evidence
reports; and `ProvenanceCard`, `validate_card`, `render_card`,
`render_family`, `render_index`, and `render_registry`.

## Shape and dtype expectations

Numerical audits accept scalar or array outputs with explicitly configured
directions and tolerances. Card and report rendering uses host-side records.

## JAX transforms and AD classification

Audit helpers execute JAX transforms and independent finite differences but are
test tooling, not runtime scientific acceptance policy.

## Failure behavior

Invalid cards, unresolved evidence anchors, stale ratchets, and failed numerical
comparisons raise or return explicit failed reports.

## Contract and evidence links

See [](../../60-validation/index.md), the generated [](./contracts.md), and the
[source-backed provenance cards](./source-provenance/source-provenance.md).

## Canonical import example

```python
from jaxstro.testing import compare_gradients
```
