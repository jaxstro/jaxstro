---
title: Computational evidence
---

# Computational evidence

## Owner import path

`jaxstro.evidence`

## Purpose

Portable computational-evidence schemas, deterministic serialization,
freshness checks, rendering, and cross-class indexing.

## Public records and callables

`ComparisonRecord`, `ComparisonRelation`, `EnvironmentRecord`,
`EvidenceArtifact`, `EvidenceFreshnessError`, `EvidenceClass`, `EvidenceIndex`,
`EvidenceIndexEntry`, `EvidenceStatus`, `MetricRecord`, `artifact_to_dict`,
`artifact_from_dict`, `artifact_to_json`, `artifact_to_markdown`,
`check_artifact`, `build_evidence_index`, `emit_artifact`, and
`validate_artifact`.

## Shape and dtype expectations

Records use deterministic scalar metadata and sequences. Every measured metric
retains producer-declared units; artifact serialization is JSON and Markdown.

## JAX transforms and AD classification

This is host-side evidence infrastructure, not a differentiable runtime kernel.

## Failure behavior

Invalid schemas and stale artifacts fail validation. Evidence records do not
set method-specific scientific thresholds or replace source provenance.

## Contract and evidence links

See [](../../60-validation/validation.md), [](../../60-validation/evidence-index.md),
and the generated [](./contracts.md) module contract.

## Canonical import example

```python
from jaxstro.evidence import validate_artifact
```
