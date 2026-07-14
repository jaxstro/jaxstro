---
title: Runtime provenance
---

# Runtime provenance

## Owner import path

`jaxstro.provenance`

## Purpose

Deterministic runtime artifact hashes, environment snapshots, method manifests,
and JSON/Markdown rendering.

## Public records and callables

`ArtifactHash`, `EnvironmentSnapshot`, `MethodManifest`, `hash_artifact`,
`environment_snapshot`, `manifest_to_json`, and `manifest_to_markdown`.

## Shape and dtype expectations

Records contain host metadata, strings, and metric mappings with explicit
units. Artifact hashing reads declared files or byte content.

## JAX transforms and AD classification

Runtime provenance is host-side tooling and has no AD claim.

## Failure behavior

Missing artifacts and invalid records fail explicitly. A runtime manifest
records a computation; it does not replace scientific-source validation.

## Contract and evidence links

See [](../../40-workflows/reproducible-research/provenance.md),
[](../../30-representations/parameters-state/serialization-and-provenance.md),
and the [source-card index](./source-provenance/source-provenance.md).

## Canonical import example

```python
from jaxstro.provenance import MethodManifest
```
