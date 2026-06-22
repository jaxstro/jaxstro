---
title: Runtime provenance
description: >-
  Deterministic artifact hashes, environment snapshots, and method manifests
  for evidence-carrying scientific workflows.
---

`jaxstro.provenance` provides runtime provenance records that are small enough to
use in tests, validation reports, and downstream workflow logs. The goal is not a
workflow database. The goal is deterministic evidence that can travel with a
method result.

## Records

`ArtifactHash` records path, digest, hash algorithm, and file size. The helper
`hash_artifact(path)` currently defaults to SHA-256 and raises
`FileNotFoundError` for missing paths.

`EnvironmentSnapshot` records Python, platform, and selected package versions.
`environment_snapshot(packages=(...))` keeps package selection explicit so
manifests do not accidentally become noisy inventories of a developer machine.

`MethodManifest` ties a method name and version to sorted inputs, sorted
parameters, artifact hashes, and an optional environment snapshot.

## Rendering

`manifest_to_json(...)` renders sorted, indented JSON. `manifest_to_markdown(...)`
renders a compact Markdown summary with deterministic section order.

The deterministic rendering contract matters because these artifacts are useful
only if diffs are meaningful.

## Boundary

The module does not manage workflow execution, remote storage, signatures,
credentials, or notebooks. Downstream packages can embed these manifests in
their own reports without making `jaxstro` own their workflow semantics.
