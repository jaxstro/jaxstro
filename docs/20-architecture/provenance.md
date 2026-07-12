---
title: Provenance architecture
description: >-
  How runtime manifests and source-backed cards answer different evidence
  questions without taking ownership of an entire scientific workflow.
---

Provenance is not one record. A scientific run needs evidence about what was
executed, while a public implementation claim needs evidence about the source,
convention, code, and tests that support it. jaxstro keeps those responsibilities
separate so that neither record can imply more than it proves.

```{list-table} Provenance ownership
:header-rows: 1
:label: tbl-provenance-ownership

* - Surface
  - Question answered
  - Inputs
  - Output
  - Validation
* - Runtime manifest — `jaxstro.provenance`
  - What method ran, with which inputs, parameters, artifacts, and environment?
  - Run-specific values and files selected by the caller.
  - Deterministic JSON or Markdown that travels with a result.
  - Hashing, schema, ordering, rendering, and missing-file behavior in
    `tests/unit/test_runtime_provenance.py`.
* - Source-backed card — `jaxstro.testing`
  - What source and convention support this bounded implementation claim?
  - An already-parsed mapping containing exact source locators, code references,
    pytest node IDs, status, and deviations.
  - A validated `ProvenanceCard` and deterministic generated MyST reference page.
  - Schema, code-symbol resolution, assertion-bearing test resolution, and
    generated-page freshness in
    `tests/validation/provenance_cards/test_registry.py`.
```

## Runtime manifests

`jaxstro.provenance` owns small records for a particular execution:

- `ArtifactHash` records a path, digest, hash algorithm, and file size;
  `hash_artifact` defaults to SHA-256 and raises `FileNotFoundError` for a missing
  path.
- `EnvironmentSnapshot` records Python, platform, and an explicit package set;
  `environment_snapshot` avoids turning a manifest into an accidental inventory
  of the developer machine.
- `MethodManifest` ties a method name and version to sorted inputs, parameters,
  artifacts, and an optional environment snapshot.

`manifest_to_json` and `manifest_to_markdown` provide stable ordering so that
changes remain reviewable. These helpers do not execute workflows, upload
artifacts, sign records, manage credentials, or decide which scientific inputs a
downstream package must record.

## Source-backed cards

`jaxstro.testing` owns the installed, dependency-light card schema.
`validate_card` accepts already-parsed mappings and returns an immutable
`ProvenanceCard`. The installed module does not parse YAML: repository tooling
chooses the file format, loads the registry, and calls the mapping-based API.

A card bounds one implementation claim through:

- a source reference with an exact locator and an explicit statement of what it
  supports;
- named conventions and deviations from the source;
- importable `path::qualname` code references;
- collectable, assertion-bearing pytest node IDs; and
- one of three evidence states.

The states are deliberately not interchangeable:

- `verified` requires source, code, and validation evidence;
- `needs-check` records a claim whose evidence is not yet sufficient for the
  verified state;
- `unverifiable-scanned` records that the available source cannot support a
  stronger machine-checkable claim.

Cards do not record the inputs or environment of a particular execution. They do
not turn an unavailable source into evidence, and they do not make an empty card
family complete.

## How the surfaces compose

A card can support the implementation convention used by a method; a runtime
manifest can record that the method ran on particular inputs and produced hashed
artifacts. Neither surface substitutes for the other. A manifest does not prove
the authority of a scientific convention, and a card does not prove that a
particular run used the claimed inputs or environment.

Downstream packages can therefore embed runtime manifests in their own result
records while linking the methods they used to source-backed cards. jaxstro owns
the small deterministic records and validators, not the downstream workflow,
storage, or reporting policy.

## Registry and honest gaps

The generated [](../40-api/provenance/index.md) routes to the current families:

- [](../40-api/provenance/constants.md) — constants and unit conventions;
- [](../40-api/provenance/transforms.md) — coordinate and astrometric transforms;
- [](../40-api/provenance/atmospheres.md) — the current atmosphere evidence gap.

Zero registered atmosphere cards do not mean complete atmosphere coverage. The
empty generated family is an explicit gap while atmosphere backends and their
source claims remain in progress. The [](../60-validation/index.md) records the
registry integrity and freshness gates alongside the runtime-manifest tests.
