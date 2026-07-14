---
title: Provenance for reproducible research
description: >-
  How runtime manifests and source-backed cards answer different evidence
  questions without taking ownership of downstream scientific claims.
---

(workflow-provenance)=

Use this page when a result must preserve what ran while an implementation
claim must separately preserve which sources, conventions, code, and tests
support it.

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
* - Runtime manifest - `jaxstro.provenance`
  - What method ran, with which inputs, parameters, artifacts, and environment?
  - Run-specific values and files selected by the caller.
  - Deterministic JSON or Markdown that travels with a result.
  - Hashing, schema, ordering, rendering, and missing-file behavior in
    `tests/unit/test_runtime_provenance.py`.
* - Source-backed card - `jaxstro.testing`
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

Artifact hashes use an explicit algorithm; the default is SHA-256. A hash binds
bytes to a record but does not establish that the artifact is scientifically
adequate, authoritative, or safe. Environment snapshots record only the named
packages required by the caller rather than scraping an entire machine.

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

The generated [](../../40-api/provenance/index.md) routes to the current families:

- [](../../40-api/provenance/constants.md) - constants and unit conventions;
- [](../../40-api/provenance/transforms.md) - coordinate and astrometric transforms;
- [](../../40-api/provenance/atmospheres.md) - the current atmosphere evidence gap.

Zero registered atmosphere cards do not mean complete atmosphere coverage. The
empty generated family is an explicit gap while atmosphere backends and their
source claims remain in progress. The [](../../60-validation/index.md) records the
registry integrity and freshness gates alongside the runtime-manifest tests.

## Deterministic audit procedure

1. Define the method identity, version, sorted inputs, parameters, and artifact
   paths for the runtime manifest.
2. Hash every required artifact and fail if a required path is missing.
3. Record the explicit environment policy and package set.
4. Render JSON or Markdown twice and require byte-identical output.
5. Resolve source-card locators, import paths, and assertion-bearing tests.
6. Keep missing or `needs-check` evidence visible rather than upgrading it from
   a successful run.
7. Link the two provenance classes to the evidence artifact and bounded claim.

## Where the claim stops

Deterministic manifests and validated cards make lineage inspectable. They do
not prove that inputs are representative, a source is authoritative, a
numerical comparison is sufficient, or a downstream scientific claim is true.

## Connected ideas

See [](../../30-representations/parameters-state/serialization-and-provenance.md),
[](./random-state-ownership.md),
[](./evidence-and-claim-boundaries.md), and
[](../investigations/investigations.md).
