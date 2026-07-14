---
title: Serialization and runtime provenance
description: >-
  Deterministic artifact hashes, environment snapshots, and method manifests for an
  executed scientific calculation.
---

Use this page when a result must record which method, parameters, artifacts, and
software environment produced it, without overstating that record as scientific
validation.

:::{important} Implemented Jaxstro capability
`jaxstro.provenance` implements deterministic runtime artifact hashes, environment
snapshots, method manifests, and JSON or Markdown rendering.
:::

## Representation contract

| Contract field | Current representation |
| --- | --- |
| Mathematical object | Frozen records for content hashes, selected environment fields, and one method execution with a name, version, parameters, inputs, artifacts, and optional environment. |
| Physical convention | Artifact bytes are hashed with an explicit algorithm, JSON keys are sorted, and units remain explicit in caller-owned parameter or input payloads. |
| Runtime owner | `jaxstro.provenance` owns `ArtifactHash`, `EnvironmentSnapshot`, `MethodManifest`, hashing, snapshots, and deterministic renderers. |
| Shape and unit policy | Manifest payloads are scalar and mapping records rather than JAX arrays; scientific arrays stay in referenced artifacts, and units are not inferred. |
| Transform boundary | Provenance collection and serialization are host-side tooling; they are intentionally outside `jit`, `vmap`, and `grad`. |
| Evidence | Runtime-provenance tests check SHA-256 content sensitivity, missing files, environment capture, ordering, and stable JSON and Markdown rendering. |
| Downstream interpretation boundary | A manifest does not establish source authority, scientific acceptance, model adequacy, or reproducibility when unrecorded external state exists. |

## Content identity

For artifact bytes $b$, the default identifier is

```{math}
:label: eq-provenance-content-hash

h = \operatorname{SHA256}(b).
```

`hash_artifact` records the path, algorithm, digest, and byte size. A matching digest
supports byte identity; it does not say whether the file is scientifically correct.

`environment_snapshot` records Python, platform, and explicitly requested package
versions. `MethodManifest` has exactly these dataclass fields in runtime order:
`name`, `version`, `parameters`, `inputs`, `artifacts`, and `environment`. The
environment is optional; richer measurements or narrative interpretation belong in
separate evidence and report records rather than first-class manifest sections.

## Deterministic rendering

`manifest_to_json` uses sorted keys and stable indentation. The Markdown renderer
uses deterministic ordering so committed reports can be freshness-checked and
reviewed as ordinary diffs. Callers remain responsible for representing array
storage, configuration schemas, units, and secret or personally identifying data.

:::{warning} Runtime provenance and source evidence are different records
`jaxstro.provenance` answers what a run consumed. Source-backed implementation cards
in `jaxstro.testing` answer which authority and executable tests support a bounded
package claim. Neither record substitutes for the other.
:::

The tests verify the deterministic representation in
[](#eq-provenance-content-hash) and renderer behavior. They do not prove that a
manifest captured every scientifically relevant input, that an artifact can be
recreated on another machine, or that the resulting scientific claim is warranted.
