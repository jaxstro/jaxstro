---
title: "ADR 0014 - Hoist provenance-card infrastructure into jaxstro"
description: >-
  Adopt a public machine-readable provenance registry and place dependency-light
  schema validation and deterministic rendering in jaxstro.testing.
id: 0014
date: 2026-07-11
status: accepted
supersedes: null
decided_by: user
last_read: 1
---

# 0014 - Hoist provenance-card infrastructure into jaxstro

## Context

Progenax ADR-0034 proved a machine-readable model-card registry that connects
scientific claims to exact source locators, code objects, validation tests, and
generated documentation. The pattern is useful beyond progenax models: jaxstro must
record the provenance and conventions of constants, units, coordinate transforms,
astrometry, and eventually other foundation-layer capabilities.

Copying the complete progenax implementation into every package would duplicate
schema and rendering behavior. Moving YAML parsing into jaxstro's installed runtime,
however, would violate the thin-foundation dependency boundary.

## Decision

Adopt a public machine-readable provenance-card registry in jaxstro and hoist its
package-independent machinery into `jaxstro.testing`.

1. `jaxstro.testing.provenance_cards` owns typed validation and deterministic MyST
   rendering for already-parsed mappings. It performs no filesystem access and does
   not import a YAML parser.
2. Repository tooling owns YAML loading, registry discovery, generated-file writes,
   and `--emit`/`--check` policy. PyYAML is a development dependency, not a core
   runtime dependency.
3. A verified card must contain a bounded scope, explicit conventions, at least one
   exact source locator, code references, validation references, status, and any
   deviations from its source.
4. Generated reference pages are committed and checked byte-for-byte against fresh
   rendering. Hand-authored theory and architecture pages remain separate.
5. jaxstro cards may claim only evidence established by the source-verified Slice-A
   audit. The atmosphere family remains explicitly uncarded until its scientific
   claims receive equivalent source verification.
6. Progenax migration to the shared API is a later consumer change; this slice does
   not modify the sibling repository.

## Rationale

- **One reusable contract:** schema and rendering changes are tested once in the
  foundation package rather than drifting across consumers.
- **Minimal runtime:** accepting mappings keeps serialization policy outside the
  installed module and preserves jaxstro's dependency posture.
- **Evidence-carrying docs:** exact locators, code paths, and tests make generated
  pages auditable and stale output detectable.
- **Honest incompleteness:** an empty family is preferable to a polished card whose
  source claims were not verified.
- **Prove then hoist:** progenax supplies the working precedent; jaxstro takes
  ownership only of the package-independent layer.

## Notes

This is jaxstro's local adoption of the ecosystem decision first recorded as
progenax ADR-0034. The generalized card uses domain-neutral identifiers rather than
requiring constants and transforms to masquerade as physical models.
