---
title: "ADR 0013 - Distribute as siblings; defer namespace consolidation"
description: >-
  Keep flat, independently versioned ecosystem packages for v0.x and settle the
  permanent package-name question before the first external PyPI release.
id: 0013
date: 2026-06-17
status: accepted
supersedes: null
decided_by: user
last_read: 3
---

# 0013 - Distribute as independent sibling packages for v0.x; defer namespace consolidation

## Context

Phase C release hardening left one distribution decision open: should the ecosystem
remain a set of flat sibling packages, or should it become a shared `jaxstro.*`
namespace with the core distribution renamed to `jaxstro-core`?

- **Siblings:** each package has its own distribution and top-level import -
  `jaxstro`, `gravax`, `progenax`, `fluxax` - with independent versions.
- **Namespace:** packages share `jaxstro.*` imports such as `jaxstro.core` and
  `jaxstro.gravax`, requiring an atomic restructure of the existing regular
  `jaxstro` package.

The mechanical migration cost is mostly flat while all consumers are internal. The
coordination and compatibility cost rises sharply once external users pin the
`jaxstro` distribution or import path.

## Decision

**Distribute as independent sibling packages with flat imports for the v0.x line.
Defer PEP 420 namespace consolidation.**

1. Keep the current flat distributions and independent release cadence established
   by ADR-0012.
2. Revisit the namespace choice before the first PyPI release expected to acquire
   external dependants-not at an undefined future maturity milestone.
3. Use **`jaxstro`** as the permanent core distribution and import name. At
   release staging on 2026-07-12, Anna selected `jaxstro` and rejected a rename
   to `jaxstro-core`. A future umbrella must use a different name rather than
   repurposing the established core identity.

## Rationale

- The current sibling layout already works and requires no migration churn.
- Independent packages match jaxstro's thin-foundation posture and independent
  release cadence.
- External adoption, rather than code maturity, is the event that makes a namespace
  migration expensive.
- Holding the upload is cheaper than publishing a name that later requires a
  breaking distribution and import transition.

## Notes

This decision defers namespace consolidation with an explicit revisit trigger and
resolves the distribution name as `jaxstro`. It does not authorize a PyPI upload;
that remains a separate explicit release action.
