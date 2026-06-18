---
title: Decision log (ADRs)
description: >-
  The numbered architecture-decision record for jaxstro — each entry the context,
  the choice, and the trade-offs behind a load-bearing design decision.
---

Decisions are load-bearing, so they get their own section adjacent to the
architecture prose rather than buried inside it (that choice is itself
[](./0005-diataxis-docs-with-adr-meta.md)). Each record states the **context** that
forced a choice, the **decision** taken, and the **rationale** with its trade-offs.
Read them when you want to know *why* the code is shaped the way it is — the theory
section tells you how the methods work; these tell you why the package made the
calls it did.

These are ported verbatim from the project's working ADR log; the published copies
here are the canonical record.

```{list-table} Architecture decision records
:header-rows: 1
:label: tbl-adr-index

* - ID
  - Title
  - Status
* - [0001](./0001-thin-foundation-posture.md)
  - Thin foundation posture
  - accepted
* - [0002](./0002-adopt-equinox-foundation.md)
  - Adopt equinox as core dependency
  - accepted
* - [0003](./0003-standalone-uv-hatchling-project.md)
  - Standalone uv project with hatchling
  - accepted
* - [0004](./0004-apache-2-0-license.md)
  - Apache-2.0 license
  - accepted
* - [0005](./0005-diataxis-docs-with-adr-meta.md)
  - Diataxis docs with ADR meta sections
  - accepted
* - [0006](./0006-build-own-quantity-not-unxt.md)
  - Build own Quantity, not unxt/quax
  - accepted
* - [0007](./0007-cgs-as-default-units.md)
  - CGS as default unit system
  - accepted
* - [0008](./0008-reject-ift-from-core.md)
  - Reject IFT from core
  - accepted
* - [0009](./0009-jaxstro-params-selective-inference.md)
  - jaxstro.params: selective inference (Equinox-only, not Zodiax)
  - accepted
* - [0010](./0010-ecosystem-config-architecture.md)
  - Ecosystem config architecture (layered; hydra at jaxstro-lab)
  - accepted
* - [0011](./0011-apache-2-0-standardization.md)
  - Apache-2.0 license standardization
  - accepted
```

The narrative "why" — the software's shape rather than its atomic decisions —
lives next door in [](../20-architecture/index.md).
