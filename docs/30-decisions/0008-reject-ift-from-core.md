---
title: "ADR 0008 — Reject IFT from core"
description: "Reject Information Field Theory from core; adopt NIFTy.re at the inference layer if ever needed."
id: 0008
date: 2026-06-17
status: accepted
supersedes: null
decided_by: user
last_read: 0
---

# 0008 — Reject Information Field Theory from core, adopt NIFTy.re at inference layer

## Context

As the jaxstro ecosystem grew, a proposal emerged to include Information Field Theory (IFT) capabilities in the core. IFT is a powerful statistical framework used in some astrophysics applications, and the question was whether to include it in jaxstro or keep it external.

## Decision

**Reject IFT from jaxstro core.** If/when needed, adopt `NIFTy.re` as an upper-layer inference library (not part of the foundation). Seed a future ADR for this decision.

## Rationale

- **Thin-foundation principle**: IFT is a heavy framework that violates the thin-foundation posture (see ADR-0001). Adding it would import significant complexity and dependencies into the foundation that *every* downstream package inherits.
- **Specialized domain need**: IFT is not generic infrastructure; it is a specialized inference method for specific applications. It belongs in domain-specific layers (e.g., a future astrophysics-inference package), not the foundation.
- **No foundational dependency**: jaxstro does not itself need IFT; consumers that want it can pull in `NIFTy.re` independently.
- **Maintainability**: IFT's complexity and evolution would create maintenance burden on the foundation layer, slowing releases and decoupling version cycles.
