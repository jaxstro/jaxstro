---
title: "ADR 0005 — Diataxis docs with ADR meta sections"
description: "Use Diataxis as the user-doc spine with first-class ADR, validation, and dev-log meta sections."
id: 0005
date: 2026-06-17
status: accepted
supersedes: null
decided_by: user
last_read: 3
---

# 0005 — Diátaxis docs spine with first-class ADR and knowledge-web meta sections

## Context

jaxstro needed a docs architecture that serves both a new graduate student and future maintainers as the single source of truth. The options were to follow progenax's Diátaxis model strictly, or extend it with additional meta sections (ADRs, validation, decision-log).

## Decision

**Use Diátaxis (Tutorial, Explanation, How-To, Reference) as the spine for user-facing quadrants.** Add first-class meta sections adjacent to architecture: numbered ADR log (30-decisions), validation tables (60-validation), and development log (90-development-log). Include an explicit dual-front-door landing so users can enter via "Learn the methods" or "Look up the API."

## Rationale

- **Diátaxis clarity**: proven user-doc framework for scientific software.
- **First-class ADRs**: decisions are load-bearing and deserve prominence *adjacent* to architecture prose (not tucked under it). ADRs + narrative-why read as one rationale cluster.
- **Knowledge-web**: narrative (10-theory) bridges to atomic decisions (30-decisions) bridges to API (40-api) — creates a high-fidelity onboarding path for research students unfamiliar with the codebase.
- **Meta pages**: validation results, development log, and release notes are project-critical and earn their own sections.
