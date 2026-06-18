---
title: "ADR 0003 — Standalone uv project with hatchling"
description: "Standardize jaxstro as a standalone uv project on the hatchling build backend with py.typed and uv.lock."
id: 0003
date: 2026-06-17
status: accepted
supersedes: null
decided_by: user
last_read: 0
---

# 0003 — Standalone uv project with hatchling build backend

## Context

jaxstro needed to standardize its build tooling to match release-readiness. Previously used setuptools; sibling packages (progenax, fluxax) use hatchling. Options were to adopt hatchling or remain on setuptools.

## Decision

**Standardize jaxstro as a standalone uv project** using hatchling as the build backend, py.typed marker, [tool.uv] configuration, and uv.lock. Do not use a root workspace pattern.

## Rationale

- **Consistency**: matches progenax/fluxax ecosystem standard.
- **Release-ready**: hatchling is lighter and modern compared to setuptools.
- **Tooling alignment**: uv ecosystem standardization enables reproducible builds and `uv lock --check` CI validation.
- **Per-package independence**: each package remains a standalone project for independent release cadence (via path-sources), avoiding workspace entanglement.
