---
title: "ADR 0011 — Apache-2.0 license standardization"
description: "Migrate jaxstro to Apache-2.0 with PEP 639 SPDX metadata, removing the stale BSD-3-Clause classifier."
id: 0011
date: 2026-06-17
status: accepted
supersedes: null
decided_by: user
last_read: 1
---

# 0011 — Apache-2.0 license standardization

## Context

jaxstro inherited BSD-3-Clause license from early development. The jaxstro ecosystem (fluxax, progenax, gravax) standardized on Apache-2.0 during Phase 1. Audit revealed license inconsistency: jaxstro remained BSD-3-Clause while siblings moved to Apache-2.0.

## Decision

Migrate jaxstro to Apache-2.0 license to align with ecosystem policy. Update `pyproject.toml` to use `license = "Apache-2.0"` (SPDX expression per PEP 639), include `LICENSE` file in wheel, and remove stale BSD-3-Clause Trove classifier.

## Rationale

**Consistency**: All jaxstro ecosystem packages now use Apache-2.0 as the standard permissive open-source license. **Wheels**: Modern build tooling (PEP 639/SPDX) expects consistent license metadata; mixing SPDX expressions and deprecated Trove classifiers triggers warnings. Clean SPDX-only approach prevents metadata conflicts. **Precedent**: Fluxax and progenax already set the ecosystem standard; jaxstro as the shared foundation should follow.
