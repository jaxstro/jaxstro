---
title: "ADR 0004 — Apache-2.0 license"
description: "Adopt Apache-2.0 for jaxstro, replacing BSD-3-Clause, to match the ecosystem standard."
id: 0004
date: 2026-06-17
status: accepted
supersedes: null
decided_by: user
last_read: 0
---

# 0004 — Apache-2.0 license replacing BSD-3-Clause

## Context

jaxstro was originally under BSD-3-Clause, but the broader ecosystem (progenax, fluxax) standardized on Apache-2.0. The decision was whether to align with the ecosystem standard or retain BSD-3-Clause for backward compatibility.

## Decision

**Adopt Apache-2.0 as the license for jaxstro**, replacing BSD-3-Clause. This is an ecosystem policy decision.

## Rationale

- **Ecosystem consistency**: progenax and fluxax are now Apache-2.0; one license across the foundation and higher packages simplifies compliance.
- **Legal alignment**: Apache-2.0 provides explicit patent grant and is standard for research software ecosystems.
- **No lock-in**: BSD-3-Clause → Apache-2.0 is a one-way compatible move (Apache is more permissive in patent terms).
