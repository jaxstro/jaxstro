---
title: "ADR 0007 — CGS as default unit system"
description: "Set units.DEFAULT to CGS, the true base system from which all domain systems are scaled."
id: 0007
date: 2026-06-17
status: accepted
supersedes: null
decided_by: user
last_read: 0
---

# 0007 — CGS as default unit system

## Context

jaxstro needed a default unit system for the `DEFAULT` constant, used by generic code that doesn't yet have domain-specific units. Previously set to `ASTRO_DYNAMICAL` (Msun, pc, Myr). The decision was whether to keep ASTRO_DYNAMICAL or shift to CGS.

## Decision

**Set `units.DEFAULT = CGS`** as the foundation-level default, replacing ASTRO_DYNAMICAL. Export as `jaxstro.DEFAULT_UNITS` in the public API.

## Rationale

- **Foundation principle**: CGS is the true base unit system; all domain systems (ASTRO_DYNAMICAL, ASTRO_PLANETARY) are *scaled* versions of CGS and can be derived from it.
- **No astrophysics bias at the foundation**: jaxstro is generic infrastructure, not inherently astrophysical. CGS reflects this neutrality.
- **Consistency with constants**: jaxstro.constants (CODATA 2018) are defined in CGS; using CGS as default eliminates any mismatch or implicit conversions.
- **Domain defaults**: sibling packages define their own `DEFAULT_UNITS` (e.g., progenax.DEFAULT_UNITS = ASTRO_DYNAMICAL), overriding jaxstro's base default where domain-specific scaling is needed.
