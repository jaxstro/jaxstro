---
title: "ADR 0002 — Adopt equinox as core dependency"
description: "Adopt equinox as a required base dependency for immutable PyTree state and module design."
id: 0002
date: 2026-06-17
status: accepted
supersedes: null
decided_by: user
last_read: 0
---

# 0002 — Adopt equinox as core dependency

## Context

jaxstro needs a foundation for immutable PyTree state management and module design. The options were equinox (Kidger ecosystem, JAX-native, minimal footprint) versus other patterns.

## Decision

**Adopt equinox as a required base dependency** alongside jaxtyping. Use equinox modules for state and PyTree-friendly designs throughout.

## Rationale

- Equinox is the Kidger ecosystem standard, proven in JAX scientific code.
- Minimal footprint and JAX-first philosophy align with thin-foundation posture.
- Enables immutable, composable state patterns compatible with jax.grad/jit/vmap.
- No competing dependencies on the same niche (unlike unxt/quax/astropy for units).
