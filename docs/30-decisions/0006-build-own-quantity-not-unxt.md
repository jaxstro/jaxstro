---
title: "ADR 0006 — Build own Quantity, not unxt/quax"
description: "Build a pure-equinox, zero-dependency Quantity class rather than adopting unxt or quax."
id: 0006
date: 2026-06-17
status: accepted
supersedes: null
decided_by: user
last_read: 1
---

# 0006 — Build own Quantity (pure-equinox) instead of adopting unxt or quax

## Context

jaxstro needed unit-aware array support for the emerging need to track dimensions at I/O edges. Options were to adopt unxt (SoTA, astropy-backed), adopt quax (transparent dispatch), or build a minimal custom `Quantity` class from scratch.

## Decision

**Build our own `jaxstro.quantity: Quantity` class** as a pure-equinox module (value + dimension) with explicit `.to()/.value` unwrap at JAX boundaries. Zero new dependencies. Opt-in and experimental in 0.1.0; no sibling package forced to migrate.

## Rationale

- **Lightweight foundation**: unxt uses astropy as a backend (violates thin-foundation posture), adds heavy weight to everything depending on jaxstro, and overlaps/subsumes jaxstro's `units` module.
- **No treadmill**: quax adds jaxtyping-compatible transparent dispatch but requires a permanent primitive-coverage treadmill (new `@quax.register` for every new function/transform).
- **Philosophically aligned**: pure-equinox `Quantity` (explicit units, host-side resolution) matches jaxstro's ethos; zero new deps; sidesteps the primitive long-tail.
- **Boundary clarity**: `Quantity` (dimensions-on-values) and `UnitSystem` (code-unit scaling) are complementary: use `Quantity` at I/O edges, unwrap to raw array + `UnitSystem` for dimensionless compute interior.
- **Future interop**: if transparent interop becomes a hard requirement later, add a thin `quax` adapter at the consuming layer, not in the foundation.

*Note*: Lands as experimental-only in 0.1.0 if it does not delay the release; else deferred to 0.2.0.
