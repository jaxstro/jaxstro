---
title: "ADR 0001 — Thin foundation posture"
description: "Own generic dependency-free primitives; refuse solver libraries, adopting only jaxtyping + equinox."
id: 0001
date: 2026-06-17
status: accepted
supersedes: null
decided_by: user
last_read: 0
---

# 0001 — Thin foundation posture over all-deps façade

## Context

jaxstro is the foundation layer of the differentiable-astrophysics ecosystem, used by gravax, progenax, fluxax, and planned startrax/stellax. The decision was whether to include solver libraries (diffrax, optimistix, lineax, quadax, interpax) as optional or required dependencies, or to remain lightweight and require downstream packages to pull those in directly.

## Decision

**Own generic dependency-free primitives; refuse solver libraries.** Adopt only the ecosystem foundation pair (jaxtyping + equinox). Do not absorb fast-moving solver libraries — those are peer/upper layers each domain package pulls in directly, mirroring the Kidger ecosystem topology.

## Rationale

- **Blast radius**: a units-only consumer must not inherit diffrax+optimistix+numpyro transitively.
- **Version decoupling**: solver-library churn must not force a jaxstro release and re-lock of every downstream package.
- **API stability**: a foundation promises stability; wrapping fast-moving solver APIs imports their instability and doubles the maintenance surface.
- **No lag**: callers reach SoTA solvers directly without jaxstro as a bottleneck.

This mirrors Kidger's layering (jaxtyping → equinox → {lineax, optimistix, diffrax}), which is SoTA *because* the foundation stays thin.
