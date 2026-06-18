---
title: "ADR 0012 — Ecosystem multi-package maintenance"
description: "Per-package editable path-sources with independent release cadence, standardized tooling, and a sibling smoke-test CI job that fails fast on downstream breakage."
id: 0012
date: 2026-06-18
status: accepted
supersedes: null
decided_by: user
last_read: 2
---

# 0012 — Ecosystem multi-package maintenance: per-package editable path-sources + sibling smoke-test CI

## Context

The jaxstro ecosystem consists of multiple interdependent packages (jaxstro core, gravax, progenax, fluxax, with planned startrax and stellax). Each package depends on jaxstro core but may release on independent cadences. The question was how to structure development, testing, and CI to maintain a coherent ecosystem while allowing packages to evolve independently without tight version coupling.

## Decision

**Per-package editable path-sources with independent release cadence; standardize tooling; add sibling smoke-test CI.**

1. **Editable path-sources**: Each package uses `path = "."` in its `pyproject.toml` for local jaxstro core (or sibling dependencies), allowing local development without requiring jaxstro to publish pre-release wheels. Enables rapid iteration across the ecosystem.
2. **Independent release cadence**: Each package (jaxstro, gravax, progenax, fluxax, …) maintains its own version number and release schedule via per-package `uv.lock` and `pyproject.toml`. Solver-library churn in one package does not force synchronized releases across the ecosystem.
3. **Standardized tooling**: All packages follow the same build/lint/test setup: hatchling, `py.typed`, `uv lock --check`, ruff + mypy, pytest with 3-tier organization (unit/integration/validation), CHANGELOG, semver + deprecation policy.
4. **Sibling smoke-test CI**: A jaxstro-side CI job runs the full test suites of downstream consumers (progenax, fluxax, gravax) against the candidate jaxstro build. A break in jaxstro fails immediately in jaxstro's CI (not discovered weeks later when someone runs downstream). Reciprocally, major changes in (e.g.) gravax that might affect fluxax are surfaced early.

## Rationale

- **No version tyranny**: jaxstro core can release bug fixes or new features without forcing synchronized releases of all downstream packages. Each package pins `jaxstro>=X,<Y` and upgrades on its own schedule.
- **Rapid local development**: path-sources allow a developer to change jaxstro core and test progenax/fluxax consumers in the same session without rebuilding wheels or publishing to a test index.
- **Ecosystem coherence**: standardized tooling (same ruff/mypy config, same CI template, same semver policy) ensures packages integrate cleanly despite independent releases.
- **Early failure detection**: sibling smoke-test CI catches integration regressions in jaxstro that would silently break downstream consumers. Mirrors the pattern used in large monorepos (e.g., JAX/Equinox ecosystem itself).
- **Scalability**: this design supports growing the ecosystem (startrax, stellax, future packages) without exponential CI complexity or version-coupling.

## Notes

Complements ADR-0010 (layered config architecture with jaxstro-lab above core). Works alongside ADR-0001 (thin foundation) to keep jaxstro lightweight while enabling robust downstream integration. Per-package parity-tested hoists (moving numerics from progenax/fluxax up to jaxstro) rely on this structure to verify non-regression across sibling suites.
