---
title: "ADR 0010 — Ecosystem config architecture"
description: "Layered config: core stays config-agnostic, per-package pydantic schemas, hydra at the lab layer."
id: 0010
date: 2026-06-17
status: accepted
supersedes: null
decided_by: user
last_read: 0
---

# 0010 — Ecosystem config architecture: layered (core-agnostic / per-package pydantic / hydra at lab)

## Context

Anna wants hydra + pydantic YAML configs for the *full system* (cross-package experiments). Question: does config machinery belong in jaxstro core, and what is the best design across many repos?

## Decision

A **three-layer** config architecture, dependencies flowing one way:

1. **jaxstro core stays config-agnostic** — no hydra/omegaconf/pydantic as hard deps (they would be dragged into every downstream import, violating ADR-0001). Core knows only Equinox Modules + `jaxstro.params`.
2. **Per-package pydantic schemas live with the package that owns the model** (`gravax`→`GravaxConfig`, progenax IMF/profile schemas, fluxax instrument schemas). Validation sits next to the type. They follow a shared `from_config(cfg)->eqx.Module` / `Configurable` convention. That *light convention* MAY live in core behind an optional `[config]` extra (`jaxstro.config`); pydantic is light enough for an extra, **hydra is not**.
3. **Hydra composition + CLI lives at the experiment/application layer** — a dedicated **`jaxstro-lab` repo** (separate distribution, top of the DAG). Hydra is an app framework (entry points, working-dir, multirun); putting it in a library forces every importer into that framework.

Tooling: **pydantic** (schemas/validation) + **hydra** (composition/sweeps) bridged by **hydra-zen** (build hydra configs from typed Python). Add **tyro** later as a *second* lightweight typed-CLI entry point — both front-ends consume the **same pydantic schemas as the single source of truth**.

## Rationale

- **No dependency cycle (the hard blocker):** cross-package experiments import gravax/progenax/fluxax; core is imported *by* those. A hydra lab inside the core distribution → `core → siblings → core` cycle. The lab must sit *above* core. (Repo boundary ≠ distribution boundary: a lab MAY live in the jaxstro repo, but MUST be a separate distribution above core. Anna chose a dedicated **`jaxstro-lab` repo**.)
- **pydantic-with-package** keeps validation close to the type and avoids one mega-schema repo.
- **hydra-zen** resolves the hydra↔pydantic seam (hydra is OmegaConf/dataclass-native, not pydantic-native).
- Mirrors the lib-vs-app split (GalSim/imSim, PyTorch/Lightning).

## Notes

`jaxstro.params` (ADR-0009) is the model↔vector bridge; `jaxstro.config` is the future config↔model bridge — complementary thin layers. **Informax** (inference + OED) and `jaxstro-lab` are the app-layer consumers. Interacts with the deferred release-packaging decision (jaxstro namespace? rename `jaxstro`→`jaxstro-core`?) — to be decided at release staging.
