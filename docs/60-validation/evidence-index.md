---
title: Scientific evidence index
---

# Scientific evidence index

Evidence classes remain distinct: a source citation is not a numerical validation, and a benchmark is not a physical acceptance result.

| Evidence ID | Class | Artifact | Source revision | Optional-data policy |
| --- | --- | --- | --- | --- |
| `atmosphere.interpolation-policy` | scientific_policy | [artifact](../validation/atmosphere-interpolation.json) | `sha256:3daa0e47e3857cb6f25b019d3bc3d477a3de1e321c5cd54aed947ccf9d98e678` | Policy regeneration requires approved local atmosphere holdouts. |
| `provenance.cards` | source_provenance | [artifact](../40-api/provenance/index.md) | `sha256:102af76268ad0fb81de09e0bdcce9a421dda61c20c19a8d6ce5dce9f0bc74451` | Uses repository-owned source cards; no runtime dataset required. |
| `rootfinding.implicit-gradients` | computational | [artifact](../validation/implicit-root-gradients.json) | `d292b9000ce98f62a512d56d1b3052604adc7f0d` | No external data required. |
| `rootfinding.performance` | computational | [artifact](../validation/rootfinding-performance.json) | `fd28c3a592d9feff5145f4f6d02263af22f2e228` | No external data required. |
| `spectra.performance` | computational | [artifact](../validation/spectra-performance.json) | `dbfde557532be4df515d16253933321f4f58c19b` | Requires the local NewEra artifact and the declared data extra. |
