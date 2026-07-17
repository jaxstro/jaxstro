---
title: Scientific evidence index
---

# Scientific evidence index

Evidence classes remain distinct: a source citation is not a numerical validation, and a benchmark is not a physical acceptance result.

| Evidence ID | Class | Artifact | Source revision | Optional-data policy |
| --- | --- | --- | --- | --- |
| `atmosphere.interpolation-policy` | scientific_policy | [artifact source](https://github.com/drannarosen/jaxstro/blob/main/docs/validation/atmosphere-interpolation.json) | `sha256:3daa0e47e3857cb6f25b019d3bc3d477a3de1e321c5cd54aed947ccf9d98e678` | Policy regeneration requires approved local atmosphere holdouts. |
| `provenance.cards` | source_provenance | [artifact source](https://github.com/drannarosen/jaxstro/blob/main/docs/50-api/research-infrastructure/source-provenance/source-provenance.md) | `sha256:5b68caec5b185a32ca84dcfbac50bb70a94e52acb012ca062b19b5c90651a7a0` | Uses repository-owned source cards; no runtime dataset required. |
| `quad.performance` | computational | [artifact source](https://github.com/drannarosen/jaxstro/blob/main/docs/validation/quad-performance.json) | `35736fcc0fdaa7932b3bc67780567e24cf94638a` | No external data required; timings are bounded to the recorded machine. |
| `quad.replay-derivatives` | computational | [artifact source](https://github.com/drannarosen/jaxstro/blob/main/docs/validation/quad-replay-derivatives.json) | `88192affca0b0191da2d5371f295fde66223d106` | No external data required. |
| `rootfinding.implicit-gradients` | computational | [artifact source](https://github.com/drannarosen/jaxstro/blob/main/docs/validation/implicit-root-gradients.json) | `a3f0d4bd563555e67a3ec0ca8e1ed0e8be671f60` | No external data required. |
| `rootfinding.performance` | computational | [artifact source](https://github.com/drannarosen/jaxstro/blob/main/docs/validation/rootfinding-performance.json) | `a3f0d4bd563555e67a3ec0ca8e1ed0e8be671f60` | No external data required. |
| `spectra.performance` | computational | [artifact source](https://github.com/drannarosen/jaxstro/blob/main/docs/validation/spectra-performance.json) | `dbfde557532be4df515d16253933321f4f58c19b` | Requires the local NewEra artifact and the declared data extra. |
