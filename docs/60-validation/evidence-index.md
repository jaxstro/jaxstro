---
title: Scientific evidence index
---

# Scientific evidence index

Evidence classes remain distinct: a source citation is not a numerical validation, and a benchmark is not a physical acceptance result.

| Evidence ID | Class | Artifact | Source revision | Optional-data policy |
| --- | --- | --- | --- | --- |
| `atmosphere.interpolation-policy` | scientific_policy | [artifact source](https://github.com/drannarosen/jaxstro/blob/main/docs/validation/atmosphere-interpolation.json) | `sha256:3daa0e47e3857cb6f25b019d3bc3d477a3de1e321c5cd54aed947ccf9d98e678` | Policy regeneration requires approved local atmosphere holdouts. |
| `provenance.cards` | source_provenance | [artifact source](https://github.com/drannarosen/jaxstro/blob/main/docs/50-api/research-infrastructure/source-provenance/source-provenance.md) | `sha256:0da7b3240bfdd33b47a0f28d9223deca67b854363d5a801fcec62aed2b16eced` | Uses repository-owned source cards; no runtime dataset required. |
| `quad.multidim.comparisons` | computational | [artifact source](https://github.com/drannarosen/jaxstro/blob/main/docs/validation/quad-multidim-comparisons.json) | `e091640eefc7322714126e06e380ca3f9486af07` | No external data required; labels apply only to the recorded comparator cases. |
| `quad.multidim.observed-memory` | computational | [artifact source](https://github.com/drannarosen/jaxstro/blob/main/docs/validation/quad-multidim-memory.json) | `1048b5b5fab2efb8e2c418c5249afbe5c715d603` | No external data required; CPU process RSS only, with no separate device-memory metric. |
| `quad.multidim.performance-baseline` | computational | [artifact source](https://github.com/drannarosen/jaxstro/blob/main/docs/validation/quad-multidim-performance-baseline.json) | `e091640eefc7322714126e06e380ca3f9486af07` | No external data required; this immutable baseline contains an analytic memory proxy, not observed peak memory. |
| `quad.multidim.replay-and-astro` | computational | [artifact source](https://github.com/drannarosen/jaxstro/blob/main/docs/validation/quad-multidim-replay.json) | `not-recorded` | No external data required; first-order accepted-formula replay only. |
| `quad.multidim.truth` | computational | [artifact source](https://github.com/drannarosen/jaxstro/blob/main/docs/validation/quad-multidim-truth.json) | `not-recorded` | No external data required; finite-hyperrectangle truths only. |
| `quad.performance` | computational | [artifact source](https://github.com/drannarosen/jaxstro/blob/main/docs/validation/quad-performance.json) | `35736fcc0fdaa7932b3bc67780567e24cf94638a` | No external data required; timings are bounded to the recorded machine. |
| `quad.replay-derivatives` | computational | [artifact source](https://github.com/drannarosen/jaxstro/blob/main/docs/validation/quad-replay-derivatives.json) | `251d997723114c5a92d388cc87aadc0e68fd0799` | No external data required. |
| `quad.rqmc-calibration` | computational | [artifact source](https://github.com/drannarosen/jaxstro/blob/main/docs/validation/quad-rqmc-calibration.json) | `not-recorded` | No external data required; coverage is limited to the frozen real-scalar campaigns. |
| `rootfinding.implicit-gradients` | computational | [artifact source](https://github.com/drannarosen/jaxstro/blob/main/docs/validation/implicit-root-gradients.json) | `a3f0d4bd563555e67a3ec0ca8e1ed0e8be671f60` | No external data required. |
| `rootfinding.performance` | computational | [artifact source](https://github.com/drannarosen/jaxstro/blob/main/docs/validation/rootfinding-performance.json) | `a3f0d4bd563555e67a3ec0ca8e1ed0e8be671f60` | No external data required. |
| `spectra.performance` | computational | [artifact source](https://github.com/drannarosen/jaxstro/blob/main/docs/validation/spectra-performance.json) | `dbfde557532be4df515d16253933321f4f58c19b` | Requires the local NewEra artifact and the declared data extra. |
