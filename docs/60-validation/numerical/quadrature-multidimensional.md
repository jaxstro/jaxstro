---
title: Multidimensional quadrature evidence
description: Truth, replay, randomized calibration, comparison, and performance boundaries for Phase B.
---

# Multidimensional quadrature evidence

This page connects each public Phase B claim to the artifact that owns it.
Passing one row does not promote the other rows, and none establishes universal
superiority over another library.

:::{important}
The validated geometry is a finite hyperrectangle. Simplexes, spheres,
manifolds, and domain-specific scientific acceptance are Phase C work.
:::

## Evidence map

| Question | Evidence owner | What it warrants | What it does not warrant |
| --- | --- | --- | --- |
| Do values and work records match independent truths? | [truth artifact](../../validation/quad-multidim-truth.json) | Declared analytic and numerical cases at recorded tolerances | Accuracy for unresolved discontinuities or arbitrary integrands |
| Does first-order replay match the accepted formula? | [replay artifact](../../validation/quad-multidim-replay.json) | Declared parameter, bound, measure, and quantity derivatives | Derivatives of discrete refinement or higher derivatives |
| Do randomized intervals meet the declared calibration experiment? | [RQMC artifact](../../validation/quad-rqmc-calibration.json) | Real-scalar fixed-look and bounded sequential coverage in the frozen campaign | Per-integral certainty or complex/vector confidence intervals |
| How were external capabilities compared? | [comparison artifact](../../validation/quad-multidim-comparisons.json) | Only the recorded exact, strong-match, node-match, family-match, or capability label | Algorithmic identity from a broad family label |
| Did observed replay memory warrant optimization? | [observed-memory artifact](../../validation/quad-multidim-memory.json) | No frozen CPU replay case exceeded its matched primal RSS by the 10 GiB materiality criterion | A separate device-memory measurement or a universal memory claim |

## Estimator meanings

| Family | `QuadError.kind` | Interpretation |
| --- | --- | --- |
| Fixed tensor | `UNAVAILABLE` | No truncation-error estimate is supplied |
| Adaptive tensor | `REFINEMENT_DIFFERENCE` | Difference between accepted global nested levels |
| Adaptive cubature | `EMBEDDED_RULE` | Shared-node high/low Genz-Malik disagreement |
| Sparse grid | `SPARSE_GRID_SURPLUS` | Hierarchical frontier or accepted-index evidence |
| Deterministic Sobol | `UNAVAILABLE` | No randomization, therefore no confidence interval |
| Randomized Sobol | `CONFIDENCE_INTERVAL_HALF_WIDTH` | Fixed-look Student-t or bounded sequential empirical-Bernstein evidence |

## Replay and quantity boundary

Replay differentiates

```{math}
:label: eq-validation-multidim-replay
\widehat{I}(\theta)
=
\sum_{i=1}^{N_{\mathrm{accepted}}}
w_i(\theta)f(\boldsymbol{x}_i(\theta),\theta),
```

with the accepted structure frozen. Opt-in quantity normalization may use
heterogeneous coordinate units and restores the result unit after the raw JAX
kernel. It remains alpha: downstream package adoption and direct
`Quantity`-PyTree quotient-unit Jacobians are not implied.

## Performance disposition

The immutable baseline fired only the analytic memory-proxy trigger at
dimension 16:

```{math}
:label: eq-validation-memory-trigger
\frac{36{,}864\ \mathrm{bytes}}{16{,}384\ \mathrm{bytes}}
=2.25.
```

This is expected linear coordinate storage and a small absolute proxy, not an
observed peak-memory failure. The
[optimization addendum](../../superpowers/specs/2026-07-18-quad-phase-b-memory-optimization-addendum.md)
therefore authorizes no runtime change until a measured campaign demonstrates
a material case.

The frozen fresh-process CPU campaign subsequently measured 72 supported
primal/replay cases over dimensions 2, 4, 8, and 16 and Sobol levels 8, 12,
and 16. Its largest matched replay increment was 45,973,504 bytes (43.8 MiB),
for scalar eight-replicate scrambled Sobol at dimension 16 and level 16. That
is below the predeclared 10 GiB materiality threshold, so it authorizes no
runtime optimization. The remaining 24 randomized array-payload cases are an
intentional Phase B rejection: calibrated randomized intervals are real-scalar
only. The active CPU backend exposes no reliable separate device-memory metric.

## Reproduce the focused evidence

The following checks are intentionally narrower than the full release gate:

```bash
uv run --no-sync python scripts/generate_quad_multidim_evidence.py --check
uv run --no-sync python scripts/generate_quad_rqmc_evidence.py --check
uv run --no-sync python scripts/benchmark_quad_multidim.py --suite baseline --check
uv run --no-sync python scripts/measure_quad_multidim_memory.py --check
```

The exhaustive test, documentation, clean-wheel, and observed-memory campaign
remain separate release gates.
