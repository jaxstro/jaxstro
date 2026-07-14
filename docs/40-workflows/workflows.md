---
title: Research workflows
description: Move from a scientific representation to an auditable and bounded claim.
---

# Research workflows

Use this page when you have a scientific question and need to turn it into a
computation whose assumptions, execution, evidence, and claim can be inspected.

The organizing chain is:

```text
representation -> computation plan -> execution -> audit -> evidence -> claim
```

Readers can enter through a research question, a concrete data pipeline, or an
executable investigation. The workflow is not necessarily linear: an audit may
send you back to revise the representation or computation plan.

::::{grid} 1 2 2 2
:::{card} Scientific ML
:link: ./scientific-ml/preprocessing.md

Plan preprocessing, data partitions, and fixed-step training without leakage
or hidden state. These pages describe a planned capability, not a current API.
:::
:::{card} Data pipelines
:link: ./data-pipelines/query-atmosphere-spectra.md

Prepare quantities and atmosphere artifacts at explicit host/runtime
boundaries while retaining released coordinates and provenance.
:::
:::{card} Differentiable research
:link: ./differentiable-research/what-jax-differentiates.md

Connect the executed JAX program to derivative audits, branch behavior, and
scientifically meaningful sensitivity claims.
:::
:::{card} Reproducible research
:link: ./reproducible-research/random-state-ownership.md

Own random keys, runtime manifests, source cards, and evidence-to-claim
boundaries explicitly.
:::
:::{card} Executable investigations
:link: ./investigations/investigations.md

Predict, compute, audit, and state the warranted claim with repository-owned
examples and validation targets.
:::
::::

:::{note}
A passing execution is one link in the chain. It does not by itself establish
that the representation is scientifically adequate or that the final claim is
warranted.
:::

| Family | Status | Primary output |
| --- | --- | --- |
| Scientific ML | Planned | Explicit preprocessing, split, batch, and audit contracts |
| Data pipelines | Current | Validated local artifacts and prepared runtime inputs |
| Differentiable research | Current guidance | Derivative meaning and independent audit plan |
| Reproducible research | Current guidance and APIs | Key lineage, manifests, evidence boundaries |
| Investigations | Current and executable | Metrics, audits, limitations, and bounded claims |

Start with [](./differentiable-research/science-patterns.md) when the research
question is primary, [](./data-pipelines/query-atmosphere-spectra.md) when an
artifact is primary, or [](./investigations/investigations.md) when you want a
complete executable example.
