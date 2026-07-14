# Fixed-step, auditable training

Use this page when an optimization run must be replayable as a finite executed
program with explicit state, keys, metrics, checkpoints, and claim boundaries.

:::{important} Planned Jaxstro capability
`jaxstro.ml` does not exist and has no implementation schedule. This page is a
design contract for a possible future audit surface, not a training API.
:::

## The scientific question

What exactly ran when a model was trained, and which parts of that execution
support a numerical claim rather than a scientific adequacy claim?

## Prerequisites

Review [](../../10-foundations/models-and-computation/from-relations-to-differentiable-programs.md),
[](../../20-methods/change-constraints-evolution/autodiff.md), and
[](../../20-methods/change-constraints-evolution/optimization.md).

## Mathematical objects

Keep separate records for the model $f_\theta$, loss, gradient computation,
optimizer state $s_k$, update transformation $U$, data plan, random keys,
metrics, checkpoints, and evidence. Conflating these objects makes replay and
failure diagnosis ambiguous.

## Core derivation

An auditable execution performs exactly `K` state transitions:

```{math}
:label: eq-ml-training-update

g_k=\nabla_\theta \mathcal{L}(\theta_k;B_k,\kappa_k),
\qquad
\theta_{k+1}=U(\theta_k,g_k,s_k),
\qquad
k=0,\ldots,K-1.
```

The finite output is $\theta_K$ plus a trace. The trace must identify the batch
$B_k$, key $\kappa_k$, optimizer-state transition, metric definitions, and any
nonfinite or rejected update. The transition sequence in
[](#eq-ml-training-update) is the load-bearing contract for that trace. Early
stopping defines a different executed map
unless the stop decision and resulting length are explicit artifacts.

## Assumptions and failure boundaries

Loss scaling, reduction semantics, regularization, mixed precision, gradient
clipping, and parameter freezing alter the map and must be declared. A metric
name without its formula and units is insufficient. Checkpoints must bind model,
optimizer state, step, data-plan identity, and key lineage. Resuming from a
partial checkpoint or silently changing the data order fails closed.

## Worked conceptual example

For a 100-step fit, freeze the split and batch plans, derive a training key,
record the initial model and optimizer state, and run a fixed-length scan. Emit
loss and gradient-norm metrics with definitions. Replaying the same artifact
must recover the stored checkpoint within the declared precision policy.

## Ownership boundary

Equinox owns callable model and PyTree construction. Optax owns optimizer
transformations and optimizer state. JAX owns array transformations and fixed
execution. Jaxstro could own only a thin domain-agnostic plan, trace, and
provenance contract. It must remain optimizer agnostic and must not duplicate
model or inference frameworks.

## Proposed interface

A future interface would accept explicit model, loss, update, plan, keys, and
initial state. It would return fixed-shape state and audit records. This
description intentionally defines no importable symbol.

## Evidence required before implementation

Evidence includes update-equation reference fixtures, replay from checkpoints,
JIT and scan behavior, finite and nonfinite gradients, key-use audits, masked
batch parity, metric-definition validation, optimizer-protocol compatibility,
and deterministic rendering of run manifests.

## Where the claim stops

Fixed-step execution proves only what program ran. It does not prove model
adequacy, convergence to a useful optimum, calibrated uncertainty,
generalization, or scientific validity.

## Connected ideas

Continue to [](./ecosystem-boundaries.md), [](./data-plans.md), and
[](../reproducible-research/evidence-and-claim-boundaries.md).
