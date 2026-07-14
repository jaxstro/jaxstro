# Deterministic data plans

Use this page when train, validation, test, shuffle, padding, and batching
choices must be explicit artifacts rather than hidden mutable state.

:::{important} Planned Jaxstro capability
`jaxstro.ml` does not exist and has no implementation schedule. This page
defines a possible future data-plan contract; it does not document importable
software.
:::

## The scientific question

How can repeated training runs consume the same examples in the same declared
order, while preserving honest held-out evaluation and fixed JAX shapes?

## Prerequisites

Review [](../../10-foundations/models-and-computation/what-is-a-model.md) and
[](../../20-methods/probability-sampling/random.md). Randomness controls the
construction of a plan; it must not remain an invisible property of an iterator.

## Mathematical objects

For $N$ records, a split artifact stores integer index sets, the source dataset
identity, a split algorithm/version, and key lineage. A batch artifact stores
ordered indices, masks, shapes, and the policy for incomplete batches.

## Core derivation

The split contract requires pairwise disjoint sets whose union is the declared
population:

```{math}
:label: eq-ml-disjoint-splits

I_{\mathrm{train}}\cap I_{\mathrm{validation}}=\varnothing,
\quad
I_{\mathrm{train}}\cap I_{\mathrm{test}}=\varnothing,
\quad
I_{\mathrm{validation}}\cap I_{\mathrm{test}}=\varnothing,
\quad
\bigcup_r I_r=\{0,\ldots,N-1\}.
```

For batch size $B$, each executable batch has the same array shape. If the last
batch contains $m<B$ records, the plan must choose exactly one documented
policy: drop it, pad it and provide a Boolean mask, or construct a separate
compiled shape. Padding values cannot contribute to loss or metrics.

## Assumptions and failure boundaries

Splits may need grouping or stratification to prevent related observations from
crossing boundaries. A deterministic random permutation is not automatically a
scientifically valid split. Dataset changes invalidate index artifacts unless a
stable row identity and reconciliation rule are supplied. No hidden split or
shuffle state is permitted. Key reuse, ambiguous padding, implicit dropping,
and host iterators whose state is not recorded all fail the intended contract.

## Worked conceptual example

Create the three index arrays once from a named root key. Store the dataset hash,
split fractions, grouping columns, child-key identifiers, and exact indices.
Create an epoch plan containing fixed-shape index blocks and masks. A rerun
loads the plan rather than asking an iterator to reconstruct it from ambient
state.

## Ownership boundary

The host owns dataset discovery, row identity, scientific grouping policy, and
plan construction. JAX consumes arrays of indices and masks. A future Jaxstro
surface could validate and serialize domain-agnostic plans; it would not decide
what constitutes leakage in a particular survey or simulation.

## Proposed interface

Any proposal must make indices, metadata, key lineage, padding policy, and masks
visible data. Convenience iterators cannot be the source of truth.

## Evidence required before implementation

Required checks include disjointness and coverage, deterministic regeneration,
dataset-hash mismatch failure, group-leakage fixtures, key-lineage audits,
fixed-shape JIT execution, mask-correct loss reduction, and parity between a
stored plan and a replayed run.

## Where the claim stops

A reproducible split does not establish representativeness, absence of dataset
shift, adequate sample size, or model generalization.

## Connected ideas

Continue to [](./auditable-training.md),
[](../reproducible-research/random-state-ownership.md), and
[](../reproducible-research/provenance.md).
