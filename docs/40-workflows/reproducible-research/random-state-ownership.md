# Explicit random-state ownership

Use this page when stochastic computation must be repeatable, reviewable, and
safe from accidental key reuse across batches or independent experiments.

## Keys are inputs, not ambient state

JAX random functions consume explicit keys. A reproducible workflow records a
root key and the derivation boundary for every child key. Key lineage should
follow semantic ownership: one child for split construction, another for model
initialization, and distinct descendants for training steps, posterior draws,
or replicated experiments.

```{math}
:label: eq-workflow-key-tree

\kappa_{r,a},\kappa_{r,b}
=
\operatorname{split}(\kappa_r),
\qquad
\kappa_{r,a}\ne\kappa_{r,b}.
```

The inequality is an identity distinction, not a proof of statistical
independence. Independence depends on the generator, the splitting construction,
and the way keys are consumed.

## Splitting boundaries

Split at the owner boundary, then pass the child key explicitly. A helper must
not split a hidden module-level key. Reusing the same key for two random draws
can create identical samples or unintended correlations. Conversely, changing
the shape or batching strategy may change how a key is consumed even when the
root key is unchanged.

For batched execution, decide whether each lane receives its own child key or a
single random array is generated before batching. Record that choice. `vmap`
does not automatically create independent keys, and device parallelism requires
an explicit fold-in or split policy tied to stable lane identity.

## Reproducibility is not independence

Reproducibility means that the declared software, inputs, key lineage, shapes,
and execution policy recover the same result within the stated backend and
precision contract. Statistical independence is a property of the sampling
design. Replaying one key tree demonstrates deterministic regeneration; it does
not validate Monte Carlo error estimates or independence assumptions.

## Audit procedure

1. Name the root-key source and serialization convention.
2. Draw the key tree before execution and assign semantic owners.
3. Check that no consumed key appears twice.
4. Record array shapes, batch order, device/lane identity, and fold-in values.
5. Replay the workflow and compare the declared metrics or hashes.
6. Use replicated child roots when estimating stochastic uncertainty.
7. State which backend and precision changes are allowed to change bit patterns.

## Provenance

A runtime manifest should record root-key identity, split algorithm/version,
semantic child labels, batching policy, and relevant environment. Storing only
the integer seed is insufficient when the surrounding split tree or data order
can change. The manifest belongs beside the result; it does not need to expose
secret data or credentials.

## Where the claim stops

An explicit key tree prevents hidden random state and supports replay. It does
not prove statistical independence, unbiased estimation, convergence, or
scientific adequacy.

## Connected ideas

See [](../../20-methods/probability-sampling/random.md),
[](../../10-foundations/mathematical-objects/probability-and-distributions.md),
[](../scientific-ml/data-plans.md), and [](./provenance.md).
