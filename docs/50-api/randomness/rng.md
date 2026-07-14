---
title: PRNG key mechanics
---

# PRNG key mechanics

## Owner import path

`jaxstro.numerics.rng`

## Purpose

Explicit splitting, tree-shaped allocation, and index folding for JAX random
keys.

## Public records and callables

`KeyArray`, `split_key`, `split_tree`, and `fold_in_indices`.

## Shape and dtype expectations

Keys use JAX key arrays. Split counts and tree shapes are concrete; folded
indices are integer arrays.

## JAX transforms and AD classification

PRNG operations compose with JIT and VMAP but are discrete and have no
scientific AD claim.

## Failure behavior

Invalid counts, shapes, or key representations raise through JAX. Keys are
never created from hidden global state.

## Contract and evidence links

See [](../../20-methods/probability-sampling/random.md) and
[](../../40-workflows/reproducible-research/random-state-ownership.md).

## Canonical import example

```python
from jaxstro.numerics.rng import split_tree
```
