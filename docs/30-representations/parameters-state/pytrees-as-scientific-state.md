---
title: PyTrees as scientific state
description: >-
  Structured dynamic arrays and static scientific metadata as the state transformed
  by JAX programs.
---

Use this page when deciding which parts of a scientific model should be dynamic array
leaves, which should be static metadata, and how a selected subset becomes fit state.

:::{important} Implemented Jaxstro capability
`jaxstro.params` operates on existing JAX and Equinox PyTrees and preserves their
structure while selecting free leaves. It does not define a universal model class.
:::

## Representation contract

| Contract field | Current representation |
| --- | --- |
| Mathematical object | A PyTree is a nested structure whose array leaves carry dynamic scientific state and whose container/static leaves define organization and metadata. |
| Physical convention | Tree structure and leaf meaning are model-owned; `Parameterization` records a PyTree-aligned free/fixed mask and bijector metadata. |
| Runtime owner | `jaxstro.params` owns selective PyTree/vector bridging, while JAX and Equinox own the general PyTree protocol. |
| Shape and unit policy | Each dynamic leaf keeps its original shape and caller-owned units; flattening concatenates only selected array entries into one vector. |
| Transform boundary | A stable tree definition and leaf shapes compose with `jit`, `vmap`, and AD; changing structure or static metadata can retrace and is not a dynamic branch. |
| Evidence | Parameter tests cover nested modules, static fields, empty and partial selections, ordering, reconstruction, and transformed losses. |
| Downstream interpretation boundary | Jaxstro does not decide what a leaf means, whether it is free, which cached values are valid, or how model state maps to observations. |

## One state, two views

A PyTree treedef $\tau$ separates structure from a deterministic sequence of leaves:

```{math}
:label: eq-pytree-flatten

\operatorname{flatten}(m)
=
(\tau;\ell_1,\ell_2,\ldots,\ell_k).
```

`Parameterization` uses a mask aligned with [](#eq-pytree-flatten) to partition free
and fixed array leaves. Free leaves are mapped to an unconstrained vector; fixed and
static leaves remain in the reconstructed model.

Dynamic arrays should contain values that change across evaluation. Static fields
should contain hashable metadata that defines the program, such as a leaf-selection
mask or a unit object. Static data participates in tracing and compilation identity,
so changing it can create a new compiled program.

## Shapes and batching

A free leaf of shape `(2, 3)` contributes six entries to the flat vector. Reconstruction
uses the reference model to restore each leaf shape. `vmap` usually adds a batch axis
outside an already-defined model computation; it does not redefine the meaning of a
single model leaf.

:::{warning} A PyTree is structure, not semantics
JAX can flatten two trees with matching shapes even when their leaves represent
different units or parameters. Scientific meaning still requires an explicit model
contract outside the generic PyTree mechanism.
:::

The current evidence supports structural preservation and gradient flow for selected
leaves. It does not provide automatic unit checking, state-version migration,
distributed checkpointing, or validation of domain-specific cached state.
